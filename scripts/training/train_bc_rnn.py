"""
Behavioral Cloning Séquentiel (BC + RNN) : 
Apprend à imiter l'heuristique tout en forgeant une mémoire cellulaire (GRU) 
pour déduire les cartes cachées (Belief Head).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot # Assure-toi que cette classe est accessible
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_OPPONENTS = 3

# ==========================================
# 1. ARCHITECTURE RÉCURRENTE (SOTA)
# ==========================================
class RecurrentEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_size=LATENT):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward_tensor(self, obs_t, h_in):
        """ Forward pur PyTorch pour la phase de BPTT (Supervisé) """
        features = self.feature_extractor(obs_t)
        h_out = self.rnn(features, h_in)
        return h_out

    def forward(self, obs, state=None, info={}):
        """ Forward formaté pour Tianshou (Pour le futur PPO) """
        x = torch.as_tensor(obs.obs, dtype=torch.float32)
        features = self.feature_extractor(x)
        
        if state is None or (hasattr(state, 'is_empty') and state.is_empty()):
            h_in = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        else:
            if isinstance(state, dict) and 'hidden' in state:
                h_in = state['hidden']
            elif hasattr(state, 'hidden'):
                h_in = state.hidden
            else:
                h_in = state
            h_in = torch.as_tensor(h_in, dtype=torch.float32, device=x.device)
            
        if h_in.shape[0] != x.shape[0]:
            h_in = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            
        new_state = self.rnn(features, h_in)
        return new_state, {"hidden": new_state.detach().cpu().numpy()}

class MaskedActor(nn.Module):
    def __init__(self, encoder, action_dim=ACTION_DIM, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, action_dim)

    def forward_tensor(self, features, mask_t):
        logits = self.head(features)
        logits = logits.masked_fill(~mask_t, -1e9)
        return logits

class BeliefHead(nn.Module):
    def __init__(self, encoder, num_opponents=NUM_OPPONENTS, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(latent, num_opponents * 10)

    def forward_tensor(self, features):
        logits = self.head(features)
        return logits.view(-1, self.num_opponents, 10)

class Critic(nn.Module):
    """ Instancié uniquement pour sauvegarder l'architecture complète """
    def __init__(self, encoder, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, 1)

# ==========================================
# 2. GÉNÉRATION DE DONNÉES SÉQUENTIELLES
# ==========================================
def collect_trajectories(n_games=5000):
    """
    Joue des parties (Heuristic vs Random) et sauvegarde des séquences 
    complètes (épisodes) pour entraîner le RNN.
    """
    print(f"Génération de {n_games} parties séquentielles (Trajectoires)...")
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    
    episodes = []
    
    for _ in range(n_games):
        env.reset()
        current_ep = {"obs": [], "mask": [], "action": [], "hidden_cards": []}
        
        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue
                
            # On ne collecte que le point de vue du bot expert (player_0)
            if agent == "player_0":
                # L'heuristique donne son action (Adapte la méthode selon ton bot)
                # Si ton bot utilise (obs, mask), remplace par bot.act(obs_dict["observation"], obs_dict["action_mask"])
                action = bot.choose_action(env, agent)
                
                current_ep["obs"].append(obs_dict["observation"])
                current_ep["mask"].append(obs_dict["action_mask"])
                current_ep["action"].append(action)
                # Récupération de la vérité terrain générée secrètement par l'Env
                current_ep["hidden_cards"].append(env.infos[agent].get("hidden_cards", np.full(3, -1)))
            else:
                # Les adversaires jouent aléatoirement
                valid = np.where(obs_dict["action_mask"] == 1)[0]
                action = int(np.random.choice(valid)) if len(valid) else 0
                
            env.step(action)
            
        if len(current_ep["obs"]) > 0:
            episodes.append(current_ep)
            
    return episodes

# ==========================================
# 3. DATASET & PADDING
# ==========================================
class SequentialDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        
    def __len__(self):
        return len(self.episodes)
        
    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return (
            torch.tensor(np.array(ep["obs"]), dtype=torch.float32),
            torch.tensor(np.array(ep["mask"]), dtype=torch.bool),
            torch.tensor(np.array(ep["action"]), dtype=torch.long),
            torch.tensor(np.array(ep["hidden_cards"]), dtype=torch.long),
        )

def pad_collate(batch):
    """
    Prend un batch de trajectoires de longueurs variables et les aligne 
    avec du padding (zéros) pour passer dans le RNN matriciellement.
    """
    obs_list, mask_list, act_list, hidden_list = zip(*batch)
    lengths = torch.tensor([len(x) for x in obs_list], dtype=torch.long)
    
    obs_pad = torch.nn.utils.rnn.pad_sequence(obs_list, batch_first=True)
    mask_pad = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
    act_pad = torch.nn.utils.rnn.pad_sequence(act_list, batch_first=True, padding_value=0)
    hidden_pad = torch.nn.utils.rnn.pad_sequence(hidden_list, batch_first=True, padding_value=-1)
    
    return obs_pad, mask_pad, act_pad, hidden_pad, lengths

# ==========================================
# 4. TRAINING LOOP SÉQUENTIELLE (BPTT)
# ==========================================
def train_bc_rnn():
    device = torch.device("cpu")
    
    # 1. Génération dynamique (Plus de chargement de pkl plat)
    episodes = collect_trajectories(n_games=10000) # Monte à 20k-30k si tu as la puissance
    
    # Split
    n_val = len(episodes) // 10
    train_eps = episodes[:-n_val]
    val_eps = episodes[-n_val:]
    
    train_loader = DataLoader(SequentialDataset(train_eps), batch_size=64, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(SequentialDataset(val_eps), batch_size=64, shuffle=False, collate_fn=pad_collate)

    # 2. Initialisation des composants à mémoire
    encoder = RecurrentEncoder().to(device)
    actor = MaskedActor(encoder).to(device)
    belief_head = BeliefHead(encoder).to(device)
    critic = Critic(encoder).to(device) # Juste pour la sauvegarde

    # On entraîne l'encodeur, l'acteur ET la tête de déduction
    params = list(set(list(encoder.parameters()) + list(actor.parameters()) + list(belief_head.parameters())))
    optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)

    loss_actor_fn = nn.CrossEntropyLoss(reduction='none') 
    loss_belief_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    print("\nDémarrage du Behavioral Cloning Séquentiel (BPTT)...")
    epochs = 40
    
    for epoch in range(epochs):
        encoder.train()
        actor.train()
        belief_head.train()
        
        total_act_loss, total_bel_loss = 0.0, 0.0
        total_correct_act, total_correct_bel = 0, 0
        total_steps, total_valid_cards = 0, 0

        for obs_b, mask_b, act_b, hidden_b, lengths in train_loader:
            B, T, _ = obs_b.shape
            
            # Initialisation de l'état caché (Mémoire vide au début de la manche)
            h_t = torch.zeros(B, LATENT, device=device)
            
            batch_act_loss = 0.0
            batch_bel_loss = 0.0
            
            # BPTT : Boucle temporelle (On avance tour par tour)
            for t in range(T):
                # On ne calcule les gradients que pour les parties qui ne sont pas finies (padding)
                valid_mask = (t < lengths).float().to(device)
                if valid_mask.sum() == 0:
                    break
                
                # Forward de ce tour
                h_t = encoder.forward_tensor(obs_b[:, t, :], h_t)
                
                # 1. Tâche Principale : Deviner l'action (Actor)
                logits = actor.forward_tensor(h_t, mask_b[:, t, :])
                act_loss = loss_actor_fn(logits, act_b[:, t])
                batch_act_loss += (act_loss * valid_mask).sum()
                
                preds_act = logits.argmax(dim=-1)
                total_correct_act += ((preds_act == act_b[:, t]).float() * valid_mask).sum().item()
                
                # 2. Tâche Auxiliaire : Deviner les cartes (Belief Head)
                # 2. Tâche Auxiliaire : Deviner les cartes (Belief Head)
                bel_logits = belief_head.forward_tensor(h_t) # Shape (B, 3, 10)
                bel_loss = loss_belief_fn(
                    bel_logits.reshape(B * NUM_OPPONENTS, 10), 
                    hidden_b[:, t, :].reshape(B * NUM_OPPONENTS)
                )
                
                # On remet à la dimension du batch pour appliquer le mask temporel (valid_mask)
                bel_loss = bel_loss.view(B, NUM_OPPONENTS).mean(dim=1) 
                batch_bel_loss += (bel_loss * valid_mask).sum()
                
                preds_bel = bel_logits.argmax(dim=-1)
                valid_cards_mask = (hidden_b[:, t, :] != -1) & valid_mask.unsqueeze(1).bool()
                total_correct_bel += (preds_bel[valid_cards_mask] == hidden_b[:, t, :][valid_cards_mask]).sum().item()
                total_valid_cards += valid_cards_mask.sum().item()
                
                total_steps += valid_mask.sum().item()

            # Moyenne sur le nombre d'étapes réelles du batch
            loss = (batch_act_loss + (0.5 * batch_bel_loss)) / valid_mask.sum()
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0) # Sécure le GRU contre l'explosion de gradient
            optim.step()

            total_act_loss += batch_act_loss.item()
            total_bel_loss += batch_bel_loss.item()

        # --- Logs de l'Epoch ---
        avg_act_loss = total_act_loss / total_steps
        avg_bel_loss = total_bel_loss / total_steps
        acc_act = total_correct_act / total_steps
        acc_bel = total_correct_bel / max(1, total_valid_cards)

        print(f"Epoch {epoch+1:2d} | ActLoss: {avg_act_loss:.3f} BelLoss: {avg_bel_loss:.3f} | "
              f"ActAcc: {acc_act:.3f} BelAcc: {acc_bel:.3f}")

    # Sauvegarde finale
    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "belief_head": belief_head.state_dict(),
    }, checkpoint_path("bc_rnn_best.pth"))
    print(f"\nModèle BC Séquentiel sauvegardé : {checkpoint_path('bc_rnn_best.pth')}")
    print("Prêt pour la Phase 2 : PPO Fine-Tuning !")

if __name__ == "__main__":
    train_bc_rnn()
