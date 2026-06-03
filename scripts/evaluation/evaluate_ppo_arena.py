"""
L'Arène Love Letter : Évaluation du modèle PPO (RNN) vs Configurations Asymétriques.
"""

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np

from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot
from love_letter.belief_policy import load_belief_policy
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
DEFAULT_CHECKPOINT = checkpoint_path("curriculum_phase1.pth")
# ==========================================
# 1. ARCHITECTURE EXACTE (Pour charger les poids)
# ==========================================
class RecurrentEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_size=LATENT):
        super().__init__()
        # CORRECTION : On renomme "net" en "feature_extractor"
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, hidden_size),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x, h_in):
        # CORRECTION : On utilise "feature_extractor" au lieu de "net"
        features = self.feature_extractor(x)
        return features, self.rnn(features, h_in)

class MaskedActor(nn.Module):
    def __init__(self, encoder, action_dim=ACTION_DIM):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, action_dim)

    def forward(self, obs, state=None, info={}):
        device = next(self.head.parameters()).device
        
        x = torch.as_tensor(obs.obs, dtype=torch.float32, device=device)
        h_in = torch.zeros(x.shape[0], LATENT, device=device) if state is None else torch.as_tensor(state, dtype=torch.float32, device=device)
        
        if h_in.shape[0] != x.shape[0]: 
            h_in = torch.zeros(x.shape[0], LATENT, device=device)
            
        _, features = self.encoder(x, h_in)
        new_state = features.detach().cpu().numpy()

        logits = self.head(features)
        mask = torch.as_tensor(obs.mask, dtype=torch.bool, device=device)
        logits = logits.masked_fill(~mask, -1e9)
        
        return logits, new_state

# ==========================================
# 2. MOTEUR D'ÉVALUATION EN ARÈNE
# ==========================================
def evaluate_arena(actor, opponents_config, n_games=100):
    """
    Fait jouer l'agent PPO (player_0) contre une configuration d'adversaires spécifique.
    opponents_config est une liste de 3 strings : ex ["random", "random", "heuristic"]
    """
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    
    wins = 0
    total_rewards = 0.0

    for game in range(n_games):
        env.reset()
        ppo_state = None  # La mémoire RNN est vidée au début de chaque partie
        game_reward = 0.0

        for agent in env.agent_iter():
            obs_dict, reward, term, trunc, info = env.last()

            # Seul le PPO (player_0) nous intéresse pour le score
            if agent == "player_0":
                game_reward += reward

            if term or trunc:
                env.step(None)
                continue

            # --- TOUR DU PPO ---
            if agent == "player_0":
                action, ppo_state = actor.act(obs_dict, ppo_state, agent_id=agent)

            # --- TOUR DES ADVERSAIRES ---
            else:
                # Identification de l'adversaire (player_1 -> index 0, player_2 -> index 1...)
                opp_idx = int(agent[-1]) - 1
                opp_type = opponents_config[opp_idx]

                if opp_type == "random":
                    mask = obs_dict["action_mask"]
                    valid = np.where(mask == 1)[0]
                    action = int(np.random.choice(valid)) if len(valid) else 0
                elif opp_type == "heuristic":
                    action = bot.choose_action(env, agent)

            env.step(action)

        # Fin de la partie
        total_rewards += game_reward
        if game_reward >= 1.0:
            wins += 1

    return wins / n_games, total_rewards / n_games

# ==========================================
# 3. LANCEMENT DU TOURNOI
# ==========================================
def run_tournament(checkpoint=DEFAULT_CHECKPOINT, games=100):
    device = torch.device("cpu")
    print("\n=== PRÉPARATION DE L'ARÈNE ===")
    
    print(f"Chargement de {checkpoint}...")
    try:
        actor = load_belief_policy(checkpoint)
        print("Modèle chargé avec succès !\n")
    except Exception as e:
        print(f"Erreur de chargement : {e}")
        return

    # Configuration des combats
    phases = [
        {"name": "PPO vs 3 Randoms", "config": ["random", "random", "random"], "games": games},
        {"name": "PPO vs 2 Randoms + 1 Heuristic", "config": ["heuristic", "random", "random"], "games": games},
        {"name": "PPO vs 1 Random + 2 Heuristics", "config": ["heuristic", "heuristic", "random"], "games": games},
        {"name": "PPO vs 3 Heuristics", "config": ["heuristic", "heuristic", "heuristic"], "games": games},
    ]

    print("============================================")
    print(f"DÉBUT DU TOURNOI ({games} parties par configuration)")
    print("============================================\n")

    for phase in phases:
        print(f"Combats en cours : {phase['name']} ...")
        winrate, mean_reward = evaluate_arena(actor, phase["config"], n_games=phase["games"])
        print(f"  -> Winrate : {winrate * 100:.1f}%")
        print(f"  -> Mean Reward : {mean_reward:.3f}\n")

    print("============================================")
    print("TOURNOI TERMINÉ")
    print("============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un checkpoint PPO en arène Love Letter.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Chemin du checkpoint à évaluer.")
    parser.add_argument("--games", type=int, default=100, help="Nombre de parties par configuration.")
    args = parser.parse_args()
    run_tournament(checkpoint=args.checkpoint, games=args.games)
