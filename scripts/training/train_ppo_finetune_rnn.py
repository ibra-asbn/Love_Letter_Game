from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.trainer import onpolicy_trainer

from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_OPPONENTS = 3
BC_CHECKPOINT = checkpoint_path("bc_rnn_best.pth")

# ==========================================
# 0. LE WRAPPER "ENVIRONMENT VS BOT" (CORRIGÉ)
# ==========================================
class SingleAgentWrapper(gym.Env):
    """
    Transforme l'environnement multi-agents en un environnement single-agent pour Tianshou.
    Gère silencieusement les éliminations avant le premier tour.
    """
    def __init__(self, pz_env, agent_id="player_0"):
        super().__init__()
        self.env = pz_env
        self.agent_id = agent_id
        self.observation_space = self.env.observation_space(agent_id)
        self.action_space = self.env.action_space(agent_id)

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        obs, reward, term, trunc, info = self._step_until_my_turn()
        
        # SÉCURITÉ 1 : Si le bot a été tué par un adversaire avant même de pouvoir 
        # jouer son premier tour, Tianshou va crasher. On relance une partie silencieusement.
        while term or trunc:
            self.env.reset(seed=seed, options=options)
            obs, reward, term, trunc, info = self._step_until_my_turn()
            
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.last()
        
        # SÉCURITÉ 2 : Si on est déjà mort, PettingZoo n'accepte QUE l'action "None"
        if term or trunc:
            self.env.step(None)
        else:
            self.env.step(action)
            
        return self._step_until_my_turn()

    def _step_until_my_turn(self):
        while True:
            # Sécurité PettingZoo : si l'agent selection est vide, la partie est finie
            if getattr(self.env, "agent_selection", None) is None:
                # Renvoie un état par défaut pour ne pas bloquer
                return self.observation_space.sample(), 0.0, True, False, {}

            agent = self.env.agent_selection
            obs, reward, term, trunc, info = self.env.last()

            # Si c'est notre tour (ou si la partie est finie et qu'on doit récupérer le score final)
            if agent == self.agent_id:
                return obs, reward, term, trunc, info

            # Si c'est au tour d'un adversaire
            if term or trunc:
                self.env.step(None)
            else:
                mask = obs["action_mask"]
                valid = np.where(mask == 1)[0]
                act = int(np.random.choice(valid)) if len(valid) else 0
                self.env.step(act)


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

    def forward(self, obs, state=None, info={}):
        # CORRECTION : Tianshou utilise le nom exact des clés de l'environnement Gym ("observation")
        if hasattr(obs, 'observation'):
            x_val = obs.observation
        elif isinstance(obs, dict):
            x_val = obs['observation']
        else:
            x_val = obs  # Fallback de sécurité

        # On sécurise aussi le placement sur le bon device (CPU/GPU)
        x = torch.as_tensor(x_val, dtype=torch.float32, device=next(self.parameters()).device)
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

    def forward(self, obs, state=None, info={}):
        features, state = self.encoder(obs, state, info)
        logits = self.head(features)
        
        # CORRECTION : On utilise la clé "action_mask"
        if hasattr(obs, 'action_mask'):
            mask_val = obs.action_mask
        elif isinstance(obs, dict):
            mask_val = obs['action_mask']
        else:
            mask_val = obs
            
        mask = torch.as_tensor(mask_val, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, state

class Critic(nn.Module):
    def __init__(self, encoder, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, 1)

    def forward(self, obs, state=None, info={}):
        features, _ = self.encoder(obs, state, info)
        return self.head(features).flatten()

class BeliefHead(nn.Module):
    def __init__(self, encoder, num_opponents=NUM_OPPONENTS, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(latent, num_opponents * 10)

    def forward(self, obs, state=None, info={}):
        features, _ = self.encoder(obs, state, info)
        logits = self.head(features)
        return logits.view(-1, self.num_opponents, 10)


# ==========================================
# 2. PPO AVEC LOSS AUXILIAIRE BELIEF
# ==========================================
class AuxPPOPolicy(PPOPolicy):
    def __init__(self, belief_head, belief_coef=0.05, **kwargs):
        super().__init__(**kwargs)
        self.belief_head = belief_head
        self.belief_coef = belief_coef
        self.belief_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def learn(self, batch, batch_size, repeat, **kwargs):
        res = super().learn(batch, batch_size, repeat, **kwargs)

        device = next(self.belief_head.parameters()).device
        state = batch.state if hasattr(batch, 'state') else None

        belief_logits = self.belief_head(batch.obs, state)
        hidden_cards = torch.as_tensor(batch.info.hidden_cards, dtype=torch.long, device=device)

        B = belief_logits.shape[0]
        loss_belief = self.belief_loss_fn(
            belief_logits.reshape(B * self.belief_head.num_opponents, 10),
            hidden_cards.reshape(B * self.belief_head.num_opponents),
        )

        self.optim.zero_grad()
        (self.belief_coef * loss_belief).backward()
        self.optim.step()

        with torch.no_grad():
            preds = belief_logits.argmax(dim=-1)
            valid_mask = hidden_cards != -1
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == hidden_cards[valid_mask]).float().mean().item()
            else:
                acc = 0.0

        res["loss/belief"] = loss_belief.item()
        res["belief/accuracy"] = acc
        return res


# ==========================================
# 3. ENV & TRAINING
# ==========================================
def make_env():
    pz_env = LoveLetterRLEnv(num_players=4)
    return SingleAgentWrapper(pz_env, agent_id="player_0")


def train_agent():
    print("Initialisation de l'environnement...")
    # On utilise juste le make_env pour récupérer les specs
    dummy_env = make_env()

    lr = 1e-4
    gamma = 0.99
    epoch = 80
    step_per_epoch = 10000
    batch_size = 256

    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)
    belief_head = BeliefHead(encoder)

    # Chargement BC
    if os.path.exists(BC_CHECKPOINT):
        print(f"Chargement de {BC_CHECKPOINT}...")
        ckpt = torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
        belief_head.load_state_dict(ckpt["belief_head"])
    else:
        print(f"ATTENTION: {BC_CHECKPOINT} introuvable. Entraînement from scratch.")

    # Geler l'encodeur
    for p in encoder.parameters():
        p.requires_grad = False

    all_params = list(set(
        list(actor.parameters()) + list(critic.parameters()) + list(belief_head.parameters())
    ))
    optim = torch.optim.AdamW(all_params, lr=lr)

    def dist_fn(logits):
        return Categorical(logits=logits)

    ppo_policy = AuxPPOPolicy(
        belief_head=belief_head,
        belief_coef=0.05,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=gamma,
        action_space=dummy_env.action_space,
        action_scaling=False,
        action_bound_method="",
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    # Plus de MultiAgentPolicyManager ! Tianshou voit une simple policy
    train_envs = DummyVectorEnv([make_env for _ in range(8)])
    test_envs = DummyVectorEnv([make_env for _ in range(4)])

    train_collector = Collector(
        ppo_policy, train_envs, VectorReplayBuffer(20000, len(train_envs))
    )
    test_collector = Collector(ppo_policy, test_envs)

    print("Démarrage de l'entraînement PPO sur le modèle RNN...")

    result = onpolicy_trainer(
        policy=ppo_policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=10,
        episode_per_test=100,
        batch_size=batch_size,
        step_per_collect=2000,
        stop_fn=lambda mean_rewards: mean_rewards >= 0.80,
    )

    print("\nEntraînement terminé !")
    print(f"Meilleur reward : {result.get('best_reward', 0):.2f}")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "belief_head": belief_head.state_dict(),
        },
        checkpoint_path("ppo_love_letter_belief_rnn_final.pth"),
    )
    print(f"Modèle sauvegardé : {checkpoint_path('ppo_love_letter_belief_rnn_final.pth')}")

if __name__ == "__main__":
    train_agent()
