from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import re
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.trainer import onpolicy_trainer

from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot
from love_letter.paths import MODEL_POOL_DIR, checkpoint_path

# Configuration
OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
STARTING_MODEL = checkpoint_path("ppo_love_letter_league_champion.pth")

os.makedirs(MODEL_POOL_DIR, exist_ok=True)

# ==========================================
# 1. ARCHITECTURE (Identique à ton champion)
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

    def forward(self, obs_tensor, state=None, info={}): # Ajout de info={}
        features = self.feature_extractor(obs_tensor)
        if state is None:
            state = torch.zeros(obs_tensor.shape[0], LATENT, device=obs_tensor.device)
        new_state = self.rnn(features, state)
        return new_state, new_state

class MaskedActor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, ACTION_DIM)

    def forward(self, obs_dict, state=None, info={}): # Ajout de info={}
        # Extraction robuste pour Tianshou Batch ou dict standard
        if hasattr(obs_dict, 'observation'):
            obs = obs_dict.observation
            mask = obs_dict.action_mask
        else:
            obs = obs_dict['observation']
            mask = obs_dict['action_mask']
        
        x = torch.as_tensor(obs, dtype=torch.float32)
        m = torch.as_tensor(mask, dtype=torch.bool)
        
        features, new_state = self.encoder(x, state)
        logits = self.head(features)
        logits = logits.masked_fill(~m, -1e9)
        return logits, new_state

# ==========================================
# 2. WRAPPER SELF-PLAY DYNAMIQUE
# ==========================================
class SelfPlayTopKWrapper(gym.Env):
    def __init__(self, pz_env, agent_id="player_0"):
        super().__init__()
        self.env = pz_env
        self.agent_id = agent_id
        self.observation_space = self.env.observation_space(agent_id)
        self.action_space = self.env.action_space(agent_id)
        
        self.teacher = HeuristicBot()
        self.opponents = [None, None] # Stockera les MaskedActor chargés
        self.opp_states = { "player_2": None, "player_3": None }
        
        self._refresh_opponents()

    def _refresh_opponents(self):
        """Trouve les 2 meilleurs modèles dans le pool et les charge."""
        files = glob.glob(os.path.join(MODEL_POOL_DIR, "model_score_*.pth"))
        if len(files) < 2:
            return # Pas assez de modèles pour le moment

        # On trie par score (extrait du nom du fichier)
        def get_score(f):
            match = re.search(r"model_score_([\d\.]+)", f)
            return float(match.group(1)) if match else 0.0
        
        sorted_files = sorted(files, key=get_score, reverse=True)
        top_2 = sorted_files[:2]

        for i, path in enumerate(top_2):
            enc = RecurrentEncoder()
            act = MaskedActor(enc)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            enc.load_state_dict(ckpt["encoder"])
            act.load_state_dict(ckpt["actor"])
            act.eval()
            self.opponents[i] = act
        # print(f" [Self-Play] Opposants mis à jour avec : {[os.path.basename(f) for f in top_2]}")

    def reset(self, seed=None, options=None):
        # On rafraîchit les adversaires potentiellement au reset (optionnel, toutes les X parties)
        if np.random.rand() < 0.05: self._refresh_opponents()
        
        self.env.reset(seed=seed, options=options)
        self.opp_states = { "player_2": None, "player_3": None }
        
        obs, rew, term, trunc, info = self._step_until_my_turn()
        while term or trunc:
            self.env.reset(seed=seed, options=options)
            self.opp_states = { "player_2": None, "player_3": None }
            obs, rew, term, trunc, info = self._step_until_my_turn()
        return obs, info

    def step(self, action):
        if self.env.terminations[self.agent_id] or self.env.truncations[self.agent_id]:
            self.env.step(None)
        else:
            self.env.step(action)
        return self._step_until_my_turn()

    def _step_until_my_turn(self):
        while True:
            agent = getattr(self.env, "agent_selection", None)
            if agent is None:
                return self.observation_space.sample(), 0.0, True, False, {}

            obs, reward, term, trunc, info = self.env.last()
            if agent == self.agent_id:
                return obs, reward, term, trunc, info

            if term or trunc:
                self.env.step(None)
            else:
                if agent == "player_1": # Le Professeur
                    act = self.teacher.choose_action(self.env, agent)
                else: # Self-Play (Player 2 & 3)
                    idx = 0 if agent == "player_2" else 1
                    if self.opponents[idx] is not None:
                        # Forward d'un clone
                        with torch.no_grad():
                            logits, new_s = self.opponents[idx](obs, self.opp_states[agent])
                            self.opp_states[agent] = new_s
                            act = torch.argmax(logits, dim=-1).item()
                    else:
                        # Fallback random si pas encore de modèles
                        v = np.where(obs["action_mask"] == 1)[0]
                        act = int(np.random.choice(v)) if len(v) else 0
                self.env.step(act)

# ==========================================
# 3. TRAINING SETUP
# ==========================================
def train_self_play():
    def make_env():
        return SelfPlayTopKWrapper(LoveLetterRLEnv(num_players=4))

    # Architecture standard
    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    critic = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ReLU(), nn.Linear(256, 1)) # Critic simple pour stabiliser
    # Note: On réutilise les classes du script précédent pour le PPO complet si besoin
    
    # On charge le champion
    print(f"Chargement du champion : {STARTING_MODEL}")
    ckpt = torch.load(STARTING_MODEL, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])

    # Initialisation du pool avec le champion actuel pour commencer
    first_path = os.path.join(MODEL_POOL_DIR, "model_score_0.94_epoch_0.pth")
    if not os.path.exists(first_path):
        torch.save({"encoder": encoder.state_dict(), "actor": actor.state_dict()}, first_path)

    # Paramètres PPO
    optim = torch.optim.AdamW(list(actor.parameters()) + list(encoder.parameters()), lr=3e-5)
    

    # On utilise la structure Tianshou simplifiée pour l'exemple
    policy = PPOPolicy(
        actor=actor, critic=nn.Linear(LATENT, 1), # Le critic doit matcher l'output de l'encoder
        optim=optim, dist_fn=lambda l: Categorical(logits=l),
        action_space=gym.spaces.Discrete(1000),
        discount_factor=0.99, gae_lambda=0.95,
    )

    # Redéfinition du critic pour qu'il soit compatible avec le RecurrentEncoder
    class CustomCritic(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.v = nn.Linear(LATENT, 1)
        def forward(self, obs, state=None, info={}):
            x = torch.as_tensor(obs.observation if hasattr(obs, 'observation') else obs['observation'], dtype=torch.float32)
            feat, _ = self.encoder(x, state)
            return self.v(feat).flatten()
    
    policy.critic = CustomCritic(encoder)

    train_envs = DummyVectorEnv([make_env for _ in range(4)])
    test_envs = DummyVectorEnv([make_env for _ in range(2)])

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(10000, 4))
    test_collector = Collector(policy, test_envs)

    def save_best_fn(epoch, env_step, gradient_step):
        pass # Géré dans save_checkpoint_fn

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # On évalue le reward actuel
        test_stats = test_collector.collect(n_episode=50)
        rew = test_stats['rew']
        path = os.path.join(MODEL_POOL_DIR, f"model_score_{rew:.2f}_epoch_{epoch}.pth")
        torch.save({"encoder": encoder.state_dict(), "actor": actor.state_dict()}, path)
        print(f" --- Nouveau modèle archivé : {path} (Score: {rew:.2f}) ---")
        return path
    
    policy.actor = actor
    policy.critic = CustomCritic(encoder)
    optim = torch.optim.AdamW(policy.parameters(), lr=3e-5)
    policy.optim = optim

    print("\n[DÉMARRAGE DU SELF-PLAY TOP-K]")
    onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=5000,
        repeat_per_collect=10,
        episode_per_test=50,
        batch_size=256,
        step_per_collect=1000,
        save_checkpoint_fn=save_checkpoint_fn,
    )

if __name__ == "__main__":
    train_self_play()
