from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.policy import MultiAgentPolicyManager, PPOPolicy, RandomPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.trainer import onpolicy_trainer

from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path

OBS_DIM = 111
ACTION_DIM = 1000
HIDDEN = 256


# ==========================================
# 1. RÉSEAU (FEEDFORWARD PARTAGÉ)
# ==========================================
class SharedEncoder(nn.Module):
    """ Tronc commun feedforward. """
    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = torch.as_tensor(obs.obs, dtype=torch.float32)
        return self.net(x)


class MaskedActor(nn.Module):
    """ Tête politique avec action masking. """
    def __init__(self, encoder, action_dim=ACTION_DIM):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, action_dim)

    def forward(self, obs, state=None, info={}):
        features = self.encoder(obs)
        logits = self.head(features)
        mask = torch.as_tensor(obs.mask, dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, state


class Critic(nn.Module):
    """ Tête valeur. """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(128, 1)

    def forward(self, obs, state=None, info={}):
        features = self.encoder(obs)
        return self.head(features).flatten()


# ==========================================
# 2. RANDOM POLICY COMPATIBLE PPO
# ==========================================
class ConsistentRandomPolicy(RandomPolicy):
    """
    Politique aléatoire avec action masking.
    Retourne uniquement `act` pour éviter tout conflit avec le buffer MARL.
    """
    def forward(self, batch, state=None, **kwargs):
        mask = batch.obs.mask
        actions = []
        for m in mask:
            valid = np.where(m == 1)[0]
            actions.append(np.random.choice(valid) if len(valid) else 0)
        return Batch(act=np.array(actions, dtype=np.int64))


# ==========================================
# 3. ENVIRONNEMENT
# ==========================================
def make_env():
    return PettingZooEnv(LoveLetterRLEnv(num_players=4))


# ==========================================
# 4. ENTRAÎNEMENT
# ==========================================
def train_agent():
    print("Initialisation de l'environnement...")
    env = make_env()

    # Hyperparamètres PPO
    lr = 3e-4
    gamma = 0.99
    epoch = 50
    step_per_epoch = 10000
    batch_size = 256

    encoder = SharedEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)

    all_params = list(set(list(actor.parameters()) + list(critic.parameters())))
    optim = torch.optim.AdamW(all_params, lr=lr)

    def dist_fn(logits):
        return Categorical(logits=logits)

    ppo_policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=gamma,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method="",
        # hyperparams PPO standard
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    random_policy = ConsistentRandomPolicy()

    policies = MultiAgentPolicyManager(
        policies=[ppo_policy, random_policy, random_policy, random_policy],
        env=env,
    )

    # Collecteurs
    train_envs = DummyVectorEnv([make_env for _ in range(8)])
    test_envs = DummyVectorEnv([make_env for _ in range(4)])

    train_collector = Collector(
        policies, train_envs, VectorReplayBuffer(20000, len(train_envs))
    )
    test_collector = Collector(policies, test_envs)

    print("Démarrage de l'entraînement...")

    result = onpolicy_trainer(
        policy=policies,
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
    print(f"Meilleur reward : {result['best_reward']:.2f}")

    torch.save(ppo_policy.state_dict(), checkpoint_path("ppo_love_letter_v1.pth"))
    print(f"Modèle sauvegardé : {checkpoint_path('ppo_love_letter_v1.pth')}")


if __name__ == "__main__":
    train_agent()
