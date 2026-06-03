"""
PPO fine-tuning depuis les poids BC.
Objectif : partir d'une policy déjà bonne (imitation) et l'améliorer via RL.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128

BC_CHECKPOINT = checkpoint_path("bc_best.pth")


# ==========================================
# ARCHITECTURE (identique à train_bc.py)
# ==========================================
class SharedEncoder(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = torch.as_tensor(obs.obs, dtype=torch.float32)
        return self.net(x)


class MaskedActor(nn.Module):
    def __init__(self, encoder, action_dim=ACTION_DIM, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, action_dim)

    def forward(self, obs, state=None, info={}):
        features = self.encoder(obs)
        logits = self.head(features)
        mask = torch.as_tensor(obs.mask, dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, state


class Critic(nn.Module):
    def __init__(self, encoder, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, 1)

    def forward(self, obs, state=None, info={}):
        features = self.encoder(obs)
        return self.head(features).flatten()


# ==========================================
# RANDOM POLICY
# ==========================================
class ConsistentRandomPolicy(RandomPolicy):
    def forward(self, batch, state=None, **kwargs):
        mask = batch.obs.mask
        actions = []
        for m in mask:
            valid = np.where(m == 1)[0]
            actions.append(np.random.choice(valid) if len(valid) else 0)
        return Batch(act=np.array(actions, dtype=np.int64))


# ==========================================
# ENV
# ==========================================
def make_env():
    return PettingZooEnv(LoveLetterRLEnv(num_players=4))


# ==========================================
# TRAINING
# ==========================================
def train_finetune():
    print("Initialisation de l'environnement...")
    env = make_env()

    # Hyperparamètres PPO (ajustés pour fine-tuning : LR plus bas, entropie plus basse)
    lr = 5e-5  # 6x plus bas que from-scratch — on ne veut pas détruire le BC
    gamma = 0.99
    epoch = 60
    step_per_epoch = 10000
    batch_size = 256

    # Instanciation
    encoder = SharedEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)

    # ===== CHARGEMENT DES POIDS BC =====
    print(f"Chargement des poids BC depuis {BC_CHECKPOINT}...")
    ckpt = torch.load(BC_CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    # Le critic du BC n'a pas été entraîné, on le laisse init random
    # (il va apprendre sa valeur rapidement au début du PPO)
    print("  Encoder et Actor chargés (Critic reste random)")

    all_params = list(set(
        list(actor.parameters()) + list(critic.parameters())
    ))
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
        eps_clip=0.1,          # clip plus serré en fine-tuning
        vf_coef=0.5,
        ent_coef=0.001,        # entropie basse — on ne veut pas explorer fort
        gae_lambda=0.95,
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    random_policy = ConsistentRandomPolicy()

    policies = MultiAgentPolicyManager(
        policies=[ppo_policy, random_policy, random_policy, random_policy],
        env=env,
    )

    train_envs = DummyVectorEnv([make_env for _ in range(8)])
    test_envs = DummyVectorEnv([make_env for _ in range(4)])

    train_collector = Collector(
        policies, train_envs, VectorReplayBuffer(20000, len(train_envs))
    )
    test_collector = Collector(policies, test_envs)

    # ===== ÉVAL AVANT TRAINING (baseline BC) =====
    print("\nÉval du BC (avant fine-tuning) sur 200 épisodes...")
    ppo_policy.eval()
    bc_result = test_collector.collect(n_episode=200)
    bc_reward = float(bc_result["rews"].mean()) if "rews" in bc_result else float(bc_result.get("rew", 0.0))
    print(f"  Baseline BC : mean_reward = {bc_reward:.3f}")
    ppo_policy.train()

    print("\nDémarrage du fine-tuning PPO...")

    result = onpolicy_trainer(
        policy=policies,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=5,  # moins de repeats en fine-tuning
        episode_per_test=100,
        batch_size=batch_size,
        step_per_collect=2000,
        stop_fn=lambda mean_rewards: mean_rewards >= 1.0,
    )

    print("\nFine-tuning terminé !")
    print(f"  Baseline BC : {bc_reward:.3f}")
    print(f"  Après PPO   : {result['best_reward']:.3f}")
    print(f"  Gain        : {result['best_reward'] - bc_reward:+.3f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }, checkpoint_path("ppo_finetuned_best.pth"))
    print(f"Modèle sauvegardé : {checkpoint_path('ppo_finetuned_best.pth')}")


if __name__ == "__main__":
    train_finetune()
