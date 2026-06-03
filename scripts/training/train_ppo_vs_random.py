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
NUM_OPPONENTS = 3


# ==========================================
# 1. RÉSEAU
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

    def forward_from_tensor(self, x):
        return self.net(x)

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


class BeliefHead(nn.Module):
    """ Prédit la carte en main de chaque adversaire. """
    def __init__(self, encoder, num_opponents=NUM_OPPONENTS, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(latent, num_opponents * 10)

    def forward(self, obs):
        features = self.encoder(obs)
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
        # 1. PPO standard
        res = super().learn(batch, batch_size, repeat, **kwargs)

        # 2. Belief loss — un seul passage, après le PPO
        device = next(self.belief_head.parameters()).device

        # batch.obs est un Batch avec .obs (111,) et .mask (1000,) par sample
        # On doit le reconstruire en tenseur pour le belief head
        obs_tensor = torch.as_tensor(batch.obs.obs, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(batch.obs.mask, dtype=torch.int8, device=device)
        obs_as_batch = Batch(obs=obs_tensor, mask=mask_tensor)

        # Target : hidden_cards stocké dans info par observe()
        hidden_cards = torch.as_tensor(
            batch.info.hidden_cards, dtype=torch.long, device=device
        )  # shape: (batch_size, 3)

        # Forward belief
        belief_logits = self.belief_head(obs_as_batch)  # shape: (B, 3, 10)

        # Flatten pour CrossEntropy
        B = belief_logits.shape[0]
        loss_belief = self.belief_loss_fn(
            belief_logits.reshape(B * self.belief_head.num_opponents, 10),
            hidden_cards.reshape(B * self.belief_head.num_opponents),
        )

        self.optim.zero_grad()
        (self.belief_coef * loss_belief).backward()
        self.optim.step()

        # Accuracy pour debug
        with torch.no_grad():
            preds = belief_logits.argmax(dim=-1)
            valid_mask = hidden_cards != -1
            if valid_mask.sum() > 0:
                acc = (preds[valid_mask] == hidden_cards[valid_mask]).float().mean().item()
            else:
                acc = 0.0

        res["loss/belief"] = [loss_belief.item()]
        res["belief/accuracy"] = [acc]
        return res


# ==========================================
# 3. RANDOM POLICY
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
# 4. ENV & TRAINING
# ==========================================
def make_env():
    return PettingZooEnv(LoveLetterRLEnv(num_players=4))


def train_agent():
    print("Initialisation de l'environnement...")
    env = make_env()

    lr = 3e-4
    gamma = 0.99
    epoch = 80
    step_per_epoch = 10000
    batch_size = 256

    encoder = SharedEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)
    belief_head = BeliefHead(encoder)

    all_params = list(set(
        list(actor.parameters())
        + list(critic.parameters())
        + list(belief_head.parameters())
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
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method="",
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

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "belief_head": belief_head.state_dict(),
        },
        checkpoint_path("ppo_love_letter_belief_v1.pth"),
    )
    print(f"Modèle sauvegardé : {checkpoint_path('ppo_love_letter_belief_v1.pth')}")


if __name__ == "__main__":
    train_agent()
