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
LATENT = 128
NUM_OPPONENTS = 3


# ==========================================
# 1. RÉSEAU (inchangé)
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


class BeliefHead(nn.Module):
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
# 2. PPO + BELIEF (inchangé)
# ==========================================
class AuxPPOPolicy(PPOPolicy):
    def __init__(self, belief_head, belief_coef=0.5, **kwargs):
        super().__init__(**kwargs)
        self.belief_head = belief_head
        self.belief_coef = belief_coef
        self.belief_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def learn(self, batch, batch_size, repeat, **kwargs):
        res = super().learn(batch, batch_size, repeat, **kwargs)

        device = next(self.belief_head.parameters()).device
        obs_tensor = torch.as_tensor(batch.obs.obs, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(batch.obs.mask, dtype=torch.int8, device=device)
        obs_as_batch = Batch(obs=obs_tensor, mask=mask_tensor)

        hidden_cards = torch.as_tensor(
            batch.info.hidden_cards, dtype=torch.long, device=device
        )

        belief_logits = self.belief_head(obs_as_batch)
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
# 4. ENV
# ==========================================
def make_env():
    return PettingZooEnv(LoveLetterRLEnv(num_players=4))


# ==========================================
# 5. ÉVALUATION vs RANDOM (callback manuel)
# ==========================================
def evaluate_vs_random(ppo_policy, n_episodes=200):
    """Mesure le winrate de ppo_policy (en seat 0) contre 3 random."""
    eval_env = make_env()
    random_policy = ConsistentRandomPolicy()
    eval_policies = MultiAgentPolicyManager(
        policies=[ppo_policy, random_policy, random_policy, random_policy],
        env=eval_env,
    )
    eval_envs = DummyVectorEnv([make_env for _ in range(4)])
    eval_collector = Collector(eval_policies, eval_envs)

    # Passe en mode eval (déterministe)
    was_training = ppo_policy.training
    ppo_policy.eval()

    result = eval_collector.collect(n_episode=n_episodes)

    if was_training:
        ppo_policy.train()

    # rews est un array des rewards cumulés par épisode pour TOUS les agents.
    # Comme Tianshou en MARL retourne une moyenne, on récupère via result.
    mean_rew = result.get("rews", result.get("rew", 0.0))
    if hasattr(mean_rew, "mean"):
        mean_rew = float(mean_rew.mean())
    else:
        mean_rew = float(mean_rew)

    return mean_rew


# ==========================================
# 6. ENTRAÎNEMENT SELF-PLAY
# ==========================================
def train_agent():
    print("Initialisation de l'environnement...")
    env = make_env()

    lr = 3e-4
    gamma = 0.99
    epoch = 100
    step_per_epoch = 10000
    batch_size = 256
    eval_every = 10  # éval vs random tous les 10 epochs

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
        belief_coef=0.5,
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
        ent_coef=0.005,  # réduit pour self-play
        gae_lambda=0.95,
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    # SELF-PLAY : le même PPO contrôle les 4 seats
    # Tianshou détecte que c'est le même objet et ne fait qu'une update par step.
    policies = MultiAgentPolicyManager(
        policies=[ppo_policy, ppo_policy, ppo_policy, ppo_policy],
        env=env,
    )

    train_envs = DummyVectorEnv([make_env for _ in range(8)])
    test_envs = DummyVectorEnv([make_env for _ in range(4)])

    train_collector = Collector(
        policies, train_envs, VectorReplayBuffer(20000, len(train_envs))
    )
    test_collector = Collector(policies, test_envs)

    print("Démarrage de l'entraînement self-play...")
    print("(rew ~0.25 est NORMAL en self-play — voir winrate vs random)\n")

    # Wrapper de save/reward qui fait l'éval vs random périodiquement
    best_wr_vs_random = [0.0]

    def save_best_fn(policy):
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "belief_head": belief_head.state_dict(),
            },
            checkpoint_path("ppo_love_letter_selfplay_best.pth"),
        )

    def train_fn(epoch_idx, env_step):
        # Éval vs random tous les `eval_every` epochs
        if epoch_idx % eval_every == 0 and epoch_idx > 0:
            wr = evaluate_vs_random(ppo_policy, n_episodes=200)
            print(f"\n  >>> [Epoch {epoch_idx}] Winrate vs 3 Random: {wr:.3f}"
                  f" (best: {best_wr_vs_random[0]:.3f})")
            if wr > best_wr_vs_random[0]:
                best_wr_vs_random[0] = wr
                save_best_fn(None)
                print(f"  >>> Nouveau meilleur modèle sauvegardé\n")
            else:
                print()

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
        train_fn=train_fn,
    )

    print("\nEntraînement terminé !")
    print(f"Meilleur winrate vs random : {best_wr_vs_random[0]:.3f}")

    # Éval finale
    final_wr = evaluate_vs_random(ppo_policy, n_episodes=500)
    print(f"Winrate final (500 parties) : {final_wr:.3f}")

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "belief_head": belief_head.state_dict(),
        },
        checkpoint_path("ppo_love_letter_selfplay_final.pth"),
    )
    print(f"Modèle final sauvegardé : {checkpoint_path('ppo_love_letter_selfplay_final.pth')}")


if __name__ == "__main__":
    train_agent()
