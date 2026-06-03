"""
PPO fine-tuning depuis BC — version SOTA avec :
- Phase 0 : Critic warm-up natif Tianshou (Gradients actor coupés)
- Phase 1 : Fine-tune heads seulement (encoder gelé)
- Phase 2 : Fine-tune tout (LR très bas)
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
# ARCHITECTURE
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
    def __init__(self, encoder, num_opponents=3, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(latent, num_opponents * 10)

    def forward(self, obs):
        features = self.encoder(obs)
        logits = self.head(features)
        return logits.view(-1, self.num_opponents, 10)


# ==========================================
# 2. PPO CUSTOM AVEC WARMUP MODE
# ==========================================
class AuxPPOPolicy(PPOPolicy):
    def __init__(self, belief_head, belief_coef=0.5, **kwargs):
        super().__init__(**kwargs)
        self.belief_head = belief_head
        self.belief_coef = belief_coef
        self.belief_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.warmup_mode = False # <-- FLAG SOTA

    def learn(self, batch, batch_size, repeat, **kwargs):
        # --- MODE WARMUP : Mise à jour stricte du Critic ---
        if self.warmup_mode:
            losses = []
            for _ in range(repeat):
                for minibatch in batch.split(batch_size, merge_last=True):
                    # Tianshou a déjà calculé les retours (GAE) dans process_fn
                    value = self.critic(minibatch.obs).flatten()
                    returns = minibatch.returns.flatten()
                    vf_loss = ((value - returns) ** 2).mean()

                    self.optim.zero_grad()
                    vf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optim.step()
                    losses.append(vf_loss.item())
            return {"loss/vf": [np.mean(losses)]}

        # --- MODE NORMAL (PPO + Belief) ---
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
# HELPERS
# ==========================================
class ConsistentRandomPolicy(RandomPolicy):
    def forward(self, batch, state=None, **kwargs):
        mask = batch.obs.mask
        actions = []
        for m in mask:
            valid = np.where(m == 1)[0]
            actions.append(np.random.choice(valid) if len(valid) else 0)
        return Batch(act=np.array(actions, dtype=np.int64))

def make_env():
    return PettingZooEnv(LoveLetterRLEnv(num_players=4))

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def evaluate(ppo_policy, n_episode=300):
    env = make_env()
    random_pol = ConsistentRandomPolicy()
    policies = MultiAgentPolicyManager(policies=[ppo_policy, random_pol, random_pol, random_pol], env=env)
    envs = DummyVectorEnv([make_env for _ in range(4)])
    coll = Collector(policies, envs)

    was_training = ppo_policy.training
    ppo_policy.eval()
    result = coll.collect(n_episode=n_episode)
    if was_training:
        ppo_policy.train()

    rews = result.get("rews", result.get("rew", 0.0))
    return float(rews.mean()) if hasattr(rews, "mean") else float(rews)


# ==========================================
# TRAINING
# ==========================================
def train_finetune():
    print("Initialisation...")
    env = make_env()

    encoder = SharedEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)
    belief_head = BeliefHead(encoder)

    print(f"Chargement BC depuis {BC_CHECKPOINT}...")
    ckpt = torch.load(BC_CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])

    all_params = list(set(
        list(encoder.parameters()) + list(actor.parameters()) + 
        list(critic.parameters()) + list(belief_head.parameters())
    ))
    # Optimizer unique gérant tout. Les gradients figés seront ignorés.
    optim = torch.optim.AdamW(all_params, lr=1e-5)

    def dist_fn(logits):
        return Categorical(logits=logits)

    ppo_policy = AuxPPOPolicy(
        belief_head=belief_head,
        belief_coef=0.5,
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=0.99,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method="",
        eps_clip=0.05, 
        vf_coef=0.5,
        ent_coef=0.0, 
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
    train_coll = Collector(policies, train_envs, VectorReplayBuffer(20000, 8))
    test_coll = Collector(policies, test_envs)

    print("\n=== BASELINE BC ===")
    baseline = evaluate(ppo_policy, n_episode=300)
    print(f"  BC baseline (eval déterministe) : {baseline:.3f}")

    # ========================================================
    # PHASE 0 : CRITIC WARM-UP (Via Tianshou Natif)
    # ========================================================
    print("\n=== PHASE 0 : CRITIC WARM-UP (3 epochs) ===")
    freeze(encoder)
    freeze(actor)
    freeze(belief_head)
    unfreeze(critic)
    ppo_policy.warmup_mode = True

    for param_group in optim.param_groups:
        param_group['lr'] = 3e-4  # LR fort pour initier la valeur

    onpolicy_trainer(
        policy=policies,
        train_collector=train_coll,
        test_collector=test_coll,
        max_epoch=3,
        step_per_epoch=5000,
        repeat_per_collect=10,
        episode_per_test=100,
        batch_size=256,
        step_per_collect=2000,
    )

    after_warmup = evaluate(ppo_policy, n_episode=300)
    print(f"  Après warm-up critic (policy identique) : {after_warmup:.3f}")

    # ========================================================
    # PHASE 1 : FINE-TUNE HEADS (Encoder gelé)
    # ========================================================
    print("\n=== PHASE 1 : FINE-TUNE HEADS (10 epochs) ===")
    unfreeze(actor)
    unfreeze(belief_head)
    ppo_policy.warmup_mode = False

    for param_group in optim.param_groups:
        param_group['lr'] = 1e-5  # LR modéré

    onpolicy_trainer(
        policy=policies,
        train_collector=train_coll,
        test_collector=test_coll,
        max_epoch=10,
        step_per_epoch=10000,
        repeat_per_collect=3,
        episode_per_test=100,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda r: r >= 1.0,
    )

    after_phase1 = evaluate(ppo_policy, n_episode=300)
    print(f"  Après Phase 1 : {after_phase1:.3f} (gain: {after_phase1 - baseline:+.3f})")

    # ========================================================
    # PHASE 2 : FINE-TUNE TOUT
    # ========================================================
    print("\n=== PHASE 2 : FINE-TUNE COMPLET (20 epochs) ===")
    unfreeze(encoder)

    for param_group in optim.param_groups:
        param_group['lr'] = 3e-6  # LR microscopique (Ne pas détruire le BC latent)

    onpolicy_trainer(
        policy=policies,
        train_collector=train_coll,
        test_collector=test_coll,
        max_epoch=20,
        step_per_epoch=10000,
        repeat_per_collect=3,
        episode_per_test=100,
        batch_size=256,
        step_per_collect=2000,
        stop_fn=lambda r: r >= 1.0,
    )

    final = evaluate(ppo_policy, n_episode=500)
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"  BC baseline     : {baseline:.3f}")
    print(f"  Après Phase 1   : {after_phase1:.3f}")
    print(f"  Après Phase 2   : {final:.3f}")
    print(f"  Gain total      : {final - baseline:+.3f}")

    torch.save({
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "belief_head": belief_head.state_dict(),
    }, checkpoint_path("ppo_finetuned_v2.pth"))
    print(f"\nSauvegardé : {checkpoint_path('ppo_finetuned_v2.pth')}")

if __name__ == "__main__":
    train_finetune()
