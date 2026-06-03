"""
Behavioral Cloning : apprend à imiter l'heuristique en supervisé.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot
from love_letter.paths import checkpoint_path, data_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128


# ==========================================
# 1. ARCHITECTURE (identique à celle de PPO pour transfert direct)
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

    def forward_tensor(self, x):
        return self.net(x)


class MaskedActor(nn.Module):
    def __init__(self, encoder, action_dim=ACTION_DIM, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, action_dim)

    def forward_tensor(self, obs_t, mask_t):
        features = self.encoder.forward_tensor(obs_t)
        logits = self.head(features)
        logits = logits.masked_fill(~mask_t, -1e9)
        return logits


class Critic(nn.Module):
    """Inutilisé pour BC mais instancié pour garder les poids pour PPO ensuite."""
    def __init__(self, encoder, latent=LATENT):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent, 1)


# ==========================================
# 2. ÉVALUATION vs RANDOM
# ==========================================
def random_action(env, agent):
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid = np.where(mask == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def model_action(model, env, agent, device):
    obs_dict = env.observe(agent)
    obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model.forward_tensor(obs_t, mask_t)
        action = int(logits.argmax(dim=-1).item())
    return action


def evaluate_vs_random(model, device, n_games=500):
    env = LoveLetterRLEnv(num_players=4)
    total_reward = 0.0
    wins = 0
    model.eval()
    for i in range(n_games):
        env.reset(seed=100000 + i)  # seeds différents du training
        for agent in env.agent_iter():
            _, reward, termination, truncation, _ = env.last()
            if agent == "player_0":
                total_reward += reward
            if termination or truncation:
                env.step(None)
                continue
            if agent == "player_0":
                action = model_action(model, env, agent, device)
            else:
                action = random_action(env, agent)
            env.step(action)
        if total_reward - (total_reward - reward) >= 1.0:
            pass  # dummy, on va recompter proprement
        # Compte simple : on re-mesure via un accumulator par partie
    model.train()
    return total_reward / n_games


def evaluate_vs_random_clean(model, device, n_games=500):
    """Version propre du calcul de winrate."""
    env = LoveLetterRLEnv(num_players=4)
    total_reward = 0.0
    wins = 0
    model.eval()
    for i in range(n_games):
        env.reset(seed=100000 + i)
        game_reward = 0.0
        for agent in env.agent_iter():
            _, reward, termination, truncation, _ = env.last()
            if agent == "player_0":
                game_reward += reward
            if termination or truncation:
                env.step(None)
                continue
            if agent == "player_0":
                action = model_action(model, env, agent, device)
            else:
                action = random_action(env, agent)
            env.step(action)
        total_reward += game_reward
        if game_reward >= 1.0:
            wins += 1
    model.train()
    return total_reward / n_games, wins / n_games


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_bc():
    device = torch.device("cpu")  # Mac, pas de CUDA nécessaire

    # Load dataset
    print("Chargement du dataset...")
    with open(data_path("heuristic_dataset.pkl"), "rb") as f:
        data = pickle.load(f)

    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    mask = torch.as_tensor(data["mask"], dtype=torch.bool)
    action = torch.as_tensor(data["action"], dtype=torch.long)

    print(f"  {len(obs)} transitions chargées")

    # Train/val split
    n = len(obs)
    n_val = n // 10
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:-n_val], perm[-n_val:]
    obs_train, mask_train, act_train = obs[train_idx], mask[train_idx], action[train_idx]
    obs_val, mask_val, act_val = obs[val_idx], mask[val_idx], action[val_idx]

    # DataLoader
    train_ds = TensorDataset(obs_train, mask_train, act_train)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # Model
    encoder = SharedEncoder()
    actor = MaskedActor(encoder)
    critic = Critic(encoder)  # instancié mais non entraîné

    params = list(actor.parameters())  # critic non inclus en BC
    optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-5)

    loss_fn = nn.CrossEntropyLoss()

    print("\nDémarrage du BC...")
    epochs = 50
    best_winrate = 0.0

    for epoch in range(epochs):
        actor.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for obs_b, mask_b, act_b in train_loader:
            logits = actor.forward_tensor(obs_b, mask_b)
            loss = loss_fn(logits, act_b)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * len(obs_b)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == act_b).sum().item()
            total_samples += len(obs_b)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validation accuracy
        actor.eval()
        with torch.no_grad():
            val_logits = actor.forward_tensor(obs_val, mask_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == act_val).float().mean().item()

        # Winrate vs random (tous les 2 epochs pour gagner du temps)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            mean_r, winrate = evaluate_vs_random_clean(actor, device, n_games=300)
            print(f"Epoch {epoch+1:2d} | loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} | vs_random: reward={mean_r:.3f} winrate={winrate:.3f}")

            if winrate > best_winrate:
                best_winrate = winrate
                torch.save({
                    "encoder": encoder.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                }, checkpoint_path("bc_best.pth"))
        else:
            print(f"Epoch {epoch+1:2d} | loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f}")

    print(f"\n BC terminé. Meilleur winrate vs random : {best_winrate:.3f}")
    print(f"  Poids sauvegardés : {checkpoint_path('bc_best.pth')}")


if __name__ == "__main__":
    train_bc()
