"""
Debug : pourquoi player_0 fait 0.72 vs 3 clones identiques au lieu de 0.25 ?
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

from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128


# ==========================================
# ARCHITECTURE
# ==========================================
class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, LATENT),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(LATENT, LATENT)

    def forward(self, x, h_in):
        features = self.feature_extractor(x)
        return features, self.rnn(features, h_in)


class MaskedActor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, ACTION_DIM)

    def act(self, obs_dict, hidden_state):
        with torch.no_grad():
            x = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
            mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
            h_in = hidden_state if hidden_state is not None else torch.zeros(1, LATENT)
            _, new_state = self.encoder(x, h_in)
            logits = self.head(new_state)
            logits = logits.masked_fill(~mask, -1e9)
            action = int(logits.argmax(dim=-1).item())
        return action, new_state


def load_actor(path):
    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


# ==========================================
# TEST 1 : Random vs Random vs Random vs Random
# ==========================================
def test_random_4way(n_games=2000, seeds_offset=0):
    """Mesure de référence : 4 random, on doit avoir ~0.25 par joueur."""
    env = LoveLetterRLEnv(num_players=4)
    rewards = {f"player_{i}": np.zeros(n_games) for i in range(4)}

    for game in range(n_games):
        env.reset(seed=game + seeds_offset)
        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent][game] += r
            if term or trunc:
                env.step(None)
                continue
            obs_d = env.observe(agent)
            mask = obs_d["action_mask"]
            valid = np.where(mask == 1)[0]
            env.step(int(np.random.choice(valid)) if len(valid) else 0)

    print("\n=== TEST 1 : 4 Random — REFERENCE (devrait être ~0.36 chacun avec espionne) ===")
    for i in range(4):
        r = rewards[f"player_{i}"]
        print(f"  player_{i} : reward={r.mean():.3f} ± {1.96*r.std()/np.sqrt(n_games):.3f}")


# ==========================================
# TEST 2 : 4 clones du même modèle, MÊMES poids, MÊME instance
# ==========================================
def test_4_clones_same_instance(actor_path, n_games=500):
    """4 clones partageant la MÊME instance Python (comme dans le buggé)."""
    actor = load_actor(actor_path)
    env = LoveLetterRLEnv(num_players=4)
    rewards = {f"player_{i}": np.zeros(n_games) for i in range(4)}
    actions_count = {f"player_{i}": 0 for i in range(4)}

    for game in range(n_games):
        env.reset(seed=game)
        states = {f"player_{i}": None for i in range(4)}

        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent][game] += r
            if term or trunc:
                env.step(None)
                continue
            obs_d = env.observe(agent)
            action, states[agent] = actor.act(obs_d, states[agent])
            actions_count[agent] += 1
            env.step(action)

    print("\n=== TEST 2 : 4 clones, MÊME instance ===")
    for i in range(4):
        r = rewards[f"player_{i}"]
        print(f"  player_{i} : reward={r.mean():.3f} ± {1.96*r.std()/np.sqrt(n_games):.3f}  "
              f"actions/game={actions_count[f'player_{i}']/n_games:.1f}")


# ==========================================
# TEST 3 : 4 clones, INSTANCES SÉPARÉES
# ==========================================
def test_4_clones_separate_instances(actor_path, n_games=500):
    """4 instances indépendantes des mêmes poids."""
    actors = [load_actor(actor_path) for _ in range(4)]
    env = LoveLetterRLEnv(num_players=4)
    rewards = {f"player_{i}": np.zeros(n_games) for i in range(4)}
    actions_count = {f"player_{i}": 0 for i in range(4)}

    for game in range(n_games):
        env.reset(seed=game)
        states = {f"player_{i}": None for i in range(4)}

        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent][game] += r
            if term or trunc:
                env.step(None)
                continue
            obs_d = env.observe(agent)
            idx = int(agent[-1])
            action, states[agent] = actors[idx].act(obs_d, states[agent])
            actions_count[agent] += 1
            env.step(action)

    print("\n=== TEST 3 : 4 clones, INSTANCES SÉPARÉES ===")
    for i in range(4):
        r = rewards[f"player_{i}"]
        print(f"  player_{i} : reward={r.mean():.3f} ± {1.96*r.std()/np.sqrt(n_games):.3f}  "
              f"actions/game={actions_count[f'player_{i}']/n_games:.1f}")


# ==========================================
# TEST 4 : 4 clones, SEEDS DIFFÉRENTS PAR PARTIE (pas seed=game systématique)
# ==========================================
def test_4_clones_random_seeds(actor_path, n_games=500):
    """Pour vérifier si seed=game crée un biais."""
    actor = load_actor(actor_path)
    env = LoveLetterRLEnv(num_players=4)
    rewards = {f"player_{i}": np.zeros(n_games) for i in range(4)}

    for game in range(n_games):
        env.reset()  # SEED RANDOM (pas de seed=game)
        states = {f"player_{i}": None for i in range(4)}

        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent][game] += r
            if term or trunc:
                env.step(None)
                continue
            obs_d = env.observe(agent)
            action, states[agent] = actor.act(obs_d, states[agent])
            env.step(action)

    print("\n=== TEST 4 : 4 clones, SEEDS RANDOM (pas seed=game) ===")
    for i in range(4):
        r = rewards[f"player_{i}"]
        print(f"  player_{i} : reward={r.mean():.3f} ± {1.96*r.std()/np.sqrt(n_games):.3f}")


# ==========================================
# TEST 5 : Qui démarre la partie ?
# ==========================================
def test_starting_agent_distribution(n_games=2000):
    """Vérifier qui commence vraiment à jouer en moyenne."""
    env = LoveLetterRLEnv(num_players=4)
    starters = {f"player_{i}": 0 for i in range(4)}

    for game in range(n_games):
        env.reset(seed=game)
        first_agent = env.agent_selection
        starters[first_agent] += 1

    print("\n=== TEST 5 : Distribution du starting_agent (devrait être uniforme ~25%) ===")
    for i in range(4):
        a = f"player_{i}"
        print(f"  {a} : commence dans {starters[a]/n_games*100:.1f}% des parties")


# ==========================================
# TEST 6 : Identifier qui gagne quand on force player_0 à toujours commencer
# ==========================================
def test_starting_agent_advantage(n_games=2000):
    """Si player_0 commence systématiquement, gagne-t-il plus en random ?"""
    env = LoveLetterRLEnv(num_players=4)
    rewards = {f"player_{i}": np.zeros(n_games) for i in range(4)}

    for game in range(n_games):
        env.reset(seed=game, options={"starting_agent": "player_0"})
        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            rewards[agent][game] += r
            if term or trunc:
                env.step(None)
                continue
            obs_d = env.observe(agent)
            mask = obs_d["action_mask"]
            valid = np.where(mask == 1)[0]
            env.step(int(np.random.choice(valid)) if len(valid) else 0)

    print("\n=== TEST 6 : 4 Random AVEC player_0 forcé à commencer ===")
    for i in range(4):
        r = rewards[f"player_{i}"]
        print(f"  player_{i} : reward={r.mean():.3f}")


if __name__ == "__main__":
    MODEL = checkpoint_path("ppo_love_letter_league_champion.pth")

    print("=" * 70)
    print("DEBUG : Pourquoi player_0 fait 0.72 vs 3 self ?")
    print("=" * 70)

    test_random_4way(n_games=2000)
    test_starting_agent_distribution(n_games=2000)
    test_starting_agent_advantage(n_games=2000)
    test_4_clones_same_instance(MODEL, n_games=500)
    test_4_clones_separate_instances(MODEL, n_games=500)
    test_4_clones_random_seeds(MODEL, n_games=500)

    print("\n" + "=" * 70)
    print("INTERPRÉTATION")
    print("=" * 70)
    print("""
    Si Test 1 : tous ~0.36 → engine fair, baseline OK
    Si Test 5 : starter pas uniforme → bug de seed/random dans engine
    Si Test 6 vs Test 1 : player_0 a avantage quand il commence → engine biaisé
    Si Test 2 : player_0 à 0.7+, autres à 0.1-0.2 → bug d'instance partagée
    Si Test 3 : tous équilibrés (~0.25) → bug d'instance confirmé
    Si Test 4 : tous équilibrés alors que Test 2 non → bug seed=game
    """)
