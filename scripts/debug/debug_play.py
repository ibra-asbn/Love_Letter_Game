"""
Affiche en détail 5 parties avec 4 clones pour identifier le pattern.
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

CARD_NAMES = {0:"Espionne",1:"Garde",2:"Prêtre",3:"Baron",4:"Servante",
              5:"Prince",6:"Chancelier",7:"Roi",8:"Comtesse",9:"Princesse"}


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT), nn.ReLU(),
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


def decode_action(action):
    card = action // 100
    target = (action % 100) // 10
    guess = action % 10
    if action >= 900:
        return f"Chancelier-resolution[{action-900}]"
    s = f"Joue {CARD_NAMES.get(card, '?')}"
    if card in [1,2,3,5,7]:
        s += f" sur player_{target}"
    if card == 1:
        s += f" devine {CARD_NAMES.get(guess, '?')}"
    return s


def play_one_verbose(actor, seed=0):
    env = LoveLetterRLEnv(num_players=4)
    env.reset(seed=seed)
    states = {f"player_{i}": None for i in range(4)}

    print(f"\n{'='*60}")
    print(f"PARTIE {seed} (starting: {env.agent_selection})")
    print(f"{'='*60}")
    for a in env.possible_agents:
        print(f"  {a} main initiale : {[CARD_NAMES[c] for c in env._hands[a]]}")
    print(f"  set_aside (caché) : {CARD_NAMES[env._set_aside]}")

    rewards = {f"player_{i}": 0.0 for i in range(4)}
    step = 0

    for agent in env.agent_iter():
        _, r, term, trunc, _ = env.last()
        rewards[agent] += r
        if term or trunc:
            env.step(None)
            continue

        obs_d = env.observe(agent)
        hand_before = [CARD_NAMES[c] for c in env._hands[agent]]
        action, states[agent] = actor.act(obs_d, states[agent])
        decoded = decode_action(action)

        print(f"  Step {step:2d} | {agent} | main={hand_before} | {decoded}")
        env.step(action)
        step += 1

        # Vérifier si quelqu'un vient d'être éliminé
        for a in env.possible_agents:
            if env.terminations.get(a, False) and a not in [agent]:
                # Mort récente
                pass

    print(f"\n  Rewards finaux : {rewards}")
    survivors = [a for a in env.possible_agents if env._hands.get(a)]
    if survivors:
        print(f"  Survivants : {[(a, CARD_NAMES[env._hands[a][0]]) for a in survivors]}")


def main():
    actor = load_actor(checkpoint_path("ppo_love_letter_league_champion.pth"))

    print("Affichage de 5 parties en détail (4 clones, seeds 0-4)")
    for seed in range(5):
        play_one_verbose(actor, seed=seed)


if __name__ == "__main__":
    main()
