"""
Mesure l'accuracy du belief head, segmentée par taille de pioche restante.
"""

import argparse
import json
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
from love_letter.bots.heuristic import HeuristicBot
from love_letter.paths import checkpoint_path
from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefHead as BeliefConditionedHead,
)

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128
NUM_OPPONENTS = 3

DEFAULT_CHECKPOINT = checkpoint_path("curriculum_phase1.pth")


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT), nn.ReLU(),
        )
        self.rnn = nn.GRUCell(LATENT, LATENT)


class MaskedActor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, ACTION_DIM)


class BeliefHead(nn.Module):
    def __init__(self, encoder, num_opponents=NUM_OPPONENTS):
        super().__init__()
        self.encoder = encoder
        self.num_opponents = num_opponents
        self.head = nn.Linear(LATENT, num_opponents * 10)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") == "belief_conditioned_actor_v1":
        encoder = BeliefConditionedEncoder()
        actor = BeliefConditionedActor()
        belief = BeliefConditionedHead()
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
        belief.load_state_dict(ckpt["belief_head"])
        return {
            "kind": "belief_conditioned_actor_v1",
            "encoder": encoder,
            "actor": actor,
            "belief": belief,
        }

    encoder = RecurrentEncoder()
    actor = MaskedActor(encoder)
    belief = BeliefHead(encoder)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    belief.load_state_dict(ckpt["belief_head"])
    return {
        "kind": "legacy_actor_belief",
        "encoder": encoder,
        "actor": actor,
        "belief": belief,
    }


def model_step(model, obs_dict, hidden_state):
    x = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
    mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
    h_in = hidden_state if hidden_state is not None else torch.zeros(1, LATENT)

    if model["kind"] == "belief_conditioned_actor_v1":
        new_state = model["encoder"].forward_hidden(x, h_in)
        belief_logits, belief_probs = model["belief"](new_state)
        logits = model["actor"](new_state, belief_probs, mask)
    else:
        feat = model["encoder"].feature_extractor(x)
        new_state = model["encoder"].rnn(feat, h_in)
        belief_logits = model["belief"].head(new_state).view(-1, NUM_OPPONENTS, 10)
        logits = model["actor"].head(new_state).masked_fill(~mask, -1e9)

    action = int(logits.argmax(dim=-1).item())
    preds = belief_logits.argmax(dim=-1).squeeze(0).numpy()
    return action, new_state, preds


def measure_belief(checkpoint=DEFAULT_CHECKPOINT, n_games=300, opponents="heuristic"):
    model = load_model(checkpoint)
    model["encoder"].eval()
    model["actor"].eval()
    model["belief"].eval()

    bot = HeuristicBot()
    env = LoveLetterRLEnv(num_players=4)

    # Stocker (pred, target, deck_size, target_card)
    records = []

    for game in range(n_games):
        env.reset(seed=20_000 + game)
        h_state = None

        for agent in env.agent_iter():
            obs_dict, _, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue

            if agent == "player_0":
                with torch.no_grad():
                    action, h_state, preds = model_step(model, obs_dict, h_state)

                    hidden_cards = info.get("hidden_cards", np.full(3, -1))
                    deck_size = len(env._deck)

                    for i in range(3):
                        if hidden_cards[i] != -1:
                            records.append({
                                "pred": int(preds[i]),
                                "target": int(hidden_cards[i]),
                                "deck_size": deck_size,
                            })

            else:
                if opponents == "heuristic":
                    action = bot.choose_action(env, agent)
                else:
                    m = obs_dict["action_mask"]
                    valid = np.where(m == 1)[0]
                    action = int(np.random.choice(valid)) if len(valid) else 0

            env.step(action)

    preds = np.array([r["pred"] for r in records])
    targets = np.array([r["target"] for r in records])
    deck_sizes = np.array([r["deck_size"] for r in records])

    overall_acc = float((preds == targets).mean()) if len(records) else 0.0

    print(f"\n=== BELIEF ACCURACY (vs {opponents}, {n_games} parties, {len(records)} préds) ===")
    print(f"  Accuracy globale : {overall_acc*100:.1f}% (random = 10%)")

    # Par tranche de taille de pioche
    print("\n  📊 Accuracy par taille de pioche restante :")
    print(f"  {'Tranche':>15} | {'N préds':>8} | {'Accuracy':>10}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*10}")

    bins = [(0, 2), (3, 5), (6, 9), (10, 13), (14, 17)]
    bin_summary = {}
    for lo, hi in bins:
        mask = (deck_sizes >= lo) & (deck_sizes <= hi)
        n = mask.sum()
        if n > 0:
            acc = float((preds[mask] == targets[mask]).mean())
            label = f"{lo}-{hi} cartes"
            print(f"  {label:>15} | {n:>8} | {acc*100:>9.1f}%")
            bin_summary[label] = {"n": int(n), "accuracy": acc}
        else:
            label = f"{lo}-{hi} cartes"
            print(f"  {label:>15} | {0:>8} | {'  N/A':>10}")
            bin_summary[label] = {"n": 0, "accuracy": None}

    # Détail par carte cible
    print("\n  📊 Accuracy par type de carte adverse réelle :")
    names = ["Espionne", "Garde", "Prêtre", "Baron", "Servante",
             "Prince", "Chancelier", "Roi", "Comtesse", "Princesse"]
    card_summary = {}
    for c in range(10):
        m = targets == c
        if m.sum() > 0:
            acc_c = float((preds[m] == c).mean())
            print(f"    {names[c]:12s} (n={m.sum():4d}) : {acc_c*100:5.1f}%")
            card_summary[names[c]] = {"n": int(m.sum()), "accuracy": acc_c}

    return {
        "checkpoint": str(checkpoint),
        "model_kind": model["kind"],
        "opponents": opponents,
        "games": n_games,
        "predictions": len(records),
        "overall_accuracy": overall_acc,
        "by_deck_size": bin_summary,
        "by_target_card": card_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure belief-head accuracy for a checkpoint.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    report = {
        "heuristic": measure_belief(checkpoint=args.checkpoint, n_games=args.games, opponents="heuristic"),
        "random": measure_belief(checkpoint=args.checkpoint, n_games=args.games, opponents="random"),
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output}")
