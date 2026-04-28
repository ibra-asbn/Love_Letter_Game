"""Generate a detailed tactical diagnostic for a Love Letter policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.belief_policy import load_belief_policy
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path


CARD_NAMES = {
    0: "Espionne",
    1: "Garde",
    2: "Pretre",
    3: "Baron",
    4: "Servante",
    5: "Prince",
    6: "Chancelier",
    7: "Roi",
    8: "Comtesse",
    9: "Princesse",
}

GUARD_GUESSES = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def decode_action(action):
    if action >= 900:
        return {"kind": "chancellor_choice", "action": int(action)}
    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    return {
        "kind": "card",
        "card": int(card),
        "card_name": CARD_NAMES.get(card, str(card)),
        "target": f"player_{target_idx}" if target_idx < 4 else None,
        "guess": int(guess),
        "guess_name": CARD_NAMES.get(guess, str(guess)),
        "action": int(action),
    }


def target_belief_dim(my_idx, target_idx):
    if target_idx >= 4 or target_idx == my_idx:
        return None
    rel = (target_idx - my_idx) % 4
    return rel - 1 if rel else None


def extract_belief(policy):
    debug = getattr(policy, "last_debug", None)
    if debug is None:
        return None
    if hasattr(debug, "belief_probs") and debug.belief_probs is not None:
        return debug.belief_probs.squeeze(0).numpy()
    if hasattr(debug, "probs") and debug.probs is not None:
        return debug.probs
    return None


def top_belief_for_target(belief, my_idx, target):
    if belief is None or target is None:
        return None
    target_idx = int(target.rsplit("_", 1)[1])
    dim = target_belief_dim(my_idx, target_idx)
    if dim is None:
        return None
    probs = belief[dim]
    ranked = sorted(
        [
            {
                "card": int(card),
                "card_name": CARD_NAMES[card],
                "prob": float(prob),
            }
            for card, prob in enumerate(probs)
        ],
        key=lambda item: item["prob"],
        reverse=True,
    )
    return ranked[:3]


def best_guard_guess(belief_top):
    if not belief_top:
        return None
    for item in belief_top:
        if item["card"] != 1:
            return item["card"]
    return None


def non_princess_value(card):
    return -0.5 if card == 9 else card


def chancellor_choice(pool, action):
    pool_size = len(pool)
    action_idx = action - 900
    if pool_size >= 3:
        keep_idx = action_idx // 2
    elif pool_size == 2:
        keep_idx = action_idx
    else:
        keep_idx = 0
    if keep_idx >= pool_size or keep_idx < 0:
        keep_idx = 0
    return keep_idx, pool[keep_idx] if pool else None


def inspect_player_action(env, policy, action, obs_dict, hand_before, belief):
    decoded = decode_action(action)
    my_idx = env.possible_agents.index("player_0")
    event = {
        "turn_agent": "player_0",
        "hand_before": [CARD_NAMES[c] for c in hand_before],
        "hand_before_ids": [int(c) for c in hand_before],
        "decoded": decoded,
        "notes": [],
        "tags": [],
    }

    if decoded["kind"] == "chancellor_choice":
        pool = list(env._chancellor_pool)
        keep_idx, kept = chancellor_choice(pool, action)
        best_idx = max(range(len(pool)), key=lambda i: non_princess_value(pool[i])) if pool else None
        best = pool[best_idx] if best_idx is not None else None
        event.update(
            {
                "chancellor_pool": [CARD_NAMES[c] for c in pool],
                "kept": CARD_NAMES[kept] if kept is not None else None,
                "best_non_princess_keep": CARD_NAMES[best] if best is not None else None,
            }
        )
        if kept is not None and best is not None and kept == best:
            event["tags"].append("fulgurance")
            event["notes"].append("Chancelier garde la meilleure carte non-Princesse du pool.")
        elif kept is not None:
            event["tags"].append("bizarre")
            event["notes"].append("Chancelier ne garde pas la meilleure option visible dans le pool.")
        return event

    card = decoded["card"]
    target = decoded["target"]
    target_hand = list(env._hands.get(target, [])) if target else []
    actual_target_card = target_hand[0] if target_hand else None
    kept_cards = list(hand_before)
    if card in kept_cards:
        kept_cards.remove(card)
    kept = kept_cards[0] if kept_cards else None
    belief_top = top_belief_for_target(belief, my_idx, target)

    event.update(
        {
            "target_card_before": CARD_NAMES[actual_target_card] if actual_target_card is not None else None,
            "target_card_before_id": int(actual_target_card) if actual_target_card is not None else None,
            "kept_after_play": CARD_NAMES[kept] if kept is not None else None,
            "kept_after_play_id": int(kept) if kept is not None else None,
            "belief_top_target": belief_top,
        }
    )

    if card == 1 and actual_target_card is not None:
        best_guess = best_guard_guess(belief_top)
        if decoded["guess"] == actual_target_card:
            event["tags"].append("fulgurance")
            event["notes"].append("Garde touche exactement la carte adverse.")
        elif best_guess is not None and decoded["guess"] != best_guess:
            selected_prob = 0.0
            best_prob = 0.0
            for item in belief_top or []:
                if item["card"] == decoded["guess"]:
                    selected_prob = item["prob"]
                if item["card"] == best_guess:
                    best_prob = item["prob"]
            if best_prob - selected_prob >= 0.05:
                event["tags"].append("bizarre")
                event["notes"].append("Garde devine une carte moins probable que le top belief.")

    elif card == 3 and kept is not None and actual_target_card is not None:
        if kept > actual_target_card:
            event["tags"].append("fulgurance")
            event["notes"].append("Baron favorable: la carte gardee bat la cible.")
        elif kept < actual_target_card:
            event["tags"].append("bizarre")
            event["notes"].append("Baron perdant: il cible une main plus forte.")
        elif kept <= 4:
            event["tags"].append("bizarre")
            event["notes"].append("Baron neutre avec une carte gardee faible.")

    elif card == 5 and actual_target_card is not None:
        if target == "player_0" and actual_target_card >= 7:
            event["tags"].append("bizarre")
            event["notes"].append("Prince sur soi en sacrifiant une carte forte.")
        elif target != "player_0" and actual_target_card == 9:
            event["tags"].append("fulgurance")
            event["notes"].append("Prince force la Princesse adverse.")

    elif card == 7 and kept is not None and actual_target_card is not None:
        if actual_target_card > kept:
            event["tags"].append("fulgurance")
            event["notes"].append("Roi echange contre une carte meilleure.")
        elif actual_target_card < kept:
            event["tags"].append("bizarre")
            event["notes"].append("Roi donne une meilleure carte que celle recue.")

    elif card == 8 and not (5 in hand_before or 7 in hand_before):
        event["tags"].append("bizarre")
        event["notes"].append("Comtesse jouee volontairement sans Prince/Roi en main.")

    return event


def play_diagnostic(checkpoint, games):
    env = LoveLetterRLEnv(num_players=4)
    policy = load_belief_policy(checkpoint)
    bot = HeuristicBot()
    game_logs = []

    for game in range(games):
        env.reset(seed=game)
        state = None
        total_reward = 0.0
        events = []

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, reward, term, trunc, _info = env.last()
            if agent == "player_0":
                total_reward += float(reward)
            if term or trunc:
                env.step(None)
                continue

            if agent == "player_0":
                hand_before = list(env._hands.get(agent, []))
                action, state = policy.act(obs_dict, state, agent_id=agent)
                belief = extract_belief(policy)
                event = inspect_player_action(env, policy, action, obs_dict, hand_before, belief)
                event["turn"] = int(turn)
                events.append(event)
            else:
                action = bot.choose_action(env, agent)

            env.step(action)

        game_logs.append(
            {
                "game": game + 1,
                "reward": total_reward,
                "win": bool(total_reward >= 1.0),
                "round_winners": getattr(env, "_round_winners", []),
                "round_win_reason": getattr(env, "_round_win_reason", None),
                "events": events,
            }
        )

    return game_logs


def summarize(game_logs):
    wins = sum(1 for game in game_logs if game["win"])
    tags = Counter()
    card_counts = Counter()
    weird_by_note = Counter()
    sharp_by_note = Counter()
    action_examples = defaultdict(list)

    for game in game_logs:
        for event in game["events"]:
            decoded = event["decoded"]
            if decoded["kind"] == "card":
                card_counts[decoded["card_name"]] += 1
            else:
                card_counts["Chancelier choice"] += 1
            for tag in event["tags"]:
                tags[tag] += 1
            for note in event["notes"]:
                if "bizarre" in event["tags"]:
                    weird_by_note[note] += 1
                if "fulgurance" in event["tags"]:
                    sharp_by_note[note] += 1
                action_examples[note].append(
                    {
                        "game": game["game"],
                        "hand": event["hand_before"],
                        "action": event["decoded"],
                        "target_card": event.get("target_card_before"),
                        "belief_top": event.get("belief_top_target"),
                    }
                )

    return {
        "games": len(game_logs),
        "wins": wins,
        "winrate": wins / len(game_logs) if game_logs else 0.0,
        "mean_reward": float(np.mean([game["reward"] for game in game_logs])) if game_logs else 0.0,
        "tags": dict(tags),
        "card_counts": dict(card_counts),
        "weird_by_note": dict(weird_by_note),
        "sharp_by_note": dict(sharp_by_note),
        "examples": {
            note: examples[:3]
            for note, examples in action_examples.items()
        },
    }


def write_markdown(path, checkpoint, summary, raw_json):
    lines = [
        f"# Diagnostic modele - {Path(checkpoint).name}",
        "",
        "Date: 2026-04-24",
        "",
        "Configuration: player_0 joue le checkpoint, player_1/player_2/player_3 jouent HeuristicBot.",
        f"Log brut: `{raw_json}`",
        "",
        "## Resultat",
        "",
        f"- Parties: {summary['games']}",
        f"- Victoires: {summary['wins']} ({summary['winrate']:.1%})",
        f"- Reward moyen: {summary['mean_reward']:.3f}",
        "",
        "## Actions jouees par player_0",
        "",
    ]
    for card, count in sorted(summary["card_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {card}: {count}")

    lines.extend(["", "## Moments bizarres", ""])
    if summary["weird_by_note"]:
        for note, count in sorted(summary["weird_by_note"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {note}: {count}")
    else:
        lines.append("- Aucun motif bizarre detecte par les regles automatiques.")

    lines.extend(["", "## Fulgurances", ""])
    if summary["sharp_by_note"]:
        for note, count in sorted(summary["sharp_by_note"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {note}: {count}")
    else:
        lines.append("- Aucune fulgurance nette detectee par les regles automatiques.")

    lines.extend(["", "## Exemples", ""])
    for note, examples in summary["examples"].items():
        lines.append(f"### {note}")
        for example in examples:
            action = example["action"]
            if action["kind"] == "card":
                action_text = f"{action['card_name']} -> {action.get('target')} guess {action.get('guess_name')}"
            else:
                action_text = "choix Chancelier"
            lines.append(
                f"- Partie {example['game']}: main={example['hand']}, action={action_text}, "
                f"carte cible={example['target_card']}, belief_top={example['belief_top']}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic detaille d'un checkpoint Love Letter.")
    parser.add_argument("--checkpoint", default=str(checkpoint_path("curriculum_phase1.pth")))
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    game_logs = play_diagnostic(args.checkpoint, args.games)
    summary = summarize(game_logs)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"checkpoint": args.checkpoint, "summary": summary, "games": game_logs}, indent=2),
        encoding="utf-8",
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(output_md, args.checkpoint, summary, str(output_json))

    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
