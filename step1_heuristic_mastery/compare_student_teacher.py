"""Action-by-action comparison between a student checkpoint and HeuristicBot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefHead,
    LATENT,
)
from love_letter.paths import checkpoint_path
from step1_heuristic_mastery.common import (
    ExperimentLogger,
    STEP_CHECKPOINT_DIR,
    STEP_DATA_DIR,
    STEP_REPORT_DIR,
    decode_action,
    ensure_step_dirs,
    now_stamp,
    resolve_step_path,
)


def resolve_checkpoint(name_or_path):
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = STEP_CHECKPOINT_DIR / name_or_path
    if candidate.exists():
        return candidate
    candidate = checkpoint_path(name_or_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    encoder = BeliefConditionedEncoder().to(device)
    belief_head = BeliefHead().to(device)
    actor = BeliefConditionedActor().to(device)
    encoder.load_state_dict(ckpt["encoder"])
    belief_head.load_state_dict(ckpt["belief_head"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    belief_head.eval()
    actor.eval()
    return encoder, belief_head, actor


def action_parts(action):
    decoded = decode_action(action)
    if decoded["kind"] == "chancellor_choice":
        return ("ChancellorChoice", None, None)
    return (decoded["card_name"], decoded["target"], decoded["guess"])


def compare(args):
    dataset_path = resolve_step_path(args.dataset, STEP_DATA_DIR)
    with dataset_path.open("rb") as f:
        dataset = pickle.load(f)
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    encoder, belief_head, actor = load_model(checkpoint, device)

    rng = np.random.default_rng(args.seed)
    sequences = list(dataset["sequences"])
    if args.max_sequences and args.max_sequences < len(sequences):
        indices = rng.choice(len(sequences), size=args.max_sequences, replace=False)
        sequences = [sequences[int(i)] for i in indices]

    totals = Counter()
    by_teacher_card = defaultdict(Counter)
    disagreements = Counter()
    examples = []

    with torch.no_grad():
        for seq in sequences:
            hidden = torch.zeros(1, LATENT, device=device)
            for step in seq["steps"]:
                obs = torch.as_tensor(step["obs"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.as_tensor(step["mask"], dtype=torch.bool, device=device).unsqueeze(0)
                hidden = encoder.forward_hidden(obs, hidden)
                _belief_logits, belief_probs = belief_head(hidden)
                logits = actor(hidden, belief_probs, mask)
                pred = int(logits.argmax(dim=-1).item())
                teacher = int(step["action"])
                teacher_card, teacher_target, teacher_guess = action_parts(teacher)
                pred_card, pred_target, pred_guess = action_parts(pred)

                totals["samples"] += 1
                by_teacher_card[teacher_card]["samples"] += 1
                if pred == teacher:
                    totals["exact"] += 1
                    by_teacher_card[teacher_card]["exact"] += 1
                if pred_card == teacher_card:
                    totals["same_card"] += 1
                    by_teacher_card[teacher_card]["same_card"] += 1
                if teacher_target is not None and pred_target == teacher_target:
                    totals["same_target"] += 1
                    by_teacher_card[teacher_card]["same_target"] += 1
                if teacher_guess is not None and pred_guess == teacher_guess:
                    totals["same_guess"] += 1
                    by_teacher_card[teacher_card]["same_guess"] += 1

                if pred != teacher:
                    key = f"{teacher_card} -> {pred_card}"
                    disagreements[key] += 1
                    if len(examples) < args.max_examples:
                        examples.append(
                            {
                                "game": int(seq["game"]),
                                "agent": seq["agent"],
                                "teacher_action": decode_action(teacher),
                                "student_action": decode_action(pred),
                            }
                        )

    samples = max(1, totals["samples"])
    by_card = {}
    for card, counts in by_teacher_card.items():
        n = max(1, counts["samples"])
        by_card[card] = {
            "samples": int(counts["samples"]),
            "exact_acc": counts["exact"] / n,
            "same_card_acc": counts["same_card"] / n,
            "same_target_acc": counts["same_target"] / n,
            "same_guess_acc": counts["same_guess"] / n,
        }

    return {
        "created_at": now_stamp(),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "samples": int(totals["samples"]),
        "exact_acc": totals["exact"] / samples,
        "same_card_acc": totals["same_card"] / samples,
        "same_target_acc": totals["same_target"] / samples,
        "same_guess_acc": totals["same_guess"] / samples,
        "by_teacher_card": by_card,
        "top_disagreements": dict(disagreements.most_common(20)),
        "examples": examples,
    }


def main():
    ensure_step_dirs()
    parser = argparse.ArgumentParser(description="Compare student actions against teacher labels.")
    parser.add_argument("--checkpoint", default="heuristic_student_attempt1.pth")
    parser.add_argument("--dataset", default="teacher_sequences_attempt1.pkl")
    parser.add_argument("--output", default="heuristic_student_attempt1_action_compare.json")
    parser.add_argument("--run-log", default="step1_heuristic_mastery/logs/2026-04-24_step1_compare_attempt1.md")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9300)
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut comparaison action par action",
        expected="Identifier ce que l'etudiant imite deja et les decisions heuristiques qui restent mal apprises.",
        actual=f"checkpoint={args.checkpoint}, dataset={args.dataset}",
        details=vars(args),
    )
    report = compare(args)
    output = resolve_step_path(args.output, STEP_REPORT_DIR)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin comparaison action par action",
        expected="L'exact_acc doit etre elevee, et les erreurs restantes doivent etre visibles.",
        actual=f"exact_acc={report['exact_acc']:.4f}, same_card_acc={report['same_card_acc']:.4f}",
        details=report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

