"""Distill belief-guided retargeting into the actor.

This is a focused repair step after the counterfactual result:
the belief head contains useful probabilities, but the actor often fails to
use them for target/guess choices. We freeze encoder + belief, collect states
from the current actor, compute the retargeted action, and train the actor to
produce that action directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefHead,
    LATENT,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import CHECKPOINT_DIR, checkpoint_path
from scripts.evaluation.evaluate_belief_counterfactual import (
    best_retarget_action,
    cards_from_obs,
    decode_card,
)


PARIS_TZ = ZoneInfo("Europe/Paris")


def now_stamp():
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


class ExperimentLogger:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(self, title, expected=None, actual=None, details=None):
        message = f"[{now_stamp()}] {title}"
        print(message, flush=True)
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"## {message}\n\n")
            if expected is not None:
                f.write(f"**Attendu**: {expected}\n\n")
            if actual is not None:
                f.write(f"**Obtenu**: {actual}\n\n")
            if details is not None:
                f.write("```json\n")
                f.write(json.dumps(details, indent=2, ensure_ascii=False))
                f.write("\n```\n\n")


def resolve_checkpoint(name_or_path):
    path = Path(name_or_path)
    if path.exists():
        return path
    candidate = checkpoint_path(name_or_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {name_or_path}")


def output_checkpoint(name_or_path):
    path = Path(name_or_path)
    if path.parent != Path("."):
        return path
    return CHECKPOINT_DIR / path


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "belief_conditioned_actor_v1":
        raise ValueError(f"{path} is not a belief-conditioned actor checkpoint")

    encoder = BeliefConditionedEncoder()
    belief_head = BeliefHead()
    actor = BeliefConditionedActor()
    encoder.load_state_dict(ckpt["encoder"])
    belief_head.load_state_dict(ckpt["belief_head"])
    actor.load_state_dict(ckpt["actor"])
    return ckpt, encoder, belief_head, actor


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def opponent_action(env, agent, obs_dict, n_heuristics, bot):
    seat_idx = int(agent.rsplit("_", 1)[1])
    if 1 <= seat_idx <= n_heuristics:
        return bot.choose_action(env, agent)
    return random_action(obs_dict)


@dataclass
class Collection:
    hidden: torch.Tensor
    belief: torch.Tensor
    mask: torch.Tensor
    raw_action: torch.Tensor
    target_action: torch.Tensor
    changed: torch.Tensor
    summary: dict


def collect_retarget_dataset(encoder, belief_head, actor, args, device, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    opponent_mix = list(args.opponent_mix)

    hiddens = []
    beliefs = []
    masks = []
    raw_actions = []
    target_actions = []
    changed_flags = []

    changes_by_reason = Counter()
    raw_cards = Counter()
    target_cards = Counter()
    rewards = []
    games_by_heuristics = Counter()

    encoder.eval()
    belief_head.eval()
    actor.eval()

    for game in range(args.games):
        np.random.seed(args.seed + game)
        env.reset(seed=args.seed + game)
        n_heuristics = int(np.random.choice(opponent_mix))
        games_by_heuristics[n_heuristics] += 1
        hidden_state = torch.zeros(1, LATENT, device=device)
        total_reward = 0.0

        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                total_reward += float(reward)
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)

                with torch.no_grad():
                    hidden_state = encoder.forward_hidden(obs_t, hidden_state)
                    _belief_logits, belief_probs = belief_head(hidden_state)
                    logits = actor(hidden_state, belief_probs, mask_t)
                    raw_action = int(logits.argmax(dim=-1).item())

                belief_np = belief_probs.squeeze(0).detach().cpu().numpy()
                hand = cards_from_obs(obs_dict["observation"])
                target_action, reason = best_retarget_action(
                    raw_action,
                    obs_dict["action_mask"],
                    hand,
                    belief_np,
                    my_idx=0,
                    env=env,
                )
                if target_action is None:
                    target_action = raw_action

                changed = int(target_action != raw_action)
                if changed:
                    changes_by_reason[reason or "changed"] += 1

                raw_cards[decode_card(raw_action)] += 1
                target_cards[decode_card(target_action)] += 1

                hiddens.append(hidden_state.squeeze(0).detach().cpu().numpy())
                beliefs.append(belief_probs.squeeze(0).detach().cpu().numpy())
                masks.append(obs_dict["action_mask"].copy())
                raw_actions.append(raw_action)
                target_actions.append(target_action)
                changed_flags.append(changed)

                action = raw_action
            else:
                action = opponent_action(env, agent, obs_dict, n_heuristics, bot)

            env.step(action)

        rewards.append(total_reward)
        if args.log_every_games and (game + 1) % args.log_every_games == 0:
            logger.write(
                "Collecte dataset retarget",
                expected="Accumuler des decisions avec cibles corrigees par belief.",
                actual=f"{game + 1}/{args.games} parties collectees",
                details={
                    "samples": len(target_actions),
                    "change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
                    "mean_reward_actor_raw": float(np.mean(rewards)) if rewards else 0.0,
                },
            )

    summary = {
        "games": args.games,
        "samples": len(target_actions),
        "changed_samples": int(sum(changed_flags)),
        "change_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
        "changes_by_reason": dict(changes_by_reason),
        "raw_card_counts": dict(raw_cards),
        "target_card_counts": dict(target_cards),
        "mean_reward_actor_raw": float(np.mean(rewards)) if rewards else 0.0,
        "games_by_heuristics": {str(k): int(v) for k, v in sorted(games_by_heuristics.items())},
    }
    return Collection(
        hidden=torch.as_tensor(np.array(hiddens), dtype=torch.float32, device=device),
        belief=torch.as_tensor(np.array(beliefs), dtype=torch.float32, device=device),
        mask=torch.as_tensor(np.array(masks), dtype=torch.bool, device=device),
        raw_action=torch.as_tensor(raw_actions, dtype=torch.long, device=device),
        target_action=torch.as_tensor(target_actions, dtype=torch.long, device=device),
        changed=torch.as_tensor(changed_flags, dtype=torch.float32, device=device),
        summary=summary,
    )


def split_collection(data, val_ratio, seed):
    n = len(data.target_action)
    generator = torch.Generator(device=data.target_action.device)
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=data.target_action.device)
    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def evaluate_actor_on_indices(actor, data, indices, old_actor=None):
    actor.eval()
    with torch.no_grad():
        logits = actor(data.hidden[indices], data.belief[indices], data.mask[indices])
        pred = logits.argmax(dim=-1)
        target = data.target_action[indices]
        raw = data.raw_action[indices]
        changed = data.changed[indices].bool()
        metrics = {
            "target_acc": float((pred == target).float().mean().item()),
            "raw_acc": float((pred == raw).float().mean().item()),
            "changed_target_acc": float((pred[changed] == target[changed]).float().mean().item())
            if changed.any()
            else None,
            "unchanged_raw_acc": float((pred[~changed] == raw[~changed]).float().mean().item())
            if (~changed).any()
            else None,
        }
        if old_actor is not None:
            old_pred = old_actor(data.hidden[indices], data.belief[indices], data.mask[indices]).argmax(dim=-1)
            metrics["changed_from_old_rate"] = float((pred != old_pred).float().mean().item())
    actor.train()
    return metrics


def train_actor(actor, data, args, device, logger):
    old_actor = deepcopy(actor).to(device).eval()
    for param in old_actor.parameters():
        param.requires_grad_(False)

    train_idx, val_idx = split_collection(data, args.val_ratio, args.seed + 999)
    train_ds = TensorDataset(train_idx)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")
    history = []

    logger.write(
        "Debut distillation actor",
        expected=(
            "L'actor doit apprendre les actions retarget tout en conservant les actions brutes "
            "sur les cas ou retarget ne change rien."
        ),
        actual=f"train_samples={len(train_idx)}, val_samples={len(val_idx)}",
        details={
            "initial_train": evaluate_actor_on_indices(actor, data, train_idx, old_actor),
            "initial_val": evaluate_actor_on_indices(actor, data, val_idx, old_actor),
        },
    )

    for epoch in range(1, args.epochs + 1):
        actor.train()
        losses = []
        for (idx,) in loader:
            idx = idx.to(device)
            logits = actor(data.hidden[idx], data.belief[idx], data.mask[idx])
            weights = 1.0 + data.changed[idx] * (args.changed_weight - 1.0)
            loss = (ce(logits, data.target_action[idx]) * weights).mean()

            if args.kl_coef > 0:
                with torch.no_grad():
                    old_logits = old_actor(data.hidden[idx], data.belief[idx], data.mask[idx])
                    old_log_probs = torch.log_softmax(old_logits, dim=-1)
                    old_probs = torch.softmax(old_logits, dim=-1)
                new_log_probs = torch.log_softmax(logits, dim=-1)
                kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
                loss = loss + args.kl_coef * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        train_metrics = evaluate_actor_on_indices(actor, data, train_idx, old_actor)
        val_metrics = evaluate_actor_on_indices(actor, data, val_idx, old_actor)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        logger.write(
            f"Epoch {epoch}/{args.epochs}",
            expected="La target_acc et surtout changed_target_acc doivent monter.",
            actual=(
                f"loss={row['loss']:.4f}, "
                f"val_target_acc={val_metrics['target_acc']:.3f}, "
                f"val_changed_target_acc={val_metrics['changed_target_acc']}"
            ),
            details=row,
        )

    return history


def save_checkpoint(path, base_ckpt, encoder, belief_head, actor, args, collection_summary, history):
    output = output_checkpoint(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "belief_conditioned_actor_v1",
        "encoder": encoder.cpu().state_dict(),
        "belief_head": belief_head.cpu().state_dict(),
        "actor": actor.cpu().state_dict(),
        "metadata": {
            "source_checkpoint": str(resolve_checkpoint(args.start)),
            "training": "belief_retarget_distillation",
            "args": vars(args),
            "collection_summary": collection_summary,
            "history": history,
        },
    }
    if "critic" in base_ckpt:
        payload["critic"] = base_ckpt["critic"]
    torch.save(payload, output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Distill belief retargeting into a Love Letter actor.")
    parser.add_argument("--start", default="champion_belief_ppo_attempt2_tactical_best.pth")
    parser.add_argument("--output", default="champion_belief_retarget_distilled_attempt1.pth")
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--opponent-mix", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--changed-weight", type=float, default=4.0)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--log-every-games", type=int, default=1000)
    parser.add_argument("--run-log", default="logs/runs/2026-04-24_belief_retarget_distillation_attempt1.md")
    parser.add_argument("--report", default="logs/evaluations/2026-04-24_belief_retarget_distillation_attempt1_train.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=6262)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    logger = ExperimentLogger(args.run_log)

    start_path = resolve_checkpoint(args.start)
    base_ckpt, encoder, belief_head, actor = load_checkpoint(start_path)
    encoder.to(device).eval()
    belief_head.to(device).eval()
    actor.to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    for param in belief_head.parameters():
        param.requires_grad_(False)

    logger.write(
        "Constat et correction prevue",
        expected=(
            "Constat: le mode retarget belief a ameliore le score composite de 0.2345 a 0.2948 "
            "sur 1000 parties/config, donc le belief est utile mais l'actor brut l'ignore trop souvent."
        ),
        actual=(
            "Correction: geler encoder+belief, collecter des decisions de l'actor brut, calculer "
            "la cible retarget, puis distiller cette cible dans l'actor."
        ),
        details=vars(args),
    )

    data = collect_retarget_dataset(encoder, belief_head, actor, args, device, logger)
    logger.write(
        "Dataset retarget collecte",
        expected="Obtenir assez de decisions corrigees pour apprendre le comportement cible.",
        actual=(
            f"samples={data.summary['samples']}, changed={data.summary['changed_samples']} "
            f"({data.summary['change_rate']:.1%})"
        ),
        details=data.summary,
    )

    history = train_actor(actor, data, args, device, logger)
    output_path = save_checkpoint(args.output, base_ckpt, encoder, belief_head, actor, args, data.summary, history)

    report = {
        "created_at": now_stamp(),
        "start": str(start_path),
        "output": str(output_path),
        "collection_summary": data.summary,
        "history": history,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin distillation actor",
        expected="Sauvegarder un checkpoint qui devrait reduire l'ecart raw vs retarget.",
        actual=f"checkpoint={output_path}, report={report_path}",
        details={
            "final_val": history[-1]["val"] if history else None,
            "collection_summary": data.summary,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
