"""Train a fast action-value head from determinized rollout labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import BELIEF_DIM, LATENT
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import OPPONENT_CONFIGS, random_action
from step3_action_value.evaluate_rollout_guided import evaluate_candidate_actions
from step3_action_value.mini_rollout_probe import choose_actions_for_probe, classify_state, decode_action
from step3_action_value.train_regret_override import Player0FeaturePolicy, opponent_action


STEP_DIR = PROJECT_ROOT / "step3_action_value"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def action_parts(actions: torch.Tensor):
    actions = actions.long()
    is_choice = (actions >= 900) & (actions <= 905)
    card = torch.div(actions, 100, rounding_mode="floor").clamp(0, 9)
    target = torch.div(actions % 100, 10, rounding_mode="floor").clamp(0, 9)
    guess = (actions % 10).clamp(0, 9)
    card = torch.where(is_choice, torch.full_like(card, 10), card)
    target = torch.where(is_choice, (actions - 900).clamp(0, 9), target)
    guess = torch.where(is_choice, torch.zeros_like(guess), guess)
    return card, target, guess


class ActionValueHead(nn.Module):
    def __init__(self, hidden_dim=192, embed_dim=16):
        super().__init__()
        self.card_emb = nn.Embedding(11, embed_dim)
        self.target_emb = nn.Embedding(10, embed_dim // 2)
        self.guess_emb = nn.Embedding(10, embed_dim // 2)
        action_feat = embed_dim + embed_dim // 2 + embed_dim // 2
        input_dim = LATENT + BELIEF_DIM + action_feat * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _embed_action(self, actions):
        card, target, guess = action_parts(actions)
        return torch.cat(
            [self.card_emb(card), self.target_emb(target), self.guess_emb(guess)],
            dim=-1,
        )

    def forward(self, hidden, belief, candidate_action, model_action):
        candidate_emb = self._embed_action(candidate_action)
        model_emb = self._embed_action(model_action)
        is_model = (candidate_action == model_action).float().unsqueeze(-1)
        x = torch.cat(
            [hidden, belief.reshape(hidden.shape[0], -1), candidate_emb, model_emb, is_model],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


def collect_action_value_rows(args, checkpoint, logger):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot()
    states = []
    rows = []
    category_counts = Counter()
    config_counts = Counter()
    candidate_counts = Counter()
    config_names = args.collect_configs

    for game in range(args.collect_games):
        if all(category_counts[category] >= args.states_per_category for category in args.categories):
            break
        config_name = config_names[game % len(config_names)]
        opponents = OPPONENT_CONFIGS[config_name]
        seed = args.seed + game
        np.random.seed(seed)
        env.reset(seed=seed)
        policy = Player0FeaturePolicy(checkpoint)

        for turn, agent in enumerate(env.agent_iter()):
            obs_dict, _reward, terminated, truncated, _info = env.last()
            if terminated or truncated:
                env.step(None)
                continue

            if agent == "player_0":
                category = classify_state(env, agent)
                model_action, hidden, belief = policy.act(obs_dict)
                if (
                    category in args.categories
                    and category_counts[category] < args.states_per_category
                    and int(obs_dict["action_mask"].sum()) > 1
                ):
                    heuristic_action = int(bot.choose_action(env, agent))
                    candidates = choose_actions_for_probe(env, args.max_actions)
                    for forced in [model_action, heuristic_action]:
                        if forced not in candidates:
                            candidates = [forced] + candidates
                    candidates = list(dict.fromkeys(candidates))[: args.max_actions]
                    eval_args = argparse.Namespace(
                        rollouts_per_action=args.rollouts_per_action,
                        player0_continuation=args.player0_continuation,
                    )
                    evaluated = evaluate_candidate_actions(
                        env,
                        candidates,
                        checkpoint,
                        opponents,
                        eval_args,
                        decision_seed=seed * 100 + turn,
                    )
                    state_id = len(states)
                    states.append(
                        {
                            "category": category,
                            "config": config_name,
                            "seed": seed,
                            "turn": turn,
                            "hidden": hidden,
                            "belief": belief,
                            "model_action": int(model_action),
                            "best_action": int(evaluated[0]["action"]),
                            "best_winrate": float(evaluated[0]["winrate"]),
                            "model_winrate": float(
                                next(row["winrate"] for row in evaluated if row["action"] == model_action)
                            ),
                        }
                    )
                    for row in evaluated:
                        score = float(row["winrate"] + args.reward_score_weight * row["mean_reward"])
                        rows.append(
                            {
                                "state_id": state_id,
                                "action": int(row["action"]),
                                "target": score,
                                "winrate": float(row["winrate"]),
                                "mean_reward": float(row["mean_reward"]),
                            }
                        )
                    category_counts[category] += 1
                    config_counts[config_name] += 1
                    candidate_counts[category] += len(evaluated)
                    if len(states) % args.log_every_states == 0:
                        logger.write(
                            "Collecte action-value",
                            expected="Accumuler des valeurs par action candidate, pas seulement un top label.",
                            actual=f"states={len(states)}, rows={len(rows)}, categories={dict(category_counts)}",
                            details={"config_counts": dict(config_counts), "candidate_counts": dict(candidate_counts)},
                        )
                env.step(model_action)
            else:
                env.step(opponent_action(env, agent, obs_dict, opponents, bot))

    if not states or not rows:
        raise RuntimeError("No action-value rows collected")

    regrets = [state["best_winrate"] - state["model_winrate"] for state in states]
    summary = {
        "states": len(states),
        "rows": len(rows),
        "category_counts": dict(category_counts),
        "config_counts": dict(config_counts),
        "candidate_counts": dict(candidate_counts),
        "mean_regret": float(np.mean(regrets)),
        "high_regret_rate_12pts": float(np.mean([regret >= 0.12 for regret in regrets])),
        "high_regret_rate_20pts": float(np.mean([regret >= 0.20 for regret in regrets])),
    }
    return states, rows, summary


def tensors_from_records(states, rows, device):
    hidden = np.array([states[row["state_id"]]["hidden"] for row in rows], dtype=np.float32)
    belief = np.array([states[row["state_id"]]["belief"] for row in rows], dtype=np.float32)
    model_action = np.array([states[row["state_id"]]["model_action"] for row in rows], dtype=np.int64)
    return {
        "hidden": torch.as_tensor(hidden, dtype=torch.float32, device=device),
        "belief": torch.as_tensor(belief, dtype=torch.float32, device=device),
        "model_action": torch.as_tensor(model_action, dtype=torch.long, device=device),
        "action": torch.as_tensor([row["action"] for row in rows], dtype=torch.long, device=device),
        "target": torch.as_tensor([row["target"] for row in rows], dtype=torch.float32, device=device),
        "winrate": torch.as_tensor([row["winrate"] for row in rows], dtype=torch.float32, device=device),
    }


def evaluate_rows(head, data):
    head.eval()
    with torch.no_grad():
        pred = head(data["hidden"], data["belief"], data["action"], data["model_action"])
        mse = torch.mean((pred - data["target"]) ** 2)
        mae = torch.mean(torch.abs(pred - data["target"]))
        corr = torch.corrcoef(torch.stack([pred, data["target"]]))[0, 1]
    head.train()
    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "corr": float(corr.item()) if torch.isfinite(corr) else 0.0,
        "pred_mean": float(pred.mean().item()),
        "target_mean": float(data["target"].mean().item()),
    }


def train_head(states, rows, args, logger):
    device = torch.device(args.device)
    data = tensors_from_records(states, rows, device)
    head = ActionValueHead(hidden_dim=args.hidden_dim, embed_dim=args.embed_dim).to(device)

    n = len(rows)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 303)
    perm = torch.randperm(n, generator=generator, device=device)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)
    history = []

    logger.write(
        "Debut entrainement action-value",
        expected="Predire une valeur par action candidate; correlation val positive attendue.",
        actual=f"train_rows={len(train_idx)}, val_rows={len(val_idx)}",
        details={"initial_all": evaluate_rows(head, data)},
    )

    for epoch in range(1, args.epochs + 1):
        losses = []
        for (idx,) in loader:
            pred = head(data["hidden"][idx], data["belief"][idx], data["action"][idx], data["model_action"][idx])
            target = data["target"][idx]
            weight = 1.0 + args.high_value_weight * data["winrate"][idx]
            loss = (((pred - target) ** 2) * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        train_data = {key: value[train_idx] for key, value in data.items()}
        val_data = {key: value[val_idx] for key, value in data.items()}
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": evaluate_rows(head, train_data),
            "val": evaluate_rows(head, val_data),
            "all": evaluate_rows(head, data),
        }
        history.append(row)
        logger.write(
            f"Epoch action-value {epoch}/{args.epochs}",
            expected="La MAE doit baisser sans effondrer la correlation de validation.",
            actual=f"loss={row['loss']:.4f}, val_mae={row['val']['mae']:.3f}, val_corr={row['val']['corr']:.3f}",
            details=row,
        )

    return head, history


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train a step3 action-value head.")
    parser.add_argument("--start", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--output", default="step3_action_value_head_attempt1.pth")
    parser.add_argument("--collect-games", type=int, default=5200)
    parser.add_argument("--states-per-category", type=int, default=80)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["guard", "priest", "spy", "king", "prince", "chancellor_card", "baron"],
    )
    parser.add_argument("--collect-configs", nargs="+", default=list(OPPONENT_CONFIGS.keys()))
    parser.add_argument("--rollouts-per-action", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=14)
    parser.add_argument("--reward-score-weight", type=float, default=0.05)
    parser.add_argument("--player0-continuation", choices=["heuristic", "model", "random"], default="heuristic")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--high-value-weight", type=float, default=1.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--log-every-states", type=int, default=40)
    parser.add_argument("--report", default="step3_action_value_head_attempt1_train.json")
    parser.add_argument("--run-log", default="step3_action_value/logs/2026-04-25_step3_action_value_head_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=10100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = resolve_checkpoint(args.start)
    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut step3 action-value head",
        expected="Reprendre l'etape regret sous forme Q/action-value, plus stable qu'un top-label.",
        actual=f"start={checkpoint}",
        details=vars(args),
    )

    states, rows, collection_summary = collect_action_value_rows(args, checkpoint, logger)
    logger.write(
        "Dataset action-value collecte",
        expected="Avoir beaucoup plus de supervision car chaque etat fournit plusieurs actions valuees.",
        actual=f"states={collection_summary['states']}, rows={collection_summary['rows']}",
        details=collection_summary,
    )
    head, history = train_head(states, rows, args, logger)

    output = Path(args.output)
    if output.parent == Path("."):
        output = CHECKPOINT_DIR / output
    payload = {
        "model_type": "step3_action_value_head_v1",
        "created_at": now_stamp(),
        "base_checkpoint": str(checkpoint),
        "head": head.cpu().state_dict(),
        "head_hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "categories": args.categories,
        "reward_score_weight": args.reward_score_weight,
        "metadata": {
            "args": vars(args),
            "collection_summary": collection_summary,
            "history": history,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    report = {
        "created_at": now_stamp(),
        "start": str(checkpoint),
        "output": str(output),
        "collection_summary": collection_summary,
        "history": history,
        "sample_states": [
            {
                key: value
                for key, value in state.items()
                if key not in {"hidden", "belief"}
            }
            | {
                "model_decoded": decode_action(state["model_action"]),
                "best_decoded": decode_action(state["best_action"]),
            }
            for state in states[:50]
        ],
    }
    report_path = Path(args.report)
    if report_path.parent == Path("."):
        report_path = REPORT_DIR / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.write(
        "Fin step3 action-value head",
        expected="Sauvegarder une Q-head candidate a evaluer en arena.",
        actual=f"checkpoint={output}, report={report_path}",
        details={"collection_summary": collection_summary, "final_metrics": history[-1]},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
