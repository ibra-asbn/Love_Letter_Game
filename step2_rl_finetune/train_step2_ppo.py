"""PPO fine-tuning from the step 1 heuristic student.

The goal is not to relearn the rules from scratch. The actor starts as the
heuristic student, keeps a decaying imitation anchor, and is optimized for real
game reward against random/heuristic mixes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefConditionedPolicy,
    BeliefHead,
    LATENT,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import (
    ExperimentLogger,
    STEP_CHECKPOINT_DIR,
    STEP_REPORT_DIR,
    arena_summary,
    composite_score,
    ensure_step_dirs,
    now_stamp,
    resolve_checkpoint,
)
from step2_rl_finetune.evaluate_step2 import run_evaluation


class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, hidden):
        return self.net(hidden).squeeze(-1)


@dataclass
class Batch:
    obs: torch.Tensor
    masks: torch.Tensor
    hidden_in: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    teacher_actions: torch.Tensor


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") != "belief_conditioned_actor_v1":
        raise ValueError(f"{path} is not a belief-conditioned checkpoint")

    encoder = BeliefConditionedEncoder()
    belief_head = BeliefHead()
    actor = BeliefConditionedActor()
    critic = ValueHead()
    encoder.load_state_dict(ckpt["encoder"])
    belief_head.load_state_dict(ckpt["belief_head"])
    actor.load_state_dict(ckpt["actor"])
    if "critic" in ckpt:
        critic.load_state_dict(ckpt["critic"])
    return encoder, belief_head, actor, critic


def save_checkpoint(path, encoder, belief_head, actor, critic, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "belief_conditioned_actor_v1",
            "encoder": encoder.cpu().state_dict(),
            "belief_head": belief_head.cpu().state_dict(),
            "actor": actor.cpu().state_dict(),
            "critic": critic.cpu().state_dict(),
            "metadata": metadata,
        },
        path,
    )


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def opponent_action(env, agent, obs_dict, n_heuristics, bot):
    seat_idx = int(agent.rsplit("_", 1)[1])
    if 1 <= seat_idx <= n_heuristics:
        return bot.choose_action(env, agent)
    return random_action(obs_dict)


def sample_n_heuristics(spec):
    if isinstance(spec, (list, tuple)):
        return int(np.random.choice(spec))
    return int(spec)


def forward_policy(encoder, belief_head, actor, obs, hidden_in, masks, train_encoder=False):
    if train_encoder:
        hidden = encoder.forward_hidden(obs, hidden_in)
        _belief_logits, belief_probs = belief_head(hidden)
    else:
        with torch.no_grad():
            hidden = encoder.forward_hidden(obs, hidden_in)
            _belief_logits, belief_probs = belief_head(hidden)
    logits = actor(hidden.detach() if not train_encoder else hidden, belief_probs.detach() if not train_encoder else belief_probs, masks)
    return logits, hidden


def collect_batch(encoder, belief_head, actor, critic, n_heuristics, target_steps, device, seed_offset, sample_actions):
    bot = HeuristicBot()
    records = []
    game_rewards = []
    game_lengths = []
    game_heuristic_counts = []
    game_idx = 0

    encoder.eval()
    belief_head.eval()
    actor.eval()
    critic.eval()

    while len(records) < target_steps:
        env = LoveLetterRLEnv(num_players=4)
        seed = seed_offset + game_idx
        np.random.seed(seed)
        env.reset(seed=seed)
        game_n_heuristics = sample_n_heuristics(n_heuristics)
        game_heuristic_counts.append(game_n_heuristics)
        game_idx += 1
        hidden_state = torch.zeros(1, LATENT, device=device)
        game_records = []
        total_reward = 0.0

        for agent in env.agent_iter():
            obs_dict, reward, term, trunc, _info = env.last()
            if agent == "player_0":
                total_reward += float(reward)

            if term or trunc:
                env.step(None)
                continue

            if agent == "player_0":
                teacher_action = bot.choose_action(env, agent)
                obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                hidden_in = hidden_state.detach()

                with torch.no_grad():
                    hidden_out = encoder.forward_hidden(obs_t, hidden_in)
                    _belief_logits, belief_probs = belief_head(hidden_out)
                    logits = actor(hidden_out, belief_probs, mask_t)
                    dist = Categorical(logits=logits)
                    if sample_actions:
                        action_t = dist.sample()
                    else:
                        action_t = logits.argmax(dim=-1)
                    logprob_t = dist.log_prob(action_t)
                    value_t = critic(hidden_out)

                action = int(action_t.item())
                hidden_state = hidden_out.detach()
                game_records.append(
                    {
                        "obs": obs_dict["observation"].copy(),
                        "mask": obs_dict["action_mask"].copy(),
                        "hidden_in": hidden_in.squeeze(0).cpu().numpy(),
                        "action": action,
                        "logprob": float(logprob_t.item()),
                        "value": float(value_t.item()),
                        "teacher_action": int(teacher_action),
                    }
                )
            else:
                action = opponent_action(env, agent, obs_dict, game_n_heuristics, bot)

            env.step(action)

        for rec in game_records:
            rec["return"] = total_reward
        records.extend(game_records)
        game_rewards.append(total_reward)
        game_lengths.append(len(game_records))

    records = records[:target_steps]
    return Batch(
        obs=torch.as_tensor(np.array([r["obs"] for r in records]), dtype=torch.float32, device=device),
        masks=torch.as_tensor(np.array([r["mask"] for r in records]), dtype=torch.bool, device=device),
        hidden_in=torch.as_tensor(np.array([r["hidden_in"] for r in records]), dtype=torch.float32, device=device),
        actions=torch.as_tensor([r["action"] for r in records], dtype=torch.long, device=device),
        old_logprobs=torch.as_tensor([r["logprob"] for r in records], dtype=torch.float32, device=device),
        returns=torch.as_tensor([r["return"] for r in records], dtype=torch.float32, device=device),
        values=torch.as_tensor([r["value"] for r in records], dtype=torch.float32, device=device),
        teacher_actions=torch.as_tensor([r["teacher_action"] for r in records], dtype=torch.long, device=device),
    ), {
        "mean_reward": float(np.mean(game_rewards)) if game_rewards else 0.0,
        "games": int(len(game_rewards)),
        "mean_actions_player0": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "mean_n_heuristics": float(np.mean(game_heuristic_counts)) if game_heuristic_counts else 0.0,
        "counts": {str(k): int(v) for k, v in zip(*np.unique(game_heuristic_counts, return_counts=True))}
        if game_heuristic_counts
        else {},
    }


def masked_kl_to_anchor(logits, anchor_logits, masks):
    log_p = F.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)
    q = F.softmax(anchor_logits.masked_fill(~masks, -1e9), dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def ppo_update(
    encoder,
    belief_head,
    actor,
    critic,
    anchor_encoder,
    anchor_belief,
    anchor_actor,
    optimizer,
    batch,
    args,
    bc_coef,
    anchor_coef,
):
    encoder.train(args.train_encoder)
    belief_head.train(args.train_encoder)
    actor.train()
    critic.train()

    advantages = batch.returns - batch.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = torch.arange(len(batch.actions), device=batch.actions.device)
    ce_teacher = nn.CrossEntropyLoss()
    metrics = []

    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(len(indices), device=indices.device)]
        for start in range(0, len(perm), args.minibatch_size):
            idx = perm[start : start + args.minibatch_size]
            logits, hidden = forward_policy(
                encoder,
                belief_head,
                actor,
                batch.obs[idx],
                batch.hidden_in[idx],
                batch.masks[idx],
                train_encoder=args.train_encoder,
            )
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(batch.actions[idx])
            entropy = dist.entropy().mean()
            values = critic(hidden.detach() if not args.train_encoder else hidden)

            ratio = torch.exp(logprobs - batch.old_logprobs[idx])
            adv = advantages[idx]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((values - batch.returns[idx]) ** 2).mean()
            teacher_loss = ce_teacher(logits, batch.teacher_actions[idx])

            anchor_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
            if anchor_coef > 0:
                with torch.no_grad():
                    anchor_hidden = anchor_encoder.forward_hidden(batch.obs[idx], batch.hidden_in[idx])
                    _anchor_belief_logits, anchor_probs = anchor_belief(anchor_hidden)
                    anchor_logits = anchor_actor(anchor_hidden, anchor_probs, batch.masks[idx])
                anchor_loss = masked_kl_to_anchor(logits, anchor_logits, batch.masks[idx])

            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + bc_coef * teacher_loss
                + anchor_coef * anchor_loss
            )

            optimizer.zero_grad()
            loss.backward()
            params = list(actor.parameters()) + list(critic.parameters())
            if args.train_encoder:
                params += list(encoder.parameters()) + list(belief_head.parameters())
            nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()

            metrics.append(
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "teacher_loss": float(teacher_loss.item()),
                    "anchor_loss": float(anchor_loss.item()),
                    "entropy": float(entropy.item()),
                    "bc_coef": float(bc_coef),
                    "anchor_coef": float(anchor_coef),
                }
            )

    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}


def make_policy(encoder, belief_head, actor):
    return BeliefConditionedPolicy(encoder.cpu(), belief_head.cpu(), actor.cpu()).eval()


def parse_phase_spec(text):
    phases = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        left, right = item.split(":")
        updates = int(right)
        if left == "mix":
            phases.append(([0, 1, 2, 3], updates))
        else:
            phases.append((int(left), updates))
    return phases


def linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    t = min(1.0, max(0.0, step / float(total_steps - 1)))
    return float(start + t * (end - start))


def train(args):
    ensure_step_dirs()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_path = resolve_checkpoint(args.start)
    encoder, belief_head, actor, critic = load_model(start_path)
    anchor_encoder, anchor_belief, anchor_actor, _anchor_critic = load_model(start_path)

    encoder.to(device)
    belief_head.to(device)
    actor.to(device)
    critic.to(device)
    anchor_encoder.to(device).eval()
    anchor_belief.to(device).eval()
    anchor_actor.to(device).eval()
    for module in [anchor_encoder, anchor_belief, anchor_actor]:
        for param in module.parameters():
            param.requires_grad_(False)

    for param in encoder.parameters():
        param.requires_grad_(args.train_encoder)
    for param in belief_head.parameters():
        param.requires_grad_(args.train_encoder)

    params = list(actor.parameters()) + list(critic.parameters())
    if args.train_encoder:
        params += list(encoder.parameters()) + list(belief_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    output_prefix = args.output_prefix
    best_path = STEP_CHECKPOINT_DIR / f"{output_prefix}_best.pth"
    final_path = STEP_CHECKPOINT_DIR / f"{output_prefix}_final.pth"
    report_path = STEP_REPORT_DIR / f"{output_prefix}_train.json"
    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()

    phases = parse_phase_spec(args.phase_spec)
    total_updates = sum(updates for _n, updates in phases)
    seed_cursor = args.seed * 10_000
    history = []
    eval_history = []

    logger.write(
        "Demarrage etape 2 PPO",
        expected=(
            "Partir du student heuristique, ameliorer le score composite contre les mixes, "
            "et ne sauvegarder comme best qu'un checkpoint qui bat le point de depart."
        ),
        actual=f"start={start_path}, output_prefix={output_prefix}, phases={args.phase_spec}",
        details=vars(args),
    )

    baseline_report = run_evaluation(
        checkpoint=start_path,
        games=args.baseline_games,
        seed_start=args.eval_seed_start,
        include_role_tables=False,
    )
    baseline_score = baseline_report["model_composite"]
    heuristic_score = baseline_report["heuristic_composite"]
    success_threshold = max(baseline_score + args.min_improvement, heuristic_score + args.min_heuristic_margin)
    best_score = baseline_score
    best_label = "start"

    logger.write(
        "Baseline avant entrainement",
        expected="Mesurer le point de depart et fixer le seuil de succes.",
        actual=(
            f"baseline={baseline_score:.5f}, heuristic={heuristic_score:.5f}, "
            f"success_threshold={success_threshold:.5f}"
        ),
        details={
            "model": arena_summary(baseline_report["model_configs"]),
            "heuristic": arena_summary(baseline_report["heuristic_configs"]),
        },
    )

    def evaluate_and_maybe_save(label, update_index):
        nonlocal best_score, best_label
        save_checkpoint(
            STEP_CHECKPOINT_DIR / f"{output_prefix}_{label}.pth",
            encoder,
            belief_head,
            actor,
            critic,
            {"label": label, "temporary": True, "update_index": update_index},
        )
        encoder.to(device)
        belief_head.to(device)
        actor.to(device)
        critic.to(device)
        candidate_path = STEP_CHECKPOINT_DIR / f"{output_prefix}_{label}.pth"
        report = run_evaluation(
            checkpoint=candidate_path,
            games=args.eval_games,
            seed_start=args.eval_seed_start,
            include_role_tables=False,
        )
        score = report["model_composite"]
        improved = score > best_score
        success = score >= success_threshold
        eval_record = {
            "label": label,
            "update_index": update_index,
            "score": score,
            "delta_vs_baseline": score - baseline_score,
            "delta_vs_heuristic": score - heuristic_score,
            "success": success,
            "report": report,
            "created_at": now_stamp(),
        }
        eval_history.append(eval_record)
        if improved:
            best_score = score
            best_label = label
            save_checkpoint(
                best_path,
                encoder,
                belief_head,
                actor,
                critic,
                {
                    "label": label,
                    "best_score": best_score,
                    "baseline_score": baseline_score,
                    "heuristic_score": heuristic_score,
                    "success_threshold": success_threshold,
                    "history": history,
                    "eval_history": eval_history,
                },
            )
            encoder.to(device)
            belief_head.to(device)
            actor.to(device)
            critic.to(device)
        logger.write(
            f"Evaluation {label}",
            expected=f"Depasser {success_threshold:.5f} pour declarer un succes provisoire.",
            actual=(
                f"score={score:.5f}, best={best_score:.5f}, "
                f"delta_baseline={score - baseline_score:.5f}, success={success}"
            ),
            details=arena_summary(report["model_configs"]),
        )
        return success

    global_update = 0
    success_seen = False
    for phase_idx, (n_heuristics, n_updates) in enumerate(phases, start=1):
        phase_name = "mix" if isinstance(n_heuristics, list) else f"{n_heuristics}H"
        logger.write(
            f"Phase {phase_idx} - {phase_name}",
            expected=(
                "Explorer contre cette composition, avec imitation decroissante pour garder les bases "
                "sans rester prisonnier du teacher."
            ),
        )
        for update in range(1, n_updates + 1):
            bc_coef = linear_schedule(args.bc_coef_start, args.bc_coef_end, global_update, total_updates)
            anchor_coef = linear_schedule(args.anchor_coef_start, args.anchor_coef_end, global_update, total_updates)
            batch, collect_meta = collect_batch(
                encoder,
                belief_head,
                actor,
                critic,
                n_heuristics=n_heuristics,
                target_steps=args.steps_per_update,
                device=device,
                seed_offset=seed_cursor,
                sample_actions=not args.greedy_collect,
            )
            seed_cursor += collect_meta["games"] + 17
            metrics = ppo_update(
                encoder,
                belief_head,
                actor,
                critic,
                anchor_encoder,
                anchor_belief,
                anchor_actor,
                optimizer,
                batch,
                args,
                bc_coef=bc_coef,
                anchor_coef=anchor_coef,
            )
            global_update += 1
            row = {
                "phase": phase_idx,
                "phase_name": phase_name,
                "update": update,
                "global_update": global_update,
                "collect": collect_meta,
                **metrics,
            }
            history.append(row)
            print(
                f"phase={phase_idx} update={update:03d}/{n_updates} "
                f"reward={collect_meta['mean_reward']:.3f} entropy={metrics['entropy']:.3f} "
                f"bc={bc_coef:.4f} anchor={anchor_coef:.4f}",
                flush=True,
            )
            if args.eval_every_updates > 0 and global_update % args.eval_every_updates == 0:
                success_seen = evaluate_and_maybe_save(f"u{global_update}", global_update) or success_seen
                if success_seen and args.stop_on_success:
                    logger.write(
                        "Arret sur succes provisoire",
                        expected="Stopper quand le seuil court est atteint, puis confirmer en evaluation longue.",
                        actual=f"best_score={best_score:.5f}, best_label={best_label}",
                    )
                    break
        if success_seen and args.stop_on_success:
            break

    save_checkpoint(
        final_path,
        encoder,
        belief_head,
        actor,
        critic,
        {
            "label": "final",
            "baseline_score": baseline_score,
            "heuristic_score": heuristic_score,
            "success_threshold": success_threshold,
            "history": history,
            "eval_history": eval_history,
        },
    )

    final_report = run_evaluation(
        checkpoint=final_path,
        games=args.eval_games,
        seed_start=args.eval_seed_start,
        include_role_tables=False,
    )
    final_score = final_report["model_composite"]
    if final_score > best_score:
        best_score = final_score
        best_label = "final"
        save_checkpoint(
            best_path,
            encoder,
            belief_head,
            actor,
            critic,
            {
                "label": "final",
                "best_score": best_score,
                "baseline_score": baseline_score,
                "heuristic_score": heuristic_score,
                "success_threshold": success_threshold,
                "history": history,
                "eval_history": eval_history,
            },
        )

    long_report = None
    if best_path.exists() and (best_score >= success_threshold or args.confirm_long_always):
        long_report = run_evaluation(
            checkpoint=best_path,
            games=args.confirm_games,
            seed_start=args.confirm_seed_start,
            include_role_tables=not args.skip_confirm_roles,
            role_games=args.confirm_role_games,
        )
        logger.write(
            "Confirmation longue du meilleur checkpoint",
            expected="Verifier que le succes court tient sur plus de parties.",
            actual=(
                f"long_score={long_report['model_composite']:.5f}, "
                f"heuristic={long_report['heuristic_composite']:.5f}, "
                f"delta={long_report['model_minus_heuristic_composite']:.5f}"
            ),
            details=arena_summary(long_report["model_configs"]),
        )

    report = {
        "created_at": now_stamp(),
        "start": str(start_path),
        "final": str(final_path),
        "best": str(best_path) if best_path.exists() else None,
        "baseline_report": baseline_report,
        "baseline_score": baseline_score,
        "heuristic_score": heuristic_score,
        "success_threshold": success_threshold,
        "best_score": best_score,
        "best_label": best_label,
        "final_score": final_score,
        "final_report": final_report,
        "long_report": long_report,
        "args": vars(args),
        "history": history,
        "eval_history": eval_history,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.write(
        "Fin entrainement etape 2",
        expected="Obtenir un checkpoint qui depasse le student heuristique et HeuristicBot.",
        actual=(
            f"best_score={best_score:.5f}, final_score={final_score:.5f}, "
            f"success_threshold={success_threshold:.5f}, best={best_path}"
        ),
        details={
            "final": arena_summary(final_report["model_configs"]),
            "long": arena_summary(long_report["model_configs"]) if long_report else None,
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2 PPO from heuristic student.")
    parser.add_argument("--start", default="heuristic_student_attempt4_player0_chancellor_order.pth")
    parser.add_argument("--output-prefix", default="step2_ppo_attempt1")
    parser.add_argument("--phase-spec", default="0:8,1:8,2:8,3:8,mix:16")
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--bc-coef-start", type=float, default=0.08)
    parser.add_argument("--bc-coef-end", type=float, default=0.005)
    parser.add_argument("--anchor-coef-start", type=float, default=0.03)
    parser.add_argument("--anchor-coef-end", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--greedy-collect", action="store_true")
    parser.add_argument("--baseline-games", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--eval-every-updates", type=int, default=8)
    parser.add_argument("--eval-seed-start", type=int, default=700000)
    parser.add_argument("--confirm-games", type=int, default=5000)
    parser.add_argument("--confirm-role-games", type=int, default=1000)
    parser.add_argument("--confirm-seed-start", type=int, default=710000)
    parser.add_argument("--confirm-long-always", action="store_true")
    parser.add_argument("--skip-confirm-roles", action="store_true")
    parser.add_argument("--min-improvement", type=float, default=0.003)
    parser.add_argument("--min-heuristic-margin", type=float, default=0.001)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--run-log", default="step2_rl_finetune/logs/2026-04-24_step2_ppo_attempt1.md")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

