"""Train a Step7 self-play candidate against the active league.

The candidate is a composed policy:

- trainable belief actor + critic;
- frozen Step3 advantage head;
- frozen Step5 Chancellor/Baron/Prince execution modules.

This file intentionally keeps the first self-play loop conservative.  We update
only the actor and critic, with a strong KL anchor toward the parent actor and
an auxiliary BC loss toward the final post-head action.  The league can become
more ambitious later, but v1 should first produce reliable candidates without
destroying the tactical modules we already validated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import generic_candidate_actions
from love_letter.belief_actor import LATENT
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step1_heuristic_mastery.common import (
    absolute_to_relative_action,
    absolute_to_relative_mask,
    relative_to_absolute_action,
)
from step2_rl_finetune.common import ExperimentLogger, now_stamp, resolve_checkpoint
from step2_rl_finetune.train_step2_ppo import (
    Batch,
    load_model,
    masked_kl_to_anchor,
    save_checkpoint,
)
from step3_action_value.evaluate_advantage_head_v2 import dynamic_margin, load_advantage_bundle
from step3_action_value.mini_rollout_probe import classify_state
from step3_action_value.train_advantage_head_v2 import state_features
from step5_execution_heads.chancellor_head import load_chancellor_head
from step5_execution_heads.evaluate_combined_three_heads import Step5ThreeSeat
from step7_self_play_league.league_policy import (
    DEFAULT_PROMOTION_PATH,
    DEFAULT_ROSTER_PATH,
    LeaguePolicyFactory,
    LeagueRuntimeArgs,
    active_policies,
    append_jsonl,
    load_roster,
    make_candidate_roster_entry,
    policy_by_id,
    save_roster,
    upsert_policy,
)


STEP_DIR = PROJECT_ROOT / "step7_self_play_league"
CHECKPOINT_DIR = STEP_DIR / "checkpoints"
REPORT_DIR = STEP_DIR / "reports"
LOG_DIR = STEP_DIR / "logs"


def ensure_dirs() -> None:
    for path in [CHECKPOINT_DIR, REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path, PROJECT_ROOT / path, CHECKPOINT_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


@dataclass
class TrainRecord:
    obs: np.ndarray
    mask: np.ndarray
    hidden_in: np.ndarray
    action: int
    logprob: float
    value: float
    teacher_action: int


class TrainableAdvantageBase:
    """Trainable base actor wrapped with the frozen Step3 advantage head."""

    def __init__(
        self,
        encoder,
        belief_head,
        actor,
        critic,
        advantage_head,
        advantage_categories: list[str],
        args,
        sample_actions: bool,
    ):
        self.encoder = encoder
        self.belief_head = belief_head
        self.actor = actor
        self.critic = critic
        self.head = advantage_head.to(args.device).eval()
        self.categories = set(advantage_categories)
        self.max_actions = int(args.max_actions)
        self.override_margin = float(args.override_margin)
        self.device = torch.device(args.device)
        self.sample_actions = bool(sample_actions)
        self.state = None
        self.bot = HeuristicBot(shuffle_targets=True)
        self.pending: dict | None = None
        self.records: list[TrainRecord] = []
        self.stats = Counter()

    def _score_advantage(self, obs, hidden, belief, extra, actions, model_action, heuristic_action):
        n = len(actions)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        hidden_t = torch.as_tensor(hidden, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        belief_t = torch.as_tensor(belief, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1, -1)
        extra_t = torch.as_tensor(extra, dtype=torch.float32, device=self.device).unsqueeze(0).expand(n, -1)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        model_t = torch.full_like(actions_t, int(model_action))
        heuristic_t = torch.full_like(actions_t, int(heuristic_action))
        with torch.no_grad():
            scores = self.head(obs_t, hidden_t, belief_t, extra_t, actions_t, model_t, heuristic_t)
        model_positions = [idx for idx, action in enumerate(actions) if int(action) == int(model_action)]
        if model_positions:
            scores = scores - scores[model_positions[0]]
        return scores

    def act(self, env, obs_dict, agent: str) -> int:
        my_idx = int(agent.rsplit("_", 1)[1])
        mask_rel_np = absolute_to_relative_mask(obs_dict["action_mask"], my_idx)
        obs_np = obs_dict["observation"].astype(np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_rel_np, dtype=torch.bool, device=self.device).unsqueeze(0)
        hidden_in = self.state.detach() if self.state is not None else torch.zeros(1, LATENT, device=self.device)

        with torch.no_grad():
            hidden_out = self.encoder.forward_hidden(obs_t, hidden_in)
            _belief_logits, belief_probs = self.belief_head(hidden_out)
            logits = self.actor(hidden_out, belief_probs, mask_t)
            dist = Categorical(logits=logits)
            action_t = dist.sample() if self.sample_actions else logits.argmax(dim=-1)
            base_action_rel = int(action_t.item())
            value_t = self.critic(hidden_out)

        self.state = hidden_out.detach()
        base_action_abs = relative_to_absolute_action(base_action_rel, my_idx)
        chosen_abs = int(base_action_abs)
        category = classify_state(env, agent)
        self.stats["decisions"] += 1

        if category in self.categories and int(obs_dict["action_mask"].sum()) > 1:
            heuristic_abs = int(self.bot.choose_action(env, agent))
            heuristic_rel = absolute_to_relative_action(heuristic_abs, my_idx)
            candidate_abs = generic_candidate_actions(env, agent, base_action_abs, heuristic_abs, self.max_actions)
            pairs = []
            seen_rel = set()
            for action_abs in candidate_abs:
                action_rel = absolute_to_relative_action(action_abs, my_idx)
                if action_rel in seen_rel:
                    continue
                pairs.append((int(action_abs), int(action_rel)))
                seen_rel.add(action_rel)
            if base_action_rel not in seen_rel:
                pairs.insert(0, (int(base_action_abs), int(base_action_rel)))

            actions_abs = [pair[0] for pair in pairs]
            actions_rel = [pair[1] for pair in pairs]
            belief_np = belief_probs.detach().cpu().squeeze(0).numpy().astype(np.float32)
            extra = state_features(env, belief_np, mask_rel_np)
            scores = self._score_advantage(
                obs_np,
                hidden_out.detach().cpu().squeeze(0).numpy().astype(np.float32),
                belief_np,
                extra,
                actions_rel,
                base_action_rel,
                heuristic_rel,
            )
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
            margin = dynamic_margin(self.override_margin, 0.0, belief_np)
            self.stats["advantage_checks"] += 1
            if actions_abs[best_idx] != base_action_abs and best_score >= margin:
                chosen_abs = int(actions_abs[best_idx])
                self.stats["advantage_overrides"] += 1

        self.pending = {
            "obs": obs_np.copy(),
            "mask": mask_rel_np.copy(),
            "hidden_in": hidden_in.detach().cpu().squeeze(0).numpy().astype(np.float32),
            "logits": logits.detach(),
            "value": float(value_t.item()),
            "base_action_rel": int(base_action_rel),
            "candidate_agent_idx": int(my_idx),
        }
        return int(chosen_abs)

    def finalize(self, final_action_abs: int) -> None:
        if self.pending is None:
            return
        my_idx = int(self.pending["candidate_agent_idx"])
        final_rel = absolute_to_relative_action(int(final_action_abs), my_idx)
        mask = self.pending["mask"]
        if final_rel < 0 or final_rel >= len(mask) or not bool(mask[final_rel]):
            self.stats["skipped_invalid_final"] += 1
            self.pending = None
            return
        dist = Categorical(logits=self.pending["logits"])
        action_t = torch.as_tensor([final_rel], dtype=torch.long, device=self.pending["logits"].device)
        logprob = float(dist.log_prob(action_t).item())
        self.records.append(
            TrainRecord(
                obs=self.pending["obs"],
                mask=mask,
                hidden_in=self.pending["hidden_in"],
                action=int(final_rel),
                logprob=logprob,
                value=float(self.pending["value"]),
                teacher_action=int(final_rel),
            )
        )
        if final_rel != self.pending["base_action_rel"]:
            self.stats["final_overrides"] += 1
        self.pending = None


class TrainingCompositeSeat:
    """Candidate seat: trainable base + frozen Step3/Step5 corrections."""

    def __init__(self, base: TrainableAdvantageBase, chancellor_head, args):
        self.base = base
        self.step5 = Step5ThreeSeat(
            base,
            chancellor_head,
            args,
            use_chancellor=True,
            use_baron=True,
            use_prince=True,
        )

    def act(self, env, obs_dict, agent: str) -> int:
        action = int(self.step5.act(env, obs_dict, agent))
        self.base.finalize(action)
        return action

    @property
    def records(self) -> list[TrainRecord]:
        return self.base.records

    @property
    def stats(self) -> Counter:
        stats = Counter(self.base.stats)
        stats.update({f"chancellor_{key}": value for key, value in self.step5.chancellor_stats.items()})
        stats.update({f"baron_{key}": value for key, value in self.step5.baron_stats.items()})
        stats.update({f"prince_{key}": value for key, value in self.step5.prince_stats.items()})
        return stats


def make_training_candidate(encoder, belief_head, actor, critic, advantage_head, categories, chancellor_head, args, sample_actions):
    base = TrainableAdvantageBase(
        encoder=encoder,
        belief_head=belief_head,
        actor=actor,
        critic=critic,
        advantage_head=advantage_head,
        advantage_categories=categories,
        args=args,
        sample_actions=sample_actions,
    )
    return TrainingCompositeSeat(base, chancellor_head, args)


def choose_parent(roster: dict, parent_id: str | None) -> dict:
    policies = policy_by_id(roster)
    if parent_id:
        if parent_id not in policies:
            raise ValueError(f"Parent not found in roster: {parent_id}")
        return policies[parent_id]
    active = active_policies(roster)
    if not active:
        raise ValueError("No active policies in roster.")
    return max(active, key=lambda item: float(item.get("elo", 1500.0)))


def opponent_softmax(active: list[dict], rng: np.random.Generator, temperature: float) -> dict:
    elos = np.asarray([float(policy.get("elo", 1500.0)) for policy in active], dtype=np.float64)
    scaled = (elos - elos.max()) / max(float(temperature), 1e-6)
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return active[int(rng.choice(len(active), p=probs))]


def choose_opponent(active: list[dict], candidate_snapshot: dict | None, rng: np.random.Generator, args) -> dict:
    roll = float(rng.random())
    if candidate_snapshot is not None and roll < args.snapshot_prob:
        return candidate_snapshot
    if roll < args.snapshot_prob + args.softmax_prob:
        return opponent_softmax(active, rng, args.elo_softmax_temperature)
    return active[int(rng.integers(0, len(active)))]


def collect_batch(
    encoder,
    belief_head,
    actor,
    critic,
    advantage_head,
    advantage_categories,
    chancellor_head,
    parent_snapshot: dict | None,
    active: list[dict],
    factory: LeaguePolicyFactory,
    target_steps: int,
    args,
    seed_offset: int,
) -> tuple[Batch, dict]:
    env = LoveLetterRLEnv(num_players=4)
    seats = [f"player_{idx}" for idx in range(4)]
    records: list[dict] = []
    game_rewards = []
    candidate_seats = Counter()
    opponent_counts = Counter()
    module_stats = Counter()
    rng = np.random.default_rng(seed_offset)
    game_idx = 0

    encoder.eval()
    belief_head.eval()
    actor.eval()
    critic.eval()

    while len(records) < target_steps:
        seed = seed_offset + game_idx
        np.random.seed(seed)
        env.reset(seed=seed, options={"starting_agent": seats[game_idx % 4]})
        candidate_agent = seats[int(rng.integers(0, 4))]
        candidate_seats[candidate_agent] += 1
        candidate_policy = make_training_candidate(
            encoder,
            belief_head,
            actor,
            critic,
            advantage_head,
            advantage_categories,
            chancellor_head,
            args,
            sample_actions=not args.greedy_collect,
        )
        roles = {seat: "model" for seat in seats}
        seat_policies = {candidate_agent: candidate_policy}
        for seat in seats:
            if seat == candidate_agent:
                continue
            spec = choose_opponent(active, parent_snapshot, rng, args)
            opponent_counts[spec["policy_id"]] += 1
            seat_policies[seat] = factory.make(spec, seat, roles)

        rewards = {seat: 0.0 for seat in seats}
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            rewards[agent] += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            action = int(seat_policies[agent].act(env, obs_dict, agent))
            env.step(action)

        reward = float(rewards[candidate_agent])
        for rec in candidate_policy.records:
            records.append(
                {
                    "obs": rec.obs,
                    "mask": rec.mask,
                    "hidden_in": rec.hidden_in,
                    "action": rec.action,
                    "logprob": rec.logprob,
                    "value": rec.value,
                    "teacher_action": rec.teacher_action,
                    "return": reward,
                }
            )
        game_rewards.append(reward)
        module_stats.update(candidate_policy.stats)
        game_idx += 1

    records = records[:target_steps]
    batch = Batch(
        obs=torch.as_tensor(np.asarray([r["obs"] for r in records]), dtype=torch.float32, device=args.device),
        masks=torch.as_tensor(np.asarray([r["mask"] for r in records]), dtype=torch.bool, device=args.device),
        hidden_in=torch.as_tensor(np.asarray([r["hidden_in"] for r in records]), dtype=torch.float32, device=args.device),
        actions=torch.as_tensor([r["action"] for r in records], dtype=torch.long, device=args.device),
        old_logprobs=torch.as_tensor([r["logprob"] for r in records], dtype=torch.float32, device=args.device),
        returns=torch.as_tensor([r["return"] for r in records], dtype=torch.float32, device=args.device),
        values=torch.as_tensor([r["value"] for r in records], dtype=torch.float32, device=args.device),
        teacher_actions=torch.as_tensor([r["teacher_action"] for r in records], dtype=torch.long, device=args.device),
    )
    meta = {
        "games": int(len(game_rewards)),
        "mean_reward": float(np.mean(game_rewards)) if game_rewards else 0.0,
        "score_ge_1": float(np.mean([reward >= 1.0 for reward in game_rewards])) if game_rewards else 0.0,
        "candidate_seats": dict(candidate_seats),
        "opponents": dict(opponent_counts),
        "module_stats": dict(module_stats),
    }
    return batch, meta


def ppo_update(
    encoder,
    belief_head,
    actor,
    critic,
    anchor_encoder,
    anchor_belief,
    anchor_actor,
    optimizer,
    batch: Batch,
    args,
) -> dict:
    encoder.eval()
    belief_head.eval()
    actor.train()
    critic.train()
    advantages = batch.returns - batch.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = torch.arange(len(batch.actions), device=batch.actions.device)
    ce_loss = nn.CrossEntropyLoss()
    metrics = []

    for _epoch in range(args.ppo_epochs):
        perm = indices[torch.randperm(len(indices), device=indices.device)]
        for start in range(0, len(perm), args.minibatch_size):
            idx = perm[start : start + args.minibatch_size]
            with torch.no_grad():
                hidden = encoder.forward_hidden(batch.obs[idx], batch.hidden_in[idx])
                _belief_logits, belief_probs = belief_head(hidden)
            logits = actor(hidden, belief_probs, batch.masks[idx])
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(batch.actions[idx])
            entropy = dist.entropy().mean()
            values = critic(hidden)
            ratio = torch.exp(logprobs - batch.old_logprobs[idx])
            adv = advantages[idx]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((values - batch.returns[idx]) ** 2).mean()
            bc_loss = ce_loss(logits, batch.teacher_actions[idx])

            with torch.no_grad():
                anchor_hidden = anchor_encoder.forward_hidden(batch.obs[idx], batch.hidden_in[idx])
                _anchor_belief_logits, anchor_probs = anchor_belief(anchor_hidden)
                anchor_logits = anchor_actor(anchor_hidden, anchor_probs, batch.masks[idx])
            kl_loss = masked_kl_to_anchor(logits, anchor_logits, batch.masks[idx])

            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + args.bc_coef * bc_loss
                + args.kl_coef * kl_loss
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm)
            optimizer.step()
            metrics.append(
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "bc_loss": float(bc_loss.item()),
                    "kl_loss": float(kl_loss.item()),
                    "entropy": float(entropy.item()),
                    "mean_ratio": float(ratio.mean().item()),
                }
            )
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def write_training_report(payload: dict, path: Path) -> None:
    lines = [
        "# Step7 - Self-Play Candidate Training",
        "",
        f"Date: {payload['created_at']}.",
        "",
        f"Parent: `{payload['parent_id']}`.",
        f"Candidate: `{payload['candidate_id']}`.",
        "",
        "| Iter | Decisions | Games | Reward | Score >=1 | Entropy | KL | BC | Checkpoint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["history"]:
        lines.append(
            f"| {row['iteration']} | {row['decisions']} | {row['collect']['games']} | "
            f"{row['collect']['mean_reward']:.4f} | {pct(row['collect']['score_ge_1'])} | "
            f"{row['update']['entropy']:.4f} | {row['update']['kl_loss']:.4f} | "
            f"{row['update']['bc_loss']:.4f} | `{Path(row['checkpoint']).name}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_checkpoint_manifest(args, candidate_id: str, parent_id: str, history: list[dict], latest_checkpoint: Path) -> None:
    manifest = {
        "created_at": now_stamp(),
        "candidate_id": candidate_id,
        "parent_id": parent_id,
        "latest_checkpoint": str(latest_checkpoint),
        "checkpoints": [
            {
                "iteration": int(row["iteration"]),
                "checkpoint": row["checkpoint"],
                "collect_mean_reward": float(row["collect"]["mean_reward"]),
                "collect_score_ge_1": float(row["collect"]["score_ge_1"]),
                "kl_loss": float(row["update"]["kl_loss"]),
                "entropy": float(row["update"]["entropy"]),
            }
            for row in history
        ],
        "training_args": vars(args),
    }
    path = REPORT_DIR / f"{args.output_prefix}_checkpoint_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def train(args) -> dict:
    ensure_dirs()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    roster = load_roster(args.roster)
    parent = choose_parent(roster, args.parent_id)
    parent_id = parent["policy_id"]
    parent_checkpoint = resolve_checkpoint(parent.get("base_checkpoint") or parent.get("checkpoint"))
    advantage_checkpoint, _default_base, advantage_head, advantage_ckpt = load_advantage_bundle(
        parent.get("advantage_checkpoint", args.advantage_checkpoint),
        None,
    )
    chancellor_head = load_chancellor_head(resolve_project_path(args.chancellor_head), device)[0]
    encoder, belief_head, actor, critic = load_model(parent_checkpoint)
    anchor_encoder, anchor_belief, anchor_actor, _anchor_critic = load_model(parent_checkpoint)

    for module in [encoder, belief_head, actor, critic, anchor_encoder, anchor_belief, anchor_actor]:
        module.to(device)
    for module in [encoder, belief_head, anchor_encoder, anchor_belief, anchor_actor, advantage_head, chancellor_head]:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)
    actor.train()
    critic.train()

    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    runtime = LeagueRuntimeArgs(
        device=args.device,
        override_margin=args.override_margin,
        max_actions=args.max_actions or int(advantage_ckpt.get("max_actions", 14)),
        chancellor_margin=args.chancellor_margin,
        retarget_margin=args.retarget_margin,
        veto_score=args.veto_score,
        force_score=args.force_score,
        self_force_score=args.self_force_score,
        min_princess_prob=args.min_princess_prob,
        example_limit=args.example_limit,
    )
    runtime_ns = runtime.namespace()
    runtime_ns.device = device
    runtime_ns.max_actions = int(runtime.max_actions)
    runtime_ns.greedy_collect = bool(args.greedy_collect)
    runtime_ns.snapshot_prob = float(args.snapshot_prob)
    runtime_ns.softmax_prob = float(args.softmax_prob)
    runtime_ns.elo_softmax_temperature = float(args.elo_softmax_temperature)
    factory = LeaguePolicyFactory(runtime)
    active = active_policies(roster)
    active = [policy for policy in active if policy["policy_id"] != args.candidate_id]
    candidate_snapshot = None

    logger = ExperimentLogger(args.run_log)
    if args.run_log:
        logger.reset()
    logger.write(
        "Debut Step7 self-play candidate",
        expected="Entrainer acteur+critic contre la ligue active, avec KL parent et BC post-head.",
        actual=f"parent={parent_id}, parent_checkpoint={parent_checkpoint}, active={len(active)}",
        details=vars(args),
    )

    history = []
    seed_cursor = args.seed * 10000
    last_checkpoint = None
    categories = list(advantage_ckpt.get("categories", []))
    for iteration in range(1, args.iterations + 1):
        batch, collect_meta = collect_batch(
            encoder,
            belief_head,
            actor,
            critic,
            advantage_head,
            categories,
            chancellor_head,
            candidate_snapshot,
            active,
            factory,
            args.steps_per_iteration,
            runtime_ns,
            seed_cursor,
        )
        seed_cursor += collect_meta["games"] + 31
        update_meta = ppo_update(
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
        )
        checkpoint = CHECKPOINT_DIR / f"{args.output_prefix}_iter_{iteration:04d}_candidate.pth"
        save_checkpoint(
            checkpoint,
            encoder,
            belief_head,
            actor,
            critic,
            {
                "step": "step7_self_play_league",
                "candidate_id": args.candidate_id,
                "parent_id": parent_id,
                "parent_checkpoint": str(parent_checkpoint),
                "advantage_checkpoint": str(advantage_checkpoint),
                "chancellor_head": args.chancellor_head,
                "iteration": iteration,
                "created_at": now_stamp(),
                "roster": args.roster,
                "training_args": vars(args),
            },
        )
        for module in [encoder, belief_head, actor, critic]:
            module.to(device)
        last_checkpoint = checkpoint
        candidate_snapshot = make_candidate_roster_entry(
            policy_id=f"{args.candidate_id}_snapshot_iter_{iteration:04d}",
            checkpoint=str(checkpoint.relative_to(PROJECT_ROOT)),
            parent_id=parent_id,
            advantage_checkpoint=str(advantage_checkpoint.relative_to(PROJECT_ROOT)),
            chancellor_head=args.chancellor_head,
            elo=float(parent.get("elo", 1500.0)),
            active=False,
        )
        row = {
            "iteration": iteration,
            "decisions": int(args.steps_per_iteration),
            "collect": collect_meta,
            "update": update_meta,
            "checkpoint": str(checkpoint),
        }
        history.append(row)
        write_checkpoint_manifest(args, args.candidate_id, parent_id, history, checkpoint)
        logger.write(
            f"Iteration {iteration}/{args.iterations}",
            expected="Ameliorer doucement sans quitter la region de confiance du parent.",
            actual=(
                f"reward={collect_meta['mean_reward']:.4f}, score>=1={pct(collect_meta['score_ge_1'])}, "
                f"entropy={update_meta['entropy']:.4f}, kl={update_meta['kl_loss']:.4f}"
            ),
            details=row,
        )

    final_checkpoint = CHECKPOINT_DIR / f"{args.output_prefix}_final_candidate.pth"
    save_checkpoint(
        final_checkpoint,
        encoder,
        belief_head,
        actor,
        critic,
        {
            "step": "step7_self_play_league",
            "candidate_id": args.candidate_id,
            "parent_id": parent_id,
            "parent_checkpoint": str(parent_checkpoint),
            "advantage_checkpoint": str(advantage_checkpoint),
            "chancellor_head": args.chancellor_head,
            "created_at": now_stamp(),
            "history": history,
            "training_args": vars(args),
        },
    )
    last_checkpoint = final_checkpoint
    write_checkpoint_manifest(args, args.candidate_id, parent_id, history, final_checkpoint)

    candidate_entry = make_candidate_roster_entry(
        policy_id=args.candidate_id,
        checkpoint=str(last_checkpoint.relative_to(PROJECT_ROOT)),
        parent_id=parent_id,
        advantage_checkpoint=str(advantage_checkpoint.relative_to(PROJECT_ROOT)),
        chancellor_head=args.chancellor_head,
        elo=float(parent.get("elo", 1500.0)),
        active=False,
    )
    if args.register_candidate:
        roster = upsert_policy(roster, candidate_entry)
        save_roster(roster, args.roster)
        append_jsonl(
            args.promotion_history,
            {
                "event": "candidate_registered",
                "created_at": now_stamp(),
                "candidate_id": args.candidate_id,
                "parent_id": parent_id,
                "checkpoint": str(last_checkpoint.relative_to(PROJECT_ROOT)),
            },
        )

    payload = {
        "created_at": now_stamp(),
        "candidate_id": args.candidate_id,
        "parent_id": parent_id,
        "parent_checkpoint": str(parent_checkpoint),
        "final_checkpoint": str(last_checkpoint),
        "registered": bool(args.register_candidate),
        "history": history,
        "candidate_entry": candidate_entry,
    }
    report_json = REPORT_DIR / f"{args.output_prefix}_train_report.json"
    report_md = REPORT_DIR / f"{args.output_prefix}_train_report.md"
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_training_report(payload, report_md)
    logger.write(
        "Fin Step7 self-play candidate",
        actual=f"checkpoint={last_checkpoint}, report={report_md}",
        details={"candidate_entry": candidate_entry},
    )
    return payload


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Train a Step7 self-play league candidate.")
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER_PATH))
    parser.add_argument("--promotion-history", default=str(DEFAULT_PROMOTION_PATH))
    parser.add_argument("--parent-id", default="champion_cbp")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--steps-per-iteration", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--clip-eps", type=float, default=0.12)
    parser.add_argument("--value-coef", type=float, default=0.55)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--bc-coef", type=float, default=0.08)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.8)
    parser.add_argument("--greedy-collect", action="store_true")
    parser.add_argument("--softmax-prob", type=float, default=0.50)
    parser.add_argument("--snapshot-prob", type=float, default=0.20)
    parser.add_argument("--elo-softmax-temperature", type=float, default=100.0)
    parser.add_argument("--advantage-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--chancellor-head", default="step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--chancellor-margin", type=float, default=0.10)
    parser.add_argument("--retarget-margin", type=float, default=0.10)
    parser.add_argument("--veto-score", type=float, default=0.05)
    parser.add_argument("--force-score", type=float, default=0.32)
    parser.add_argument("--self-force-score", type=float, default=0.55)
    parser.add_argument("--min-princess-prob", type=float, default=0.24)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--run-log", default="step7_self_play_league/logs/train_self_play_candidate.md")
    args = parser.parse_args()

    if args.candidate_id is None:
        args.candidate_id = f"sp_candidate_{now_stamp().replace(' ', '_').replace(':', '')}"
    if args.output_prefix is None:
        args.output_prefix = args.candidate_id
    train(args)


if __name__ == "__main__":
    main()
