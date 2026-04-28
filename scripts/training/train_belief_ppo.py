"""
Custom PPO fine-tuning for the belief-conditioned Love Letter actor.

This avoids the old Tianshou dependency/API mismatch and keeps the experiment
simple: player_0 is trained against configurable mixes of heuristic and random
opponents, then evaluated on 0H/1H/2H/3H.
"""

from pathlib import Path
import argparse
from datetime import datetime
import json
import sys
from dataclasses import dataclass
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from love_letter.belief_actor import (
    BeliefConditionedActor,
    BeliefConditionedEncoder,
    BeliefConditionedPolicy,
    BeliefHead,
    LATENT,
    NUM_CARDS,
)
from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from love_letter.paths import checkpoint_path
from scripts.evaluation.evaluate_models import evaluate_config

PARIS_TZ = ZoneInfo("Europe/Paris")


def now_stamp():
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


class ExperimentLogger:
    def __init__(self, path):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

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
                if isinstance(details, (dict, list)):
                    f.write("```json\n")
                    f.write(json.dumps(details, indent=2, ensure_ascii=False))
                    f.write("\n```\n\n")
                else:
                    f.write(f"{details}\n\n")

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
    hidden_cards: torch.Tensor


def cards_from_obs(obs_row):
    hand = []
    for card, value in enumerate(obs_row[:10].detach().cpu().numpy()):
        count = int(round(float(value) * 3.0))
        hand.extend([card] * max(0, count))
    return hand


def kept_card_after_play(hand, played_card):
    remaining = list(hand)
    if played_card in remaining:
        remaining.remove(played_card)
    return remaining[0] if remaining else None


def tactical_score(action, hand, belief_probs):
    if action >= 900:
        return -1e9

    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    target_dim = target_idx - 1 if 1 <= target_idx <= 3 else None
    probs = belief_probs[target_dim] if target_dim is not None else None
    kept = kept_card_after_play(hand, card)

    if card == 1 and probs is not None and guess != 1:
        guess_probs = probs.clone()
        guess_probs[1] = -1.0
        best_guess = int(torch.argmax(guess_probs).item())
        score = 4.0 * float(probs[guess])
        if guess == best_guess:
            score += 1.0
        return score

    if card == 3 and probs is not None and kept is not None:
        p_lower = float(probs[:kept].sum().item()) if kept > 0 else 0.0
        p_equal = float(probs[kept].item())
        p_higher = float(probs[kept + 1 :].sum().item()) if kept < 9 else 0.0
        return 3.0 * p_lower - 5.0 * p_higher - 0.5 * p_equal

    if card == 5:
        if target_idx == 0:
            if kept == 9:
                return -5.0
            if kept is not None and kept <= 2:
                return 1.0
            if kept is not None and kept <= 4:
                return 0.4
            return -0.2
        if probs is not None:
            return 5.0 * float(probs[9].item()) + 1.5 * float(probs[8].item())

    if card == 7 and probs is not None and kept is not None:
        expected = float(torch.sum(probs * torch.arange(10, dtype=probs.dtype, device=probs.device)).item())
        return 2.0 * ((expected - kept) / 9.0)

    if card == 6 and kept is not None:
        return 0.7 if kept <= 3 else 0.1

    return -1e9


def tactical_targets(obs_batch, mask_batch, belief_probs, min_score):
    targets = []
    belief_cpu = belief_probs.detach().cpu()
    for obs_row, mask_row, probs in zip(obs_batch, mask_batch, belief_cpu):
        hand = cards_from_obs(obs_row)
        valid_actions = torch.where(mask_row.detach().cpu())[0].tolist()
        best_action = -1
        best_score = float(min_score)
        for action in valid_actions:
            score = tactical_score(int(action), hand, probs)
            if score > best_score:
                best_score = score
                best_action = int(action)
        targets.append(best_action)
    return torch.as_tensor(targets, dtype=torch.long, device=obs_batch.device)


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


def random_action(obs_dict):
    valid = np.where(obs_dict["action_mask"] == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def opponent_action(env, agent, obs_dict, n_heuristics, bot):
    seat_idx = int(agent.rsplit("_", 1)[1])
    if 1 <= seat_idx <= n_heuristics:
        return bot.choose_action(env, agent)
    return random_action(obs_dict)


def sample_n_heuristics(n_heuristics):
    if isinstance(n_heuristics, (list, tuple)):
        return int(np.random.choice(n_heuristics))
    return int(n_heuristics)


def collect_batch(encoder, belief_head, actor, critic, n_heuristics, target_steps, device, seed_offset):
    bot = HeuristicBot()
    records = []
    game_rewards = []
    game_heuristic_counts = []
    game_idx = 0

    encoder.eval()
    belief_head.eval()
    actor.eval()
    critic.eval()

    while len(records) < target_steps:
        env = LoveLetterRLEnv(num_players=4)
        env.reset(seed=seed_offset + game_idx)
        game_n_heuristics = sample_n_heuristics(n_heuristics)
        game_heuristic_counts.append(game_n_heuristics)
        game_idx += 1
        hidden_state = torch.zeros(1, LATENT, device=device)
        game_records = []
        total_reward = 0.0

        for agent in env.agent_iter():
            obs_dict, reward, term, trunc, info = env.last()
            if agent == "player_0":
                total_reward += float(reward)

            if term or trunc:
                env.step(None)
                continue

            if agent == "player_0":
                obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                hidden_in = hidden_state.detach()

                with torch.no_grad():
                    hidden_out = encoder.forward_hidden(obs_t, hidden_in)
                    _belief_logits, belief_probs = belief_head(hidden_out)
                    logits = actor(hidden_out, belief_probs, mask_t)
                    dist = Categorical(logits=logits)
                    action_t = dist.sample()
                    logprob_t = dist.log_prob(action_t)
                    value_t = critic(hidden_out)

                action = int(action_t.item())
                hidden_state = hidden_out.detach()
                hidden_cards = info.get("hidden_cards", np.full(3, -1, dtype=np.int64))
                game_records.append(
                    {
                        "obs": obs_dict["observation"].copy(),
                        "mask": obs_dict["action_mask"].copy(),
                        "hidden_in": hidden_in.squeeze(0).cpu().numpy(),
                        "action": action,
                        "logprob": float(logprob_t.item()),
                        "value": float(value_t.item()),
                        "hidden_cards": hidden_cards.copy(),
                    }
                )
            else:
                action = opponent_action(env, agent, obs_dict, game_n_heuristics, bot)

            env.step(action)

        for rec in game_records:
            rec["return"] = total_reward
        records.extend(game_records)
        game_rewards.append(total_reward)

    records = records[:target_steps]
    return Batch(
        obs=torch.as_tensor(np.array([r["obs"] for r in records]), dtype=torch.float32, device=device),
        masks=torch.as_tensor(np.array([r["mask"] for r in records]), dtype=torch.bool, device=device),
        hidden_in=torch.as_tensor(np.array([r["hidden_in"] for r in records]), dtype=torch.float32, device=device),
        actions=torch.as_tensor([r["action"] for r in records], dtype=torch.long, device=device),
        old_logprobs=torch.as_tensor([r["logprob"] for r in records], dtype=torch.float32, device=device),
        returns=torch.as_tensor([r["return"] for r in records], dtype=torch.float32, device=device),
        values=torch.as_tensor([r["value"] for r in records], dtype=torch.float32, device=device),
        hidden_cards=torch.as_tensor(np.array([r["hidden_cards"] for r in records]), dtype=torch.long, device=device),
    ), float(np.mean(game_rewards)), len(game_rewards), {
        "mean_n_heuristics": float(np.mean(game_heuristic_counts)) if game_heuristic_counts else 0.0,
        "counts": {str(k): int(v) for k, v in zip(*np.unique(game_heuristic_counts, return_counts=True))}
        if game_heuristic_counts
        else {},
    }


def ppo_update(encoder, belief_head, actor, critic, optimizer, batch, args):
    encoder.train()
    belief_head.train()
    actor.train()
    critic.train()

    advantages = batch.returns - batch.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = torch.arange(len(batch.actions), device=batch.actions.device)
    metrics = []
    ce_belief = nn.CrossEntropyLoss(ignore_index=-1)
    ce_tactical = nn.CrossEntropyLoss(ignore_index=-1)

    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(len(indices), device=indices.device)]
        for start in range(0, len(perm), args.minibatch_size):
            idx = perm[start : start + args.minibatch_size]
            hidden = encoder.forward_hidden(batch.obs[idx], batch.hidden_in[idx])
            belief_logits, belief_probs = belief_head(hidden)
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
            belief_loss = ce_belief(
                belief_logits.reshape(-1, NUM_CARDS),
                batch.hidden_cards[idx].reshape(-1),
            )
            tactical_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
            if args.tactical_coef > 0:
                targets = tactical_targets(
                    batch.obs[idx],
                    batch.masks[idx],
                    belief_probs,
                    args.tactical_min_score,
                )
                if (targets >= 0).any():
                    tactical_loss = ce_tactical(logits, targets)
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + args.belief_coef * belief_loss
                + args.tactical_coef * tactical_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(belief_head.parameters())
                + list(actor.parameters())
                + list(critic.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()
            metrics.append(
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "belief_loss": float(belief_loss.item()),
                    "tactical_loss": float(tactical_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )

    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}


def make_policy(encoder, belief_head, actor):
    return BeliefConditionedPolicy(encoder.cpu(), belief_head.cpu(), actor.cpu()).eval()


def evaluate_policy(policy, n_games):
    configs = {
        "vs_0H_3R": {"player_1": "random", "player_2": "random", "player_3": "random"},
        "vs_1H_2R": {"player_1": "heuristic", "player_2": "random", "player_3": "random"},
        "vs_2H_1R": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "random"},
        "vs_3H": {"player_1": "heuristic", "player_2": "heuristic", "player_3": "heuristic"},
    }
    return {
        name: {k: float(v) for k, v in evaluate_config(policy, opponents, n_games=n_games).items()}
        for name, opponents in configs.items()
    }


def composite_score(arena):
    weights = {
        "vs_0H_3R": 0.10,
        "vs_1H_2R": 0.20,
        "vs_2H_1R": 0.30,
        "vs_3H": 0.40,
    }
    return float(sum(weights[name] * arena[name]["winrate"] for name in weights if name in arena))


def arena_summary(arena):
    return {
        name: {
            "winrate": round(values["winrate"], 4),
            "mean_reward": round(values["mean_reward"], 4),
        }
        for name, values in arena.items()
    }


def save_checkpoint(path, encoder, belief_head, actor, critic, metadata):
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


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encoder, belief_head, actor, critic = load_model(checkpoint_path(args.start))
    encoder.to(device)
    belief_head.to(device)
    actor.to(device)
    critic.to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(belief_head.parameters())
        + list(actor.parameters())
        + list(critic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    phase_specs = [(1, args.phase1_updates), (2, args.phase2_updates), (3, args.phase3_updates)]
    history = []
    eval_history = []
    output_dir = checkpoint_path(args.output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_cursor = args.seed * 1000
    logger = ExperimentLogger(args.run_log)

    logger.write(
        "Démarrage du pipeline PPO belief",
        expected=(
            "Partir du warm-start belief-conditioned, améliorer le score composite "
            "pondéré vers les configs avec heuristiques, et sauvegarder le meilleur checkpoint."
        ),
        actual=f"Start={checkpoint_path(args.start)}, output_prefix={args.output_prefix}",
        details=vars(args),
    )

    best_score = -1.0
    best_path = checkpoint_path(f"{args.output_prefix}_best.pth")

    def evaluate_and_maybe_save(label):
        nonlocal best_score
        policy = make_policy(encoder, belief_head, actor)
        arena = evaluate_policy(policy, args.eval_games)
        score = composite_score(arena)
        record = {
            "label": label,
            "score": score,
            "arena": arena,
            "time": now_stamp(),
        }
        eval_history.append(record)
        improved = score > best_score
        if improved:
            best_score = score
            save_checkpoint(
                best_path,
                encoder,
                belief_head,
                actor,
                critic,
                {
                    "label": label,
                    "best_score": best_score,
                    "history": history,
                    "eval_history": eval_history,
                },
            )
        logger.write(
            f"Evaluation {label}",
            expected="Le score composite doit monter; priorité aux configs 2H/3H.",
            actual=f"score={score:.4f}, best={best_score:.4f}, improved={improved}",
            details=arena_summary(arena),
        )
        encoder.to(device)
        belief_head.to(device)
        actor.to(device)
        critic.to(device)
        return arena, score

    if args.eval_initial:
        evaluate_and_maybe_save("initial")

    if args.mixed_updates > 0:
        phase_specs.append(([0, 1, 2, 3], args.mixed_updates))

    for phase_idx, (n_heuristics, n_updates) in enumerate(phase_specs, start=1):
        phase_name = (
            "mixed 0H/1H/2H/3H"
            if isinstance(n_heuristics, list)
            else f"vs {n_heuristics}H + {3 - n_heuristics}R"
        )
        print(f"\n=== Phase {phase_idx}: {phase_name} ===", flush=True)
        logger.write(
            f"Phase {phase_idx} - {phase_name}",
            expected=(
                "Collecter des trajectoires contre cette distribution d'adversaires, "
                "faire PPO sans perdre les compétences précédentes."
            ),
        )
        for update in range(1, n_updates + 1):
            batch, collect_reward, n_games, collect_meta = collect_batch(
                encoder,
                belief_head,
                actor,
                critic,
                n_heuristics=n_heuristics,
                target_steps=args.steps_per_update,
                device=device,
                seed_offset=seed_cursor,
            )
            seed_cursor += n_games + 10
            metrics = ppo_update(encoder, belief_head, actor, critic, optimizer, batch, args)
            row = {
                "phase": phase_idx,
                "n_heuristics": n_heuristics,
                "update": update,
                "collect_reward": collect_reward,
                "n_games": n_games,
                "collect_meta": collect_meta,
                **metrics,
            }
            history.append(row)
            print(
                f"phase={phase_idx} update={update:03d}/{n_updates} "
                f"reward={collect_reward:.3f} entropy={metrics['entropy']:.3f} "
                f"ploss={metrics['policy_loss']:.3f} vloss={metrics['value_loss']:.3f}",
                flush=True,
            )
            if args.eval_every_updates > 0 and update % args.eval_every_updates == 0:
                evaluate_and_maybe_save(f"phase{phase_idx}_update{update}")

        phase_path = checkpoint_path(f"{args.output_prefix}_phase{phase_idx}.pth")
        save_checkpoint(
            phase_path,
            encoder,
            belief_head,
            actor,
            critic,
            {"history": history, "phase": phase_idx, "n_heuristics": n_heuristics},
        )
        encoder.to(device)
        belief_head.to(device)
        actor.to(device)
        critic.to(device)
        print(f"Saved {phase_path}", flush=True)

    final_path = checkpoint_path(f"{args.output_prefix}_final.pth")
    save_checkpoint(
        final_path,
        encoder,
        belief_head,
        actor,
        critic,
        {"history": history, "eval_history": eval_history, "phase": "final"},
    )

    policy = make_policy(encoder, belief_head, actor)
    arena = evaluate_policy(policy, args.eval_games)
    final_score = composite_score(arena)
    report = {
        "start": str(checkpoint_path(args.start)),
        "final": str(final_path),
        "best": str(best_path) if best_path.exists() else None,
        "best_score": best_score,
        "final_score": final_score,
        "args": vars(args),
        "history": history,
        "eval_history": eval_history,
        "arena": arena,
    }
    report_path = Path("logs/evaluations") / f"{args.output_prefix}_ppo_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.write(
        "Fin du run PPO belief",
        expected="Le meilleur checkpoint doit dépasser le warm-start et idéalement approcher curriculum_phase1.",
        actual=f"final_score={final_score:.4f}, best_score={best_score:.4f}, best_path={best_path}",
        details=arena_summary(arena),
    )
    print(f"\nFinal checkpoint -> {final_path}")
    print(f"Best checkpoint -> {best_path if best_path.exists() else 'none'}")
    print(f"Eval report -> {report_path}")
    print(json.dumps(arena, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="belief_conditioned_bc.pth")
    parser.add_argument("--output-prefix", default="belief_conditioned_ppo")
    parser.add_argument("--phase1-updates", type=int, default=20)
    parser.add_argument("--phase2-updates", type=int, default=20)
    parser.add_argument("--phase3-updates", type=int, default=20)
    parser.add_argument("--mixed-updates", type=int, default=0)
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--belief-coef", type=float, default=0.05)
    parser.add_argument("--tactical-coef", type=float, default=0.0)
    parser.add_argument("--tactical-min-score", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-every-updates", type=int, default=0)
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument("--run-log", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
