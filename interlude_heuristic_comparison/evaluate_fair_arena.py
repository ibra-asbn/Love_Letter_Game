"""Fair-target arena for the current Love Letter policies.

This keeps the familiar progressive arena but uses HeuristicBot with randomized
target tie-breaks, so opponents do not systematically focus player_0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.bots.heuristic import HeuristicBot
from love_letter.engine import LoveLetterRLEnv
from step2_rl_finetune.common import ExperimentLogger, arena_summary, composite_score, now_stamp, resolve_checkpoint
from step2_rl_finetune.evaluate_step2 import ModelSeat, OPPONENT_CONFIGS, random_action, summarize_rewards
from step3_action_value.evaluate_advantage_head_v2 import evaluate_player0_advantage, load_advantage_bundle


INTERLUDE_DIR = PROJECT_ROOT / "interlude_heuristic_comparison"
REPORT_DIR = INTERLUDE_DIR / "reports"
LOG_DIR = INTERLUDE_DIR / "logs"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def opponent_action(env, agent, obs_dict, opponents, bot):
    opponent = opponents[agent]
    if opponent == "heuristic":
        return bot.choose_action(env, agent)
    if opponent == "random":
        return random_action(obs_dict)
    raise ValueError(opponent)


def evaluate_fair_heuristic(opponents, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    rewards = []
    wins = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        reward0 = 0.0
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0" or opponents[agent] == "heuristic":
                action = bot.choose_action(env, agent)
            else:
                action = random_action(obs_dict)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
    return summarize_rewards(rewards, wins)


def evaluate_fair_model(checkpoint, opponents, games, seed_start):
    env = LoveLetterRLEnv(num_players=4)
    bot = HeuristicBot(shuffle_targets=True)
    rewards = []
    wins = []
    lengths = []

    for game in range(games):
        seed = seed_start + game
        np.random.seed(seed)
        env.reset(seed=seed)
        model = ModelSeat(checkpoint)
        reward0 = 0.0
        actions0 = 0
        for agent in env.agent_iter():
            obs_dict, reward, terminated, truncated, _info = env.last()
            if agent == "player_0":
                reward0 += float(reward)
            if terminated or truncated:
                env.step(None)
                continue
            if agent == "player_0":
                action = model.act(obs_dict, agent)
                actions0 += 1
            else:
                action = opponent_action(env, agent, obs_dict, opponents, bot)
            env.step(action)
        rewards.append(reward0)
        wins.append(int(reward0 >= 1.0))
        lengths.append(actions0)
    return summarize_rewards(rewards, wins, lengths)


def build_step3_args(args, verify_rollouts: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_actions=args.max_actions,
        override_margin=args.override_margin,
        entropy_margin_scale=0.0,
        min_deck_progress=0.0,
        min_played_cards=0,
        max_active_players=4,
        verify_rollouts=verify_rollouts,
        verify_min_win_delta=args.verify_min_win_delta,
        verify_min_score_delta=args.verify_min_score_delta,
        verify_t_threshold=args.verify_t_threshold,
        verify_player0_continuation="heuristic",
        verify_reward_score_weight=0.05,
        heuristic_shuffle_targets=True,
        device=args.device,
        example_limit=0,
    )


def evaluate_step3(checkpoint_name, opponents, games, seed_start, args, verify_rollouts):
    checkpoint, base_checkpoint, head, ckpt = load_advantage_bundle(checkpoint_name, None)
    categories = ckpt.get("categories", [])
    eval_args = build_step3_args(args, verify_rollouts)
    if eval_args.max_actions is None:
        eval_args.max_actions = int(ckpt.get("max_actions", 14))
    return evaluate_player0_advantage(base_checkpoint, head, categories, opponents, games, seed_start, eval_args)


def evaluate_policy(policy_name, args, logger):
    configs = {}
    for idx, (config_name, opponents) in enumerate(OPPONENT_CONFIGS.items()):
        seed_start = args.seed_start + idx * args.seed_stride
        logger.write(
            f"{policy_name} - {config_name}",
            expected="Arena fair: heuristiques avec tie-break de cible aleatoire.",
            actual=f"games={args.games}, seed_start={seed_start}",
        )
        if policy_name == "heuristic_fair":
            result = evaluate_fair_heuristic(opponents, args.games, seed_start)
        elif policy_name == "step2_retarget":
            result = evaluate_fair_model(resolve_checkpoint(args.step2_checkpoint), opponents, args.games, seed_start)
        elif policy_name == "step3_fast_dagger":
            result = evaluate_step3(args.step3_fast_checkpoint, opponents, args.games, seed_start, args, verify_rollouts=0)
        elif policy_name == "step3_hybrid_verify16":
            result = evaluate_step3(
                args.step3_hybrid_checkpoint,
                opponents,
                args.games,
                seed_start,
                args,
                verify_rollouts=args.verify_rollouts,
            )
        else:
            raise ValueError(policy_name)
        configs[config_name] = result
        logger.write(
            f"{policy_name} termine {config_name}",
            expected="Reporter le resultat des qu'une etape de l'arena finit.",
            actual=(
                f"winrate={result['winrate']:.4f}, "
                f"reward={result['mean_reward']:.4f}, ci95={result['winrate_ci95']:.4f}"
            ),
            details=result,
        )
    return {
        "configs": configs,
        "composite": composite_score(configs),
        "summary": arena_summary(configs),
    }


def write_markdown(payload, path: Path) -> None:
    labels = {
        "heuristic_fair": "Fair HeuristicBot",
        "step2_retarget": "Step2 retarget",
        "step3_fast_dagger": "Step3 rapide DAgger",
        "step3_hybrid_verify16": "Step3 hybride verify16",
    }
    lines = [
        "# Fair Arena - HeuristicBot vs Step2 vs Step3",
        "",
        f"Date: {payload['created_at']}.",
        "",
        "Heuristiques adverses: `HeuristicBot(shuffle_targets=True)`.",
        "",
        "| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, policy in payload["policies"].items():
        c = policy["configs"]
        lines.append(
            "| {name} | {a:.2%} | {b:.2%} | {c2:.2%} | {d:.2%} | {comp:.5f} |".format(
                name=labels[name],
                a=c["vs_0H_3R"]["winrate"],
                b=c["vs_1H_2R"]["winrate"],
                c2=c["vs_2H_1R"]["winrate"],
                d=c["vs_3H"]["winrate"],
                comp=policy["composite"],
            )
        )
    lines.extend(
        [
            "",
            "## Lecture",
            "",
            payload["conclusion"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(description="Evaluate current models in a fair-target progressive arena.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=150000)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--step2-checkpoint", default="step2_retarget_distilled_attempt1.pth")
    parser.add_argument("--step3-fast-checkpoint", default="step3_advantage_v2_dagger_attempt1_iter1.pth")
    parser.add_argument("--step3-hybrid-checkpoint", default="step3_advantage_v2_attempt2_strict.pth")
    parser.add_argument("--override-margin", type=float, default=0.10)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--verify-rollouts", type=int, default=16)
    parser.add_argument("--verify-min-win-delta", type=float, default=0.125)
    parser.add_argument("--verify-min-score-delta", type=float, default=0.05)
    parser.add_argument("--verify-t-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="fair_arena_heuristic_step2_step3_1000.json")
    parser.add_argument("--markdown", default="fair_arena_heuristic_step2_step3_1000.md")
    parser.add_argument("--run-log", default="interlude_heuristic_comparison/logs/2026-04-25_fair_arena_1000.md")
    args = parser.parse_args()

    logger = ExperimentLogger(args.run_log)
    logger.reset()
    logger.write(
        "Debut fair arena",
        expected="Comparer les modeles contre des heuristiques sans focus player_0.",
        actual=f"games={args.games}, seed_start={args.seed_start}, verify_rollouts={args.verify_rollouts}",
        details=vars(args),
    )

    policies = {}
    for policy_name in ["heuristic_fair", "step2_retarget", "step3_fast_dagger", "step3_hybrid_verify16"]:
        policies[policy_name] = evaluate_policy(policy_name, args, logger)
        logger.write(
            f"Modele termine - {policy_name}",
            expected="Envoyer un checkpoint mental a la fin de chaque modele.",
            actual=f"composite={policies[policy_name]['composite']:.5f}",
            details=policies[policy_name]["summary"],
        )

    heuristic_score = policies["heuristic_fair"]["composite"]
    conclusion = " ".join(
        f"{name}: composite={policy['composite']:.5f}, delta_vs_fair_heuristic={policy['composite'] - heuristic_score:+.5f}."
        for name, policy in policies.items()
    )
    payload = {
        "created_at": now_stamp(),
        "games": args.games,
        "seed_start": args.seed_start,
        "seed_stride": args.seed_stride,
        "heuristic_mode": "shuffle_targets=True",
        "policies": policies,
        "conclusion": conclusion,
    }

    output = REPORT_DIR / args.output
    markdown = REPORT_DIR / args.markdown
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, markdown)
    logger.write(
        "Fin fair arena",
        expected="Avoir une premiere lecture fair-mode.",
        actual=conclusion,
        details={"json": str(output), "markdown": str(markdown)},
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
