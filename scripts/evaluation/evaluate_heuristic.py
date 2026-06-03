from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


"""
Compare le winrate de plusieurs agents vs 3 random.
"""
import numpy as np
from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot


def random_action(env, agent):
    """Policy random masquée."""
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid = np.where(mask == 1)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def play_one_game(env, policies, seed=None):
    """Joue une partie complète et retourne le reward cumulé par joueur."""
    env.reset(seed=seed)
    rewards = {a: 0.0 for a in env.possible_agents}

    for agent in env.agent_iter():
        _, reward, termination, truncation, _ = env.last()
        rewards[agent] += reward
        if termination or truncation:
            action = None
        else:
            policy = policies[agent]
            if policy == "random":
                action = random_action(env, agent)
            elif policy == "heuristic":
                action = HEURISTIC_BOT.choose_action(env, agent)
        env.step(action)

    return rewards


def eval_agent(agent_type, n_games=1000):
    """Évalue un agent en position 0 vs 3 random sur n_games parties."""
    env = LoveLetterRLEnv(num_players=4)
    policies = {
        "player_0": agent_type,
        "player_1": "random",
        "player_2": "random",
        "player_3": "random",
    }

    total_reward = 0.0
    wins = 0
    for i in range(n_games):
        rewards = play_one_game(env, policies, seed=i)
        r = rewards["player_0"]
        total_reward += r
        if r >= 1.0:
            wins += 1

    return {
        "mean_reward": total_reward / n_games,
        "winrate": wins / n_games,
    }


HEURISTIC_BOT = HeuristicBot()


def main():
    print("=" * 60)
    print("Évaluation : X vs 3 Random (1000 parties)")
    print("=" * 60)

    print("\n[1/2] Random vs 3 Random (baseline)...")
    result_random = eval_agent("random", n_games=1000)
    print(f"  → Mean reward : {result_random['mean_reward']:.3f}")
    print(f"  → Winrate (>=1 token) : {result_random['winrate']:.3f}")

    print("\n[2/2] Heuristique vs 3 Random...")
    result_heur = eval_agent("heuristic", n_games=1000)
    print(f"  → Mean reward : {result_heur['mean_reward']:.3f}")
    print(f"  → Winrate (>=1 token) : {result_heur['winrate']:.3f}")

    print("\n" + "=" * 60)
    print("Résumé")
    print("=" * 60)
    print(f"Random        : reward={result_random['mean_reward']:.3f}, winrate={result_random['winrate']:.3f}")
    print(f"Heuristique   : reward={result_heur['mean_reward']:.3f}, winrate={result_heur['winrate']:.3f}")
    print(f"Gain heur     : +{result_heur['mean_reward'] - result_random['mean_reward']:.3f} reward")
    print(f"\nTon PPO actuel plafonne autour de reward=0.43 selon les logs.")
    print(f"Compare avec l'heuristique pour estimer le plafond accessible.")


if __name__ == "__main__":
    main()