"""
Collecte un dataset de (obs, action) depuis l'heuristique jouant contre 3 random.
Sauvegarde en pickle.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pickle
from love_letter.engine import LoveLetterRLEnv
from love_letter.bots.heuristic import HeuristicBot
from love_letter.paths import data_path


def random_action(env, agent):
    obs = env.observe(agent)
    mask = obs["action_mask"]
    valid = np.where(mask == 1)[0]
    return int(np.random.choice(valid)) if len(valid) else 0


def collect_dataset(n_games=10000, teacher_in_all_seats=True):
    """
    Collecte les transitions où l'heuristique joue.

    Si teacher_in_all_seats=True, l'heuristique joue les 4 seats (plus de data,
    meilleure distribution d'états).
    Sinon, elle joue seat 0 contre 3 random (dataset plus homogène mais moins riche).
    """
    bot = HeuristicBot()
    env = LoveLetterRLEnv(num_players=4)

    obs_list = []
    mask_list = []
    action_list = []
    info_list = []  # on garde hidden_cards pour une éventuelle belief head au BC

    for game_idx in range(n_games):
        env.reset(seed=game_idx)

        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue

            # Qui joue dans ce seat ?
            is_teacher = teacher_in_all_seats or (agent == "player_0")

            if is_teacher:
                action = bot.choose_action(env, agent)
                # Enregistrer la transition
                obs_list.append(obs_dict["observation"].copy())
                mask_list.append(obs_dict["action_mask"].copy())
                action_list.append(action)
                # hidden_cards (infos) pour une belief head éventuelle
                hidden = info.get("hidden_cards", np.full(3, -1, dtype=np.int64))
                info_list.append(hidden.copy())
            else:
                action = random_action(env, agent)

            env.step(action)

        if (game_idx + 1) % 500 == 0:
            print(f"  {game_idx + 1}/{n_games} parties — {len(obs_list)} transitions collectées")

    dataset = {
        "obs": np.array(obs_list, dtype=np.float32),
        "mask": np.array(mask_list, dtype=np.int8),
        "action": np.array(action_list, dtype=np.int64),
        "hidden_cards": np.array(info_list, dtype=np.int64),
    }

    print(f"\n Dataset final : {len(obs_list)} transitions")
    print(f"  obs.shape = {dataset['obs'].shape}")
    print(f"  action.shape = {dataset['action'].shape}")
    print(f"  Actions uniques : {len(np.unique(dataset['action']))}")

    return dataset


if __name__ == "__main__":
    print("Collecte du dataset heuristique...")
    dataset = collect_dataset(n_games=10000, teacher_in_all_seats=True)

    with open(data_path("heuristic_dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)
    print(f" Sauvegardé : {data_path('heuristic_dataset.pkl')}")
