from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from love_letter.engine import LoveLetterRLEnv

CARD_NAMES = {
    0: "Espionne", 1: "Garde", 2: "Prêtre", 3: "Baron", 4: "Servante",
    5: "Prince", 6: "Chancelier", 7: "Roi", 8: "Comtesse", 9: "Princesse"
}


def print_state(env, title=""):
    print(f"\n===== {title} =====")
    print(f"Agent à jouer : {env.agent_selection}")
    print(f"Pioche restante : {len(env._deck)} cartes")
    print(f"Deck (top à droite) : {[CARD_NAMES[c] for c in env._deck]}")
    for a in env.possible_agents:
        alive = "✓" if a in env.agents and not env.terminations.get(a, True) else "💀"
        hand = [CARD_NAMES[c] for c in env._hands.get(a, [])]
        played = [CARD_NAMES[c] for c in env._played_cards.get(a, [])]
        print(f"  {alive} {a} | main={hand} | joué={played}")
        print(f"       tokens={env._tokens[a]} | turns_unchanged={env._turns_hand_unchanged[a]} "
              f"| min_kept={env._min_kept_card[a]} | countess_vol={env._countess_voluntary[a]}")
        # Deck knowledge
        if env._deck_knowledge[a]:
            knowl = {pos: CARD_NAMES[c] for pos, c in env._deck_knowledge[a].items()}
            print(f"       deck_knowledge={knowl}")
        # Known cards sur les autres
        for opp in env.possible_agents:
            if opp == a:
                continue
            known = env._known_cards[a][opp]
            if known.sum() > 0:
                known_cards = {CARD_NAMES[i]: float(v) for i, v in enumerate(known) if v > 0}
                print(f"       sait sur {opp} : {known_cards}")


def force_action(env, agent, card_to_play, target=None, guess=0):
    """Construit l'action et l'applique si le masque la permet."""
    obs = env.observe(agent)
    mask = obs["action_mask"]

    if target is None:
        target_idx = 9  # dummy
    else:
        target_idx = env.possible_agents.index(target)

    action = card_to_play * 100 + target_idx * 10 + guess

    if mask[action] == 0:
        # Chercher une action valide la plus proche
        valid = np.where(mask == 1)[0]
        print(f"  ⚠ Action {action} invalide pour {agent}, fallback sur {valid[0]}")
        action = int(valid[0])

    env.step(int(action))


def test_1_chancellor_tracking():
    print("\n" + "=" * 60)
    print("TEST 1 : Tracking du Chancelier")
    print("=" * 60)

    # Chercher un seed où player_0 a Chancelier en main initiale
    env = LoveLetterRLEnv(num_players=4)
    found = False
    for seed in range(200):
        env.reset(seed=seed, options={"starting_agent": "player_0"})
        if 6 in env._hands["player_0"]:
            found = True
            break
    if not found:
        print("  Pas trouvé de seed avec Chancelier pour player_0")
        return

    print(f"  Seed {seed} : player_0 a Chancelier")
    print_state(env, "Setup initial")

    p0_hand = env._hands["player_0"].copy()
    print(f"\nMain de player_0 : {[CARD_NAMES[c] for c in p0_hand]}")

    # Snapshot du deck AVANT le Chancelier
    deck_before = list(env._deck)
    print(f"Top 2 du deck avant Chancelier : {[CARD_NAMES[c] for c in deck_before[-2:]]} "
          f"(dernier = prochaine pioche)")

    # Jouer Chancelier
    print("\n→ player_0 joue Chancelier (action 690)")
    env.step(690)

    print(f"\nChancellor_pending = {env._chancellor_pending}")
    print(f"Pool disponible : {[CARD_NAMES[c] for c in env._chancellor_pool]}")
    pool_snapshot = list(env._chancellor_pool)

    # Résoudre : garder la carte 0 du pool, ordre 0 (action 900)
    print("\n→ player_0 garde la carte 0 du pool, ordre 0 (action 900)")
    kept_card = pool_snapshot[0]
    print(f"  Carte gardée : {CARD_NAMES[kept_card]}")
    # Les 2 autres cartes du pool sont remises au fond
    cards_put_back = [pool_snapshot[1], pool_snapshot[2]]
    print(f"  Cartes remises au fond (ordre 0) : {[CARD_NAMES[c] for c in cards_put_back]}")

    env.step(900)

    print_state(env, "Après résolution Chancelier")

    # Vérifier le deck_knowledge
    knowl = env._deck_knowledge["player_0"]
    print(f"\n✓ deck_knowledge player_0 : { {pos: CARD_NAMES[c] for pos, c in knowl.items()} }")
    print(f"  Taille du deck après : {len(env._deck)}")
    print(f"  Fond du deck (5 premières positions depuis le fond) : "
          f"{[CARD_NAMES[c] for c in env._deck[:5]]}")

    # Vérifier que les positions trackées correspondent au deck réel
    print(f"\n  Vérification positions :")
    for pos, card in knowl.items():
        deck_idx_from_top = len(env._deck) - 1 - pos
        if 0 <= deck_idx_from_top < len(env._deck):
            actual = env._deck[deck_idx_from_top]
            match = "✓" if actual == card else "❌"
            print(f"  {match} pos {pos} (idx {deck_idx_from_top} du deck) : "
                  f"attendu {CARD_NAMES[card]}, trouvé {CARD_NAMES[actual]}")
        else:
            print(f"  ❌ pos {pos} (idx {deck_idx_from_top}) hors deck")



def test_2_baron_min_kept():
    print("\n" + "=" * 60)
    print("TEST 2 : Baron gagné → min_kept_card")
    print("=" * 60)
    env = LoveLetterRLEnv(num_players=4)
    env.reset(seed=7, options={"starting_agent": "player_0"})

    # On cherche un seed où player_0 a Baron et peut battre quelqu'un
    for trial_seed in range(100):
        env.reset(seed=trial_seed, options={"starting_agent": "player_0"})
        p0_hand = env._hands["player_0"]
        if 3 not in p0_hand:
            continue
        # Trouver un adversaire avec une carte plus basse
        other_card = [c for c in p0_hand if c != 3][0]
        opp_cards = {a: env._hands[a][0] for a in env.possible_agents if a != "player_0"}
        target = None
        for opp, opp_card in opp_cards.items():
            if opp_card < other_card:
                target = opp
                break
        if target:
            print(f"  Seed {trial_seed} : player_0 a {[CARD_NAMES[c] for c in p0_hand]}, "
                  f"target = {target} avec {CARD_NAMES[opp_cards[target]]}")
            # Jouer Baron sur target
            action = 300 + env.possible_agents.index(target) * 10 + 0
            env.step(action)
            print_state(env, "Après Baron gagné")
            expected_bound = opp_cards[target] + 1
            print(f"\n✓ min_kept_card[player_0] devrait être {expected_bound}, "
                  f"il est {env._min_kept_card['player_0']}")
            return
    print("  Pas trouvé de seed adéquat")


def test_3_priest_known_cards():
    print("\n" + "=" * 60)
    print("TEST 3 : Prêtre → known_cards")
    print("=" * 60)
    for trial_seed in range(100):
        env = LoveLetterRLEnv(num_players=4)
        env.reset(seed=trial_seed, options={"starting_agent": "player_0"})
        if 2 in env._hands["player_0"]:
            target = "player_1"
            target_card = env._hands[target][0]
            print(f"  Seed {trial_seed} : player_0 joue Prêtre sur {target} "
                  f"(qui a {CARD_NAMES[target_card]})")
            action = 200 + 1 * 10 + 0
            env.step(action)
            known = env._known_cards["player_0"][target]
            print(f"\n✓ known_cards[player_0][{target}] = "
                  f"{ {CARD_NAMES[i]: float(v) for i, v in enumerate(known) if v > 0} }")
            return
    print("  Pas trouvé de seed adéquat")


def test_4_turns_unchanged():
    print("\n" + "=" * 60)
    print("TEST 4 : turns_hand_unchanged incrémente correctement")
    print("=" * 60)
    env = LoveLetterRLEnv(num_players=4)
    env.reset(seed=3, options={"starting_agent": "player_0"})

    print(f"État initial turns_unchanged : { {a: env._turns_hand_unchanged[a] for a in env.possible_agents} }")

    # Faire jouer quelques tours random et voir l'évolution
    for step_n in range(8):
        if all(env.terminations.values()):
            break
        agent = env.agent_selection
        if env.terminations[agent]:
            env.step(None)
            continue
        obs = env.observe(agent)
        mask = obs["action_mask"]
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            break
        action = int(np.random.choice(valid))
        env.step(action)
        print(f"  Après step {step_n} ({agent} a joué): "
              f"{ {a: env._turns_hand_unchanged[a] for a in env.possible_agents} }")


def test_5_tokens_in_obs():
    print("\n" + "=" * 60)
    print("TEST 5 : Tokens dans l'obs")
    print("=" * 60)
    env = LoveLetterRLEnv(num_players=4)
    tokens_init = {"player_0": 1, "player_1": 0, "player_2": 1, "player_3": 0}
    env.reset(seed=1, options={"tokens": tokens_init, "starting_agent": "player_0"})

    obs = env.observe("player_0")
    # Tokens sont aux indices 111-114
    print(f"  Tokens en mémoire : {env._tokens}")
    print(f"  Tokens dans obs (indices 111-114) : {obs['observation'][111:115]}")
    # player_0 est "soi" → devrait être 1/2 = 0.5
    # player_1, 2, 3 sont les adversaires dans l'ordre du seat
    print(f"  ✓ Attendu : [0.5, 0.0, 0.5, 0.0] (soi=1/2, adv1=0/2, adv2=1/2, adv3=0/2)")


def test_6_obs_shape():
    print("\n" + "=" * 60)
    print("TEST 6 : Shape de l'obs")
    print("=" * 60)
    env = LoveLetterRLEnv(num_players=4)
    env.reset(seed=0)
    obs = env.observe("player_0")
    print(f"  Shape observation : {obs['observation'].shape}")
    print(f"  Shape action_mask : {obs['action_mask'].shape}")
    print(f"  ✓ Attendu : (158,) et (1000,)")


if __name__ == "__main__":
    test_6_obs_shape()
    test_5_tokens_in_obs()
    test_3_priest_known_cards()
    test_2_baron_min_kept()
    test_4_turns_unchanged()
    test_1_chancellor_tracking()
