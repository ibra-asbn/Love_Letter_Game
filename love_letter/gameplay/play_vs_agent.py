"""
Joue contre ton agent en console — v4 avec tracking complet et propre.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn

from love_letter.engine import LoveLetterRLEnv
from love_letter.belief_policy import load_belief_policy
from love_letter.paths import checkpoint_path

OBS_DIM = 158
ACTION_DIM = 1000
HIDDEN = 256
LATENT = 128

CHECKPOINT = checkpoint_path("curriculum_phase1.pth")

CARD_NAMES = {
    0: "Espionne (0)", 1: "Garde (1)", 2: "Prêtre (2)", 3: "Baron (3)",
    4: "Servante (4)", 5: "Prince (5)", 6: "Chancelier (6)", 7: "Roi (7)",
    8: "Comtesse (8)", 9: "Princesse (9)"
}


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, LATENT), nn.ReLU(),
        )
        self.rnn = nn.GRUCell(LATENT, LATENT)


class MaskedActor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(LATENT, ACTION_DIM)


def load_actor(path):
    policy = load_belief_policy(path)
    return policy, None


def agent_action(encoder, actor, obs_dict, h_state, agent_id="player_0"):
    if hasattr(encoder, "act"):
        return encoder.act(obs_dict, h_state, agent_id=agent_id)

    with torch.no_grad():
        x = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool).unsqueeze(0)
        h_in = h_state if h_state is not None else torch.zeros(1, LATENT)
        feat = encoder.feature_extractor(x)
        new_state = encoder.rnn(feat, h_in)
        logits = actor.head(new_state).masked_fill(~mask, -1e9)
        action = int(logits.argmax(dim=-1).item())
    return action, new_state


def decode_action(action):
    if action >= 900:
        return f"choisit sa carte (Chancelier)"
    card = action // 100
    target = (action % 100) // 10
    guess = action % 10
    s = f"joue {CARD_NAMES.get(card, '?')}"
    if card in [1, 2, 3, 5, 7]:
        if target == 9:
            s += " (sans cible)"
        else:
            s += f" sur player_{target}"
    if card == 1:
        s += f", devine {CARD_NAMES.get(guess, '?')}"
    return s


def display_game_state(env, my_agent="player_0"):
    print("\n" + "─" * 60)
    print(f"  📦 PIOCHE : {len(env._deck)} cartes restantes")
    if env._hands.get(my_agent):
        print(f"  ✋ TA MAIN : {[CARD_NAMES[c] for c in env._hands[my_agent]]}")
    else:
        print(f"  ✋ TA MAIN : (vide / éliminé)")
    print()
    for a in env.possible_agents:
        if a == my_agent:
            continue
        alive = "✓" if (a in env.agents and not env.terminations.get(a, True)) else "💀"
        protected = " [PROTÉGÉ]" if env._protected.get(a, False) else ""
        played = [CARD_NAMES[c] for c in env._played_cards.get(a, [])]
        print(f"  {alive} {a}{protected}  cartes jouées : {played}")
    print("─" * 60)


def get_human_action(env, my_agent):
    obs_dict = env.observe(my_agent)
    mask = obs_dict["action_mask"]
    valid = np.where(mask == 1)[0]

    if env._chancellor_pending and env.agent_selection == my_agent and env._chancellor_pool:
        pool = env._chancellor_pool
        print(f"\n  🔮 CHANCELIER : 3 cartes dans le pool : {[CARD_NAMES[c] for c in pool]}")
        valid_chancellor = [a for a in valid if a >= 900]
        for i, a in enumerate(valid_chancellor):
            offset = a - 900
            if len(pool) >= 3:
                keep_idx = offset // 2
                order = "normal" if offset % 2 == 0 else "inversé"
                desc = f"garder {CARD_NAMES[pool[keep_idx]]} (ordre des autres : {order})"
            else:
                desc = f"garder {CARD_NAMES[pool[offset]]}"
            print(f"    [{i}] {desc}")
        while True:
            try:
                idx = int(input(f"  Choix [0-{len(valid_chancellor)-1}] : "))
                if 0 <= idx < len(valid_chancellor):
                    return int(valid_chancellor[idx])
            except (ValueError, KeyboardInterrupt):
                pass
            print("  Invalide.")

    hand = env._hands[my_agent]
    print(f"\n  🎯 TON TOUR. Main : {[CARD_NAMES[c] for c in hand]}")
    print(f"     Pioche restante : {len(env._deck)} cartes")
    print("  ACTIONS POSSIBLES :")

    options = {}
    counter = 0
    for action in valid:
        if action >= 900:
            continue
        desc = decode_action(action)
        options[counter] = action
        print(f"    [{counter:2d}] {desc}")
        counter += 1

    while True:
        try:
            choice = input(f"\n  Choisis [0-{len(options)-1}] : ").strip()
            idx = int(choice)
            if idx in options:
                return int(options[idx])
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalide.")


def snapshot_state(env):
    return {
        "alive": {a: (a in env.agents and not env.terminations.get(a, True))
                  for a in env.possible_agents},
        "hands": {a: list(env._hands.get(a, [])) for a in env.possible_agents},
        "deck_size": len(env._deck),
    }


def announce_consequences(env, agent_who_played, action, my_agent, prev):
    """Annonce détaillée des conséquences de l'action."""
    if action is None or action >= 900:
        return

    card = action // 100
    target_idx = (action % 100) // 10
    guess = action % 10
    target = f"player_{target_idx}" if target_idx < env.num_players else None

    msgs = []
    effect_eliminated = set()

    if card == 1 and target:
        target_was_alive = prev["alive"].get(target, False)
        old_hand = prev["hands"].get(target, [])
        old_card = old_hand[0] if old_hand else None
        if target_was_alive and old_card == guess:
            effect_eliminated.add(target)
            msgs.append(f"      ✅ DEVINETTE RÉUSSIE : {target} avait {CARD_NAMES[old_card]} → éliminé")
        elif target_was_alive:
            msgs.append(f"      ❌ Devinette ratée : {target} n'avait pas {CARD_NAMES[guess]}")

    if card == 2 and target and agent_who_played == my_agent:
        if env._hands.get(target):
            seen = env._hands[target][0]
            msgs.append(f"      🔍 Tu vois la main de {target} : {CARD_NAMES[seen]}")

    if card == 3 and target:
        # On récupère les cartes AVANT le combat pour l'affichage
        target_card = prev["hands"].get(target, [None])[0]
        agent_hand = prev["hands"].get(agent_who_played, [])
        # La carte comparée est celle qui n'est pas le Baron (3)
        non_baron_cards = [c for c in agent_hand if c != 3]
        agent_card = non_baron_cards[0] if non_baron_cards else (agent_hand[0] if agent_hand else None)

        if (
            agent_card is not None
            and target_card is not None
            and prev["alive"].get(agent_who_played, False)
            and prev["alive"].get(target, False)
            and agent_card > target_card
        ):
            effect_eliminated.add(target)
            msgs.append(f"      ⚔️ BARON gagné par {agent_who_played} : {CARD_NAMES[agent_card]} vs {CARD_NAMES[target_card]} → {target} éliminé")
        elif (
            agent_card is not None
            and target_card is not None
            and prev["alive"].get(agent_who_played, False)
            and prev["alive"].get(target, False)
            and agent_card < target_card
        ):
            effect_eliminated.add(agent_who_played)
            msgs.append(f"      ⚔️ BARON perdu par {agent_who_played} : {CARD_NAMES[agent_card]} vs {CARD_NAMES[target_card]} → {agent_who_played} éliminé")
        elif (
            agent_card is not None
            and target_card is not None
            and prev["alive"].get(agent_who_played, False)
            and prev["alive"].get(target, False)
        ):
            msgs.append(f"      ⚔️ BARON égalité : les deux avaient {CARD_NAMES[agent_card]}")

    if card == 4:
        msgs.append(f"      🛡️ {agent_who_played} est protégé jusqu'à son prochain tour")

    if card == 5 and target:
        target_was_alive = prev["alive"].get(target, False)
        old_card = prev["hands"].get(target, [None])[0]

        if target == my_agent:
            msgs.append(f"      ⚠️ Prince sur toi : tu as défaussé une carte et repioché")
        else:
            if old_card is not None:
                msgs.append(f"      🃏 {target} a défaussé {CARD_NAMES[old_card]}")
            if target_was_alive and old_card == 9:
                effect_eliminated.add(target)
                msgs.append(f"      💀 {target} a défaussé sa Princesse → éliminé !")

    if card == 7 and target:
        if target == my_agent:
            new_card = env._hands.get(my_agent, [None])[0]
            msgs.append(f"      👑 ROI : tu as échangé ta main avec {agent_who_played}")
            msgs.append(f"         Tu as maintenant : {CARD_NAMES[new_card]}")
            msgs.append(f"         (et {agent_who_played} sait que tu l'as)")
        elif agent_who_played == my_agent:
            new_card = env._hands.get(my_agent, [None])[0]
            msgs.append(f"      👑 ROI : tu as échangé avec {target}")
            msgs.append(f"         Tu as maintenant : {CARD_NAMES[new_card]}")
            msgs.append(f"         ({target} sait que tu as ce que tu lui as donné)")
        else:
            msgs.append(f"      👑 {agent_who_played} et {target} ont échangé leurs cartes")

    if card == 9:
        effect_eliminated.add(agent_who_played)
        msgs.append(f"      💀 {agent_who_played} a joué la Princesse → éliminé !")

    for a in env.possible_agents:
        if a in effect_eliminated and prev["alive"].get(a, False):
            already_mentioned = any(a in m and ("éliminé" in m or "💀" in m) for m in msgs)
            if not already_mentioned:
                msgs.append(f"      💀 {a} a été éliminé")

    for m in msgs:
        print(m)


def update_kept_card_after_action(last_kept, agent_who_played, action, prev):
    """
    Met à jour le tracker `last_kept` après une action.

    Cas :
    - L'agent joue une carte : la carte qu'il GARDE est l'autre. On met à jour son entrée.
    - Roi : agent et target échangent → leurs kept_card sont SWAPPÉS.
    - Prince sur target : la main du target est défaussée puis repioché → 1 nouvelle carte unique.
      Donc target n'a pas de "kept" au sens classique. On met None.
    - Chancelier : la main de l'agent est entièrement renouvelée → on saura à l'étape "resolution".
    """
    if action is None or action >= 900:
        # Phase Chancelier resolution : main entièrement nouvelle, pas de kept
        last_kept[agent_who_played] = None
        return

    card = action // 100
    target_idx = (action % 100) // 10
    target = f"player_{target_idx}" if target_idx < 4 else None

    # 1. La carte gardée par l'agent qui joue = ce qui restait dans sa main avant
    hand_before = prev["hands"].get(agent_who_played, [])
    others = [c for c in hand_before if c != card]
    last_kept[agent_who_played] = others[0] if others else None

    # 2. Cas spéciaux
    if card == 5 and target and target != agent_who_played:
        # Prince sur target : sa main est repiochée, plus de "ancienne carte"
        last_kept[target] = None

    if card == 7 and target:
        # Roi : swap. La carte gardée par l'agent était `last_kept[agent_who_played]` (mise à jour ci-dessus)
        # Mais ils ont SWAPPÉ leur main. Donc le target a maintenant la carte que l'agent venait
        # de "garder", et l'agent a maintenant la carte du target.
        # En gros : last_kept[agent] et last_kept[target] sont SWAPPÉS, mais on les calcule depuis
        # le nouvel état (env.hands actuel).
        # Plus simple : après Roi, les deux mains sont nouvelles pour leur propriétaire,
        # donc on peut considérer qu'aucun n'a de "kept" classique au prochain tour.
        last_kept[agent_who_played] = None
        last_kept[target] = None

    if card == 6:
        # Chancelier : la main de l'agent va être complètement renouvelée à la résolution
        last_kept[agent_who_played] = None


def play_one_game():
    encoder, actor = load_actor(CHECKPOINT)
    env = LoveLetterRLEnv(num_players=4)
    env.reset()
    states = {f"player_{i}": None for i in range(1, 4)}

    # Tracker "carte gardée" pour chaque agent
    last_kept = {f"player_{i}": None for i in range(4)}

    print("\n" + "=" * 60)
    print("  🃏 NOUVELLE PARTIE — Tu es player_0")
    print("=" * 60)
    print(f"  Premier joueur : {env.agent_selection}")

    rewards = {f"player_{i}": 0.0 for i in range(4)}

    for agent in env.agent_iter():
        obs_dict, r, term, trunc, info = env.last()
        rewards[agent] += r

        if term or trunc:
            env.step(None)
            continue

        prev = snapshot_state(env)

        if agent == "player_0":
            display_game_state(env, "player_0")
            action = get_human_action(env, "player_0")
            print(f"\n  🎯 TU {decode_action(action)}")
        else:
            action, states[agent] = agent_action(encoder, actor, obs_dict, states[agent], agent_id=agent)
            print(f"\n  🤖 {agent} {decode_action(action)}")

        env.step(action)
        announce_consequences(env, agent, action, "player_0", prev)

        # Annonce : a-t-il joué sa pioche ou son ancienne carte ?
        if agent != "player_0" and action < 900:
            card_played = action // 100
            kept_before_this_turn = last_kept.get(agent)
            if kept_before_this_turn is not None:
                if card_played == kept_before_this_turn:
                    print(f"      📥 {agent} a joué son ANCIENNE carte (gardée du tour précédent)")
                else:
                    print(f"      📤 {agent} a joué la carte qu'il VIENT DE PIOCHER")

        # Mettre à jour le tracker
        update_kept_card_after_action(last_kept, agent, action, prev)

        if not all(env.terminations.values()):
            input("\n  [Entrée pour continuer...]")

    print("\n" + "=" * 60)
    print("  🏁 FIN DE LA MANCHE")
    print("=" * 60)
    print(f"\n  Récompenses :")
    for a in env.possible_agents:
        emoji = "🏆" if rewards[a] >= 1.0 else ("💀" if rewards[a] < 0 else "  ")
        print(f"    {emoji} {a} : {rewards[a]:+.1f}")

    survivants = [a for a in env.possible_agents if env._hands.get(a)]
    print(f"\n  Survivants finaux :")
    for a in survivants:
        print(f"    {a} avec {CARD_NAMES[env._hands[a][0]]}")

    if hasattr(env, "_round_winners") and env._round_winners:
        reason = "dernier survivant" if env._round_win_reason == "last_alive" else "plus haute carte"
        print(f"\n  🏆 Vainqueur(s) : {', '.join(env._round_winners)} ({reason})")
    if hasattr(env, "_round_spy_winner") and env._round_spy_winner:
        print(f"  🕵️ Bonus Espionne : {env._round_spy_winner}")


def main():
    print("=" * 60)
    print("  🃏 LOVE LETTER vs IA")
    print("=" * 60)
    print(f"  Modèle : {CHECKPOINT}")
    print(f"  Tu es player_0 contre 3 copies du modèle.")

    while True:
        play_one_game()
        again = input("\n\n  Rejouer ? [o/N] : ").strip().lower()
        if again != "o":
            break

    print("\n  Merci d'avoir joué !")


if __name__ == "__main__":
    main()
