"""
Bot heuristique pour Love Letter.
Joue selon des règles stratégiques simples inspirées du jeu humain.
"""
import numpy as np


class HeuristicBot:
    """
    Bot qui joue selon des règles. Prend en entrée l'env et l'agent,
    retourne une action valide.
    """

    CARD_NAMES = {
        0: "Espionne", 1: "Garde", 2: "Prêtre", 3: "Baron", 4: "Servante",
        5: "Prince", 6: "Chancelier", 7: "Roi", 8: "Comtesse", 9: "Princesse"
    }
    # Pour le Garde : on ne peut pas deviner 1 (Garde)
    GUESSABLE_CARDS = [0, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, verbose=False, shuffle_targets=False):
        self.verbose = verbose
        self.shuffle_targets = shuffle_targets

    def choose_action(self, env, agent):
        """Retourne l'action à jouer pour cet agent dans cet env."""
        obs = env.observe(agent)
        mask = obs["action_mask"]

        if mask.sum() == 0:
            return 0

        # Cas spécial : phase Chancelier
        if env._chancellor_pending and env.agent_selection == agent:
            return self._choose_chancellor(env, agent, mask)

        hand = env._hands.get(agent, [])
        if not hand:
            return self._fallback(mask)

        # Appliquer les règles par ordre de priorité
        action = self._apply_rules(env, agent, hand, mask)
        if action is not None and mask[action] == 1:
            return int(action)

        return self._fallback(mask)

    # ==========================================
    # RÈGLES DE DÉCISION
    # ==========================================
    def _apply_rules(self, env, agent, hand, mask):
        my_idx = env.possible_agents.index(agent)
        valid_opponents = self._valid_opponents(env, agent)

        # --- Règle 0 : Comtesse forcée (déjà gérée par le mask, mais on l'applique explicitement) ---
        if 8 in hand and (5 in hand or 7 in hand):
            return 800 + 9 * 10 + 0  # Comtesse, dummy target

        # --- Règle 1 : Garde avec info certaine ---
        if 1 in hand:
            action = self._guard_with_certainty(env, agent, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 2 : Princesse jamais (sauf forcé, géré par mask) ---
        # Si on a Princesse + autre, on joue l'autre
        if 9 in hand and len(hand) == 2:
            other = [c for c in hand if c != 9][0]
            # On va laisser les règles suivantes décider sur 'other'
            hand_for_decision = [other]
        else:
            hand_for_decision = hand

        # --- Règle 3 : Prince sur adversaire qu'on sait avoir Princesse ---
        if 5 in hand_for_decision:
            action = self._prince_on_known_princess(env, agent, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 4 : Baron confiant ---
        if 3 in hand_for_decision:
            action = self._confident_baron(env, agent, hand_for_decision, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 5 : Roi seulement si fin de manche ---
        if 7 in hand_for_decision:
            action = self._late_king(env, agent, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 6 : Chancelier si main mauvaise ---
        if 6 in hand_for_decision:
            action = self._chancellor_if_bad_hand(env, agent, hand_for_decision, mask)
            if action is not None:
                return action

        # --- Règle 7 : Prêtre pour info ---
        if 2 in hand_for_decision:
            action = self._priest_for_info(env, agent, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 8 : Prince sur soi si main pourrie ---
        if 5 in hand_for_decision and len(hand) == 2:
            other_list = [c for c in hand if c != 5]
            if other_list:  # si on a deux Princes, other_list est vide
                other = other_list[0]
                if other <= 2 and other != 9:  # main pourrie mais pas Princesse
                    action = 500 + my_idx * 10 + 0
                    if mask[action] == 1:
                        return action

        # --- Règle 9 : Garde sans info (devinette probabiliste) ---
        if 1 in hand_for_decision:
            action = self._guard_probabilistic(env, agent, valid_opponents, mask)
            if action is not None:
                return action

        # --- Règle 10 : Servante défensive ---
        if 4 in hand_for_decision:
            action = 400 + 9 * 10 + 0
            if mask[action] == 1:
                return action

        # --- Règle 11 : Espionne (low priority mais bonus à la fin) ---
        if 0 in hand_for_decision:
            action = 0 + 9 * 10 + 0  # action = 0
            if mask[action] == 1:
                return action

        # --- Fallback : jouer la plus basse carte ---
        return self._play_lowest(hand_for_decision, mask, valid_opponents)

    # ==========================================
    # HELPERS DE RÈGLES
    # ==========================================
    def _valid_opponents(self, env, agent):
        """Retourne les indices des adversaires ciblables."""
        opponents = [
            i for i, a in enumerate(env.possible_agents)
            if a in env.agents
            and not env.terminations.get(a, True)
            and not env._protected.get(a, False)
            and a != agent
        ]
        if self.shuffle_targets and len(opponents) > 1:
            opponents = list(np.random.permutation(opponents))
        return opponents

    def _known_card_of(self, env, agent, opp):
        """Retourne la carte connue de opp (via Prêtre/Roi) ou None."""
        known = env._known_cards[agent][opp]
        idx = np.where(known >= 1.0)[0]
        return int(idx[0]) if len(idx) > 0 else None

    def _guard_with_certainty(self, env, agent, valid_opponents, mask):
        """Si on sait qu'un adversaire a une carte X (devinable), joue Garde dessus."""
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            known = self._known_card_of(env, agent, opp)
            if known is not None and known in self.GUESSABLE_CARDS:
                action = 100 + opp_idx * 10 + known
                if mask[action] == 1:
                    return action
        return None

    def _prince_on_known_princess(self, env, agent, valid_opponents, mask):
        """Joue Prince sur un adversaire qu'on sait avoir la Princesse."""
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            known = self._known_card_of(env, agent, opp)
            if known == 9:
                action = 500 + opp_idx * 10 + 0
                if mask[action] == 1:
                    return action
        return None

    def _confident_baron(self, env, agent, hand, valid_opponents, mask):
        other_card = [c for c in hand if c != 3]
        if not other_card:
            # Deux Barons : on joue un Baron, mais on ne sait pas contre qui évaluer
            # On skip cette règle, un fallback plus tard jouera
            return None
        my_val = other_card[0]

        # Stratégie stricte : ne joue Baron que si my_val >= 5 OU
        # si on sait que la cible a une carte < my_val
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            known = self._known_card_of(env, agent, opp)

            if known is not None:
                if known < my_val:
                    # Victoire garantie
                    action = 300 + opp_idx * 10 + 0
                    if mask[action] == 1:
                        return action
                # Si on sait qu'il a plus grand, skip
                if known > my_val:
                    continue

            # Pas d'info : on joue seulement si my_val >= 5
            if my_val >= 5:
                action = 300 + opp_idx * 10 + 0
                if mask[action] == 1:
                    return action

        return None

    def _late_king(self, env, agent, valid_opponents, mask):
        """Joue Roi seulement si peu de cartes restantes (≤ 5) ou obligé."""
        if len(env._deck) > 5:
            return None
        # Cible préférée : celle dont on ne connaît pas la main et qui a le plus joué
        best_target = None
        best_score = -1
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            known = self._known_card_of(env, agent, opp)
            if known in [9, 8]:  # Princesse/Comtesse connue : jackpot
                action = 700 + opp_idx * 10 + 0
                if mask[action] == 1:
                    return action
            score = len(env._played_cards.get(opp, []))
            if score > best_score:
                best_score = score
                best_target = opp_idx
        if best_target is not None:
            action = 700 + best_target * 10 + 0
            if mask[action] == 1:
                return action
        return None

    def _chancellor_if_bad_hand(self, env, agent, hand, mask):
        """Joue Chancelier si notre main est mauvaise."""
        other = [c for c in hand if c != 6]
        if not other:
            # Les deux cartes sont des Chanceliers, jouer l'un d'eux
            action = 600 + 9 * 10 + 0
            if mask[action] == 1:
                return action
            return None
        other_val = other[0]
        if other_val <= 3:
            action = 600 + 9 * 10 + 0
            if mask[action] == 1:
                return action
        return None

    def _priest_for_info(self, env, agent, valid_opponents, mask):
        """Joue Prêtre sur un adversaire sur lequel on n'a pas d'info."""
        # Priorité : adversaire sans info connue et qui est le plus menaçant
        best_target = None
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            if self._known_card_of(env, agent, opp) is None:
                best_target = opp_idx
                break
        if best_target is not None:
            action = 200 + best_target * 10 + 0
            if mask[action] == 1:
                return action
        # Fallback : premier opposant valide
        for opp_idx in valid_opponents:
            action = 200 + opp_idx * 10 + 0
            if mask[action] == 1:
                return action
        return None

    def _guard_probabilistic(self, env, agent, valid_opponents, mask):
        """Joue Garde en devinant la carte la plus probable dans le deck restant."""
        # Distribution des cartes restantes (utilise le block [20:30] de l'obs, ou recalcul)
        max_counts = np.array([2, 6, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.float32)
        played = np.zeros(10)
        for a in env.possible_agents:
            for c in env._played_cards.get(a, []):
                played[c] += 1
        remaining = max_counts - played
        for c in env._hands.get(agent, []):
            remaining[c] -= 1
        for opp in env.possible_agents:
            if opp == agent:
                continue
            for c in range(10):
                if env._known_cards[agent][opp][c] >= 1.0:
                    remaining[c] -= 1
        remaining = np.clip(remaining, 0, None)

        # Probabilité d'une carte C en main d'un adversaire non-connu = remaining[C] / sum(remaining)
        # On exclut Garde (1) des devinettes valides
        probs = remaining.copy()
        probs[1] = 0  # ne peut pas deviner Garde
        if probs.sum() == 0:
            return None
        probs = probs / probs.sum()
        best_guess = int(np.argmax(probs))

        # Cible : le premier adversaire valide dont on ne connaît pas la carte
        for opp_idx in valid_opponents:
            opp = env.possible_agents[opp_idx]
            if self._known_card_of(env, agent, opp) is None:
                action = 100 + opp_idx * 10 + best_guess
                if mask[action] == 1:
                    return action
        # Fallback : premier valide
        for opp_idx in valid_opponents:
            action = 100 + opp_idx * 10 + best_guess
            if mask[action] == 1:
                return action
        return None

    def _play_lowest(self, hand, mask, valid_opponents=None):
        """Fallback : jouer la carte la plus basse jouable."""
        target_order = list(valid_opponents or [])
        target_order += [t for t in range(10) if t not in target_order]
        for card in sorted(hand):
            for t in target_order:
                for g in range(10):
                    action = card * 100 + t * 10 + g
                    if mask[action] == 1:
                        return action
        return self._fallback(mask)

    def _fallback(self, mask):
        """Dernier recours : première action valide."""
        valid = np.where(mask == 1)[0]
        return int(valid[0]) if len(valid) > 0 else 0

    # ==========================================
    # CHANCELIER
    # ==========================================
    def _choose_chancellor(self, env, agent, mask):
        """Choix de la carte à garder dans le pool."""
        pool = env._chancellor_pool
        # Garde la carte la plus haute qui n'est pas Princesse (qu'on garde si pas d'autre choix)
        best_idx = 0
        best_val = -1
        for i, c in enumerate(pool):
            # On préfère garder les hautes cartes, Princesse en dernier recours
            val = c if c != 9 else -0.5  # Princesse pas top en early
            if val > best_val:
                best_val = val
                best_idx = i

        # Action : garder index best_idx, ordre 0
        if len(pool) >= 3:
            action = 900 + best_idx * 2 + 0
        else:
            action = 900 + best_idx

        if mask[action] == 1:
            return action
        return self._fallback(mask)
