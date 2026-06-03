import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector


class LoveLetterRLEnv(AECEnv):
    """
    Love Letter POMDP v2 — enrichi en knowledge state.

    Nouveautés vs v1 :
    - Tracking des positions connues du deck (via Chancelier)
    - Contrainte bayésienne "Baron gagné → min_kept_card"
    - Tokens exposés dans l'obs (first-to-2)
    - Espionne tracking explicite
    - Pattern "cartes gardées" (turns_hand_unchanged)
    - Comtesse volontaire vs forcée
    - Seat aléatoire au reset
    - Propagation deck → known_cards quand un adversaire pioche une carte trackée
    """
    metadata = {'render.modes': ['human'], "name": "love_letter_rl_v7"}

    # Constantes
    MAX_DECK_POS_TRACKED = 3  # on track jusqu'à 3 positions (2 Chancelier + marge)
    FULL_DECK = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9]
    MAX_CARD_COUNTS = np.array([2, 6, 2, 2, 2, 2, 2, 1, 1, 1], dtype=np.float32)
    OBS_DIM = 158

    def __init__(self, num_players=4):
        super().__init__()
        self.num_players = num_players
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        self.action_spaces = {
            agent: gym.spaces.Discrete(1000) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Dict({
                "observation": Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32),
                "action_mask": Box(low=0, high=1, shape=(1000,), dtype=np.int8)
            }) for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ==========================================
    # RESET
    # ==========================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        options = options or {}
        initial_tokens = options.get("tokens", {a: 0 for a in self.possible_agents})
        # Seat aléatoire : on sample un agent qui commence
        starting_agent = options.get("starting_agent", None)
        if starting_agent is None:
            starting_agent = np.random.choice(self.possible_agents)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}

        self.infos = {
            a: {"hidden_cards": np.full(self.num_players - 1, -1, dtype=np.int64)}
            for a in self.possible_agents
        }

        self._deck = list(self.FULL_DECK)
        np.random.shuffle(self._deck)

        self._set_aside = self._deck.pop()
        self._hands = {a: [self._deck.pop()] for a in self.possible_agents}
        self._played_cards = {a: [] for a in self.possible_agents}
        self._protected = {a: False for a in self.possible_agents}

        self._known_cards = {
            a: {opp: np.zeros(10, dtype=np.float32) for opp in self.possible_agents}
            for a in self.possible_agents
        }

        # ===== NOUVEAUX STATES =====

        # Deck knowledge : par joueur, dict {position_depuis_top: carte}
        # Position 0 = prochaine à piocher, 1 = suivante, etc.
        self._deck_knowledge = {a: {} for a in self.possible_agents}

        # Tokens (persistants entre manches via options)
        self._tokens = dict(initial_tokens)

        # Tours consécutifs où l'adversaire n'a pas changé de main
        # Un joueur "change de main" quand : il pioche (tour normal), Chancelier,
        # Roi swap, Prince sur lui.
        self._turns_hand_unchanged = {a: 0 for a in self.possible_agents}

        # Borne inférieure sur la carte gardée via Baron
        self._min_kept_card = {a: 0 for a in self.possible_agents}

        # Comtesse jouée volontairement cette manche (par joueur)
        self._countess_voluntary = {a: False for a in self.possible_agents}

        # ===== État chancelier =====
        self._chancellor_pending = False
        self._chancellor_pool = []

        # ===== Sélection du premier joueur =====
        # On construit un agent_selector qui démarre à starting_agent
        idx = self.possible_agents.index(starting_agent)
        ordered = self.possible_agents[idx:] + self.possible_agents[:idx]
        self._agent_selector = AgentSelector(ordered)
        self.agent_selection = self._agent_selector.reset()

        self._draw_card(self.agent_selection, is_initial=True)

    # ==========================================
    # HELPERS : DECK KNOWLEDGE
    # ==========================================
    def _shift_deck_knowledge_on_draw(self):
        """
        Appelé quand une carte est piochée du top (deck.pop()).
        Toutes les positions connues remontent de 1 (position 0 disparaît).
        """
        for agent in self.possible_agents:
            new_knowledge = {}
            for pos, card in self._deck_knowledge[agent].items():
                if pos == 0:
                    continue  # carte piochée, on perd l'info
                new_knowledge[pos - 1] = card
            self._deck_knowledge[agent] = new_knowledge

    def _propagate_top_card_to_drawer(self, drawer, card):
        """
        Si un joueur pioche une carte, et qu'un autre joueur savait que c'était
        la carte du top, alors ce joueur sait maintenant que drawer a cette carte.
        """
        for observer in self.possible_agents:
            if observer == drawer:
                continue
            # L'observer savait-il que la carte du top était X ?
            if self._deck_knowledge[observer].get(0) == card:
                self._known_cards[observer][drawer][card] = 1.0

    def _draw_card(self, agent, is_initial=False):
        """
        Pioche une carte pour l'agent.
        Met à jour deck_knowledge et propage l'info aux observers.
        """
        if not self._deck:
            return
        card = self._deck[-1]

        # Propagation avant le pop : les observers qui connaissaient la position 0
        self._propagate_top_card_to_drawer(agent, card)

        self._deck.pop()
        self._hands[agent].append(card)

        # Shift des positions pour tous les joueurs
        self._shift_deck_knowledge_on_draw()

        # L'agent qui vient de piocher : si sa pos 0 était connue, il sait
        # maintenant quelle carte il a en main (c'est automatique puisqu'il
        # voit sa main). Son deck_knowledge a déjà été shifté.

        # Reset du "main inchangée" — sauf si c'est le tir initial de reset
        if not is_initial:
            self._turns_hand_unchanged[agent] = 0
            # Ce joueur a pioché : sa main a changé, la borne min_kept est invalidée
            # SEULEMENT si la borne était sur une carte qu'il n'a plus.
            # En pratique, on l'invalide quand il JOUE la carte protégée par la borne.
            # Cf. _apply_effect : on reset min_kept_card au moment de jouer la carte.

    def _invalidate_deck_knowledge_on_chancellor_by_other(self, chancellor_player):
        """
        Un autre joueur joue Chancelier. Il a pris 2 cartes du top et va en
        remettre 2 au fond. Pour chaque observer (autre que chancellor_player) :
        - Si leurs positions trackées incluent 0 ou 1 → invalider tout
        - Sinon, les positions > 1 remontent de 2 (car 2 cartes ont été retirées
          du top), et les nouvelles positions (fond du deck) restent inconnues.
        """
        for observer in self.possible_agents:
            if observer == chancellor_player:
                continue
            knowledge = self._deck_knowledge[observer]
            if any(pos <= 1 for pos in knowledge):
                # Une de nos cartes trackées était dans le top 2 : perdue
                self._deck_knowledge[observer] = {}
            else:
                # Remonter de 2
                new_knowledge = {}
                for pos, card in knowledge.items():
                    new_knowledge[pos - 2] = card
                self._deck_knowledge[observer] = new_knowledge

    def _record_chancellor_output(self, chancellor_player, cards_in_order):
        """
        cards_in_order : liste des cartes remises dans le deck, dans l'ordre où
        elles sont insérées au fond (index 0 du deck).

        L'ordre d'insertion doit correspondre à la sémantique :
        la première carte insérée finit plus PROFONDE (vers le fond du deck),
        la seconde est juste au-dessus.

        Tes positions trackées : profondeur depuis le top = len(deck) - 1 - index_dans_deck
        Après insertion, les 2 cartes sont aux index 0 et 1 de self._deck (le fond).
        La position depuis le top de self._deck[0] = len(deck) - 1
        La position depuis le top de self._deck[1] = len(deck) - 2
        """
        deck_len = len(self._deck)
        if len(cards_in_order) == 2:
            # Premier inséré = index 0 = plus profond = position len-1
            # Second inséré = index 1 = position len-2
            self._deck_knowledge[chancellor_player] = {
                deck_len - 1: cards_in_order[0],
                deck_len - 2: cards_in_order[1],
            }
        elif len(cards_in_order) == 1:
            self._deck_knowledge[chancellor_player] = {
                deck_len - 1: cards_in_order[0],
            }
        else:
            self._deck_knowledge[chancellor_player] = {}

    # ==========================================
    # HELPER : éliminer un joueur
    # ==========================================
    def _eliminate(self, who):
        if self.terminations.get(who, False):
            return
        self.terminations[who] = True
        while self._hands.get(who):
            self._played_cards[who].append(self._hands[who].pop())
        # Nettoyage des states associés
        self._min_kept_card[who] = 0
        self._deck_knowledge[who] = {}

    # ==========================================
    # STEP
    # ==========================================
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            if self.agents and self.agent_selection not in self.agents:
                self.agent_selection = self.agents[0]
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.rewards = {a: 0.0 for a in self.agents}

        # --- SOUS-ÉTAT CHANCELIER ---
        if self._chancellor_pending:
            self._resolve_chancellor(agent, action)
            if self.agents and not all(self.terminations[a] for a in self.agents):
                self._pass_turn()
            self._accumulate_rewards()
            return

        # --- DÉCODAGE DE L'ACTION ---
        card_played = action // 100
        target_idx = (action % 100) // 10
        guess = action % 10

        hand = self._hands[agent]
        if card_played not in hand:
            # Fallback + règle Comtesse forcée
            if 8 in hand and (5 in hand or 7 in hand):
                card_played = 8
            else:
                card_played = hand[0]

        target = f"player_{target_idx}" if target_idx < self.num_players else None
        if target and (target not in self.agents or self.terminations.get(target, True)):
            target = None

        # Détection Comtesse volontaire (AVANT de retirer la carte de la main)
        if card_played == 8:
            forced = (5 in hand or 7 in hand)
            if not forced:
                self._countess_voluntary[agent] = True

        hand.remove(card_played)
        self._played_cards[agent].append(card_played)

        # Invalidation de connaissance sur la carte jouée
        for a in self.possible_agents:
            self._known_cards[a][agent][card_played] = 0.0

        # Invalidation de la borne min_kept si le joueur vient de jouer
        # la carte qui la justifiait. Simplification : on invalide dès qu'il
        # joue n'importe quelle carte (pessimiste mais sûr).
        # En pratique : si le joueur avait min_kept = 5 (via Baron gagné avec 5),
        # et qu'il joue autre chose, il a toujours le 5 → la borne reste valide.
        # Mais s'il joue le 5 directement, la borne ne s'applique plus.
        # On vérifie : si la carte jouée >= min_kept, alors min_kept pouvait
        # être justifiée par cette carte. Pour être safe, on reset.
        if card_played >= self._min_kept_card[agent]:
            # Ambigu : on ne sait pas si c'était LA carte qui justifiait la borne.
            # Reset pour être safe.
            self._min_kept_card[agent] = 0

        # --- APPLICATION DES EFFETS ---
        self._apply_effect(agent, card_played, target, guess)

        if self._chancellor_pending:
            self._accumulate_rewards()
            return

        self._pass_turn()
        self._accumulate_rewards()

    def _resolve_chancellor(self, agent, action):
        """Résout le sous-état Chancelier."""
        action_idx = action - 900
        pool_size = len(self._chancellor_pool)

        if pool_size >= 3:
            keep_idx = action_idx // 2
            order_idx = action_idx % 2
        elif pool_size == 2:
            keep_idx = action_idx
            order_idx = 0
        else:
            keep_idx = 0
            order_idx = 0

        if keep_idx >= pool_size or keep_idx < 0:
            keep_idx = 0

        kept_card = self._chancellor_pool.pop(keep_idx)
        self._hands[agent] = [kept_card]

        if order_idx == 1:
            self._chancellor_pool.reverse()

        # Les cartes du pool sont remises au fond du deck.
        # Ordre sémantique : self._chancellor_pool[0] est insérée en premier,
        # ce qui veut dire qu'elle finit PLUS PROFONDE que self._chancellor_pool[1].
        # (car chaque insert(0, card) pousse les cartes précédentes vers l'index 1+).
        cards_in_order_of_insertion = []
        for card in self._chancellor_pool:
            self._deck.insert(0, card)
            cards_in_order_of_insertion.append(card)

        # Enregistre la connaissance du deck pour ce joueur
        self._record_chancellor_output(agent, cards_in_order_of_insertion)

        # Les autres joueurs voient un Chancelier passer → invalidation partielle
        for observer in self.possible_agents:
            if observer == agent:
                continue
            knowledge = self._deck_knowledge[observer]
            if any(pos <= 1 for pos in knowledge):
                self._deck_knowledge[observer] = {}
            else:
                new_knowledge = {}
                for pos, card in knowledge.items():
                    new_knowledge[pos - 2] = card
                self._deck_knowledge[observer] = new_knowledge

        # Reset : main a changé
        self._turns_hand_unchanged[agent] = 0
        self._min_kept_card[agent] = 0

        self._chancellor_pending = False
        self._chancellor_pool = []

    # ==========================================
    # PASS TURN
    # ==========================================
    def _pass_turn(self):
        alive = [a for a in self.agents if not self.terminations[a]]

        if len(alive) <= 1 or not self._deck:
            self._resolve_round(alive)
            if self.agents:
                self.agent_selection = self.agents[0]
            return

        # Recherche linéaire du prochain vivant
        start_idx = self.possible_agents.index(self.agent_selection)
        nxt = None
        for offset in range(1, len(self.possible_agents) + 1):
            idx = (start_idx + offset) % len(self.possible_agents)
            candidate = self.possible_agents[idx]
            if candidate in self.agents and not self.terminations.get(candidate, True):
                nxt = candidate
                break
        if nxt is None:
            return
        
        current_player = self.agent_selection
        for a in alive:
            if a == nxt or a == current_player :
                continue
            self._turns_hand_unchanged[a] += 1

        # Increment du "main inchangée" pour tous les vivants SAUF celui qui va piocher
        # (celui qui va piocher verra son compteur reset dans _draw_card)
        # for a in alive:
        #     if a != nxt:
        #         # Le joueur a n'a pas pioché ce tour → sa main n'a pas changé
        #         self._turns_hand_unchanged[a] += 1

        self.agent_selection = nxt
        self._protected[nxt] = False
        self._draw_card(nxt)

    # ==========================================
    # APPLY EFFECT
    # ==========================================
    def _apply_effect(self, agent, card, target, guess):
        if card == 1 and target:
            if self._hands[target] and self._hands[target][0] == guess:
                self._eliminate(target)
                self.rewards[agent] += 0.2

        elif card == 2 and target:  # Prêtre
            if self._hands[target]:
                seen_card = self._hands[target][0]
                self._known_cards[agent][target].fill(0.0)
                self._known_cards[agent][target][seen_card] = 1.0

        elif card == 3 and target:  # Baron
            if self._hands[agent] and self._hands[target]:
                my_val = self._hands[agent][0]
                their_val = self._hands[target][0]
                if my_val > their_val:
                    self._eliminate(target)
                    self.rewards[agent] += 0.2
                    # Contrainte bayésienne : agent a gardé une carte >= my_val
                    self._min_kept_card[agent] = my_val
                elif their_val > my_val:
                    self._eliminate(agent)
                    self.rewards[agent] -= 0.2
                    self._min_kept_card[target] = their_val
                # Égalité : rien

        elif card == 4:  # Servante
            self._protected[agent] = True

        elif card == 5 and target:  # Prince
            for a in self.possible_agents:
                self._known_cards[a][target].fill(0.0)
            self._min_kept_card[target] = 0

            if self._hands[target]:
                discarded = self._hands[target].pop()
                self._played_cards[target].append(discarded)

                if discarded == 9:
                    self._eliminate(target)
                else:
                    if self._deck:
                        self._draw_card(target)
                    elif self._set_aside is not None:
                        self._hands[target].append(self._set_aside)
                        self._set_aside = None
                        # Note : set-aside devient connu de TOUS à ce moment
                        for a in self.possible_agents:
                            if a != target:
                                # Tout le monde peut déduire que target a le set-aside
                                # si le set-aside est connu. Ici le set-aside est
                                # privé (nobody knew it). Donc on ne propage pas.
                                pass

            # Reset des compteurs : main a changé
            self._turns_hand_unchanged[target] = 0

        elif card == 6:  # Chancelier (déjà retiré de la main)
            self._chancellor_pool = list(self._hands[agent])
            self._hands[agent] = []
            draw_count = min(2, len(self._deck))
            for _ in range(draw_count):
                card_drawn = self._deck.pop()
                self._chancellor_pool.append(card_drawn)
                self._shift_deck_knowledge_on_draw()
            self._chancellor_pending = True

        elif card == 7 and target:  # Roi
            if self._hands[agent] and self._hands[target]:
                self._hands[agent], self._hands[target] = self._hands[target], self._hands[agent]
                for a in self.possible_agents:
                    self._known_cards[a][agent].fill(0.0)
                    self._known_cards[a][target].fill(0.0)
                # Les deux joueurs se connaissent mutuellement
                agent_card = self._hands[agent][0]
                target_card = self._hands[target][0]
                self._known_cards[agent][target][target_card] = 1.0
                self._known_cards[target][agent][agent_card] = 1.0
                # Reset
                self._turns_hand_unchanged[agent] = 0
                self._turns_hand_unchanged[target] = 0
                self._min_kept_card[agent] = 0
                self._min_kept_card[target] = 0

        elif card == 9:  # Princesse
            self._eliminate(agent)

    # ==========================================
    # RESOLVE ROUND
    # ==========================================
    def _resolve_round(self, alive_agents):
        for a in self.agents:
            self.terminations[a] = True
        self._round_winners = []
        self._round_win_reason = None
        self._round_spy_winner = None

        valid_agents = [a for a in alive_agents if self._hands[a]]
        if len(valid_agents) == 1:
            self.rewards[valid_agents[0]] += 1.0
            self._round_winners = [valid_agents[0]]
            self._round_win_reason = "last_alive"
        elif len(valid_agents) > 1:
            max_val = max([self._hands[a][0] for a in valid_agents])
            winners = [a for a in valid_agents if self._hands[a][0] == max_val]
            for w in winners:
                self.rewards[w] += 1.0
            self._round_winners = winners
            self._round_win_reason = "highest_card"

        spy_counts = {a: self._played_cards[a].count(0) for a in alive_agents}
        if spy_counts:
            max_spies = max(spy_counts.values())
            if max_spies > 0:
                spy_masters = [a for a, c in spy_counts.items() if c == max_spies]
                if len(spy_masters) == 1:
                    self.rewards[spy_masters[0]] += 1.0
                    self._round_spy_winner = spy_masters[0]

    # ==========================================
    # OBSERVE
    # ==========================================
    def observe(self, agent):
        mask = np.zeros(1000, dtype=np.int8)
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        idx = 0

        my_idx = self.possible_agents.index(agent)

        # Hidden cards pour info (belief head target)
        opp_cards = []
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            if self._hands.get(opp):
                opp_cards.append(self._hands[opp][0])
            else:
                opp_cards.append(-1)
        self.infos[agent]["hidden_cards"] = np.array(opp_cards, dtype=np.int64)

        # [0:10] Ma main
        for card in self._hands.get(agent, []):
            obs[idx + card] += 1.0 / 3.0
        idx += 10

        # [10:20] Cartes jouées globalement
        global_played_counts = np.zeros(10, dtype=np.float32)
        for a in self.possible_agents:
            for c in self._played_cards.get(a, []):
                global_played_counts[c] += 1.0
        obs[idx:idx + 10] = global_played_counts / self.MAX_CARD_COUNTS
        idx += 10

        # [20:30] Cartes restantes dans deck+set_aside
        remaining = self.MAX_CARD_COUNTS.copy()
        remaining -= global_played_counts
        for card in self._hands.get(agent, []):
            remaining[card] -= 1.0
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            for c in range(10):
                if self._known_cards[agent][opp][c] >= 1.0:
                    remaining[c] -= 1.0
        remaining = np.clip(remaining, 0.0, None)
        obs[idx:idx + 10] = remaining / self.MAX_CARD_COUNTS
        idx += 10

        # [30:36] État adversaires : alive, protected
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            is_dead = opp not in self.agents or self.terminations.get(opp, True)
            obs[idx] = 1.0 if not is_dead else 0.0
            obs[idx + 1] = 1.0 if self._protected.get(opp, False) else 0.0
            idx += 2

        # [36:66] Dernière carte jouée par chaque adversaire
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            played = self._played_cards.get(opp, [])
            if played:
                obs[idx + played[-1]] = 1.0
            idx += 10

        # [66:69] Nombre de cartes jouées par chaque adversaire
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = min(len(self._played_cards.get(opp, [])) / 5.0, 1.0)
            idx += 1

        # [69] Taille du deck
        obs[idx] = len(self._deck) / 21.0
        idx += 1

        # [70:80] Mes propres cartes jouées
        my_played_counts = np.zeros(10, dtype=np.float32)
        for c in self._played_cards.get(agent, []):
            my_played_counts[c] += 1.0
        obs[idx:idx + 10] = my_played_counts / self.MAX_CARD_COUNTS
        idx += 10

        # [80] Flag chancelier actif
        obs[idx] = 1.0 if self._chancellor_pending else 0.0
        idx += 1

        # [81:111] Connaissance des cartes adverses
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            for c in range(10):
                obs[idx + c] = self._known_cards[agent][opp][c]
            idx += 10

        # ===== NOUVEAUX BLOCS (v2) =====

        # [111:115] Tokens de tous les joueurs (y compris soi)
        # On met soi-même en premier puis adversaires dans l'ordre
        obs[idx] = min(self._tokens[agent] / 2.0, 1.0)
        idx += 1
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = min(self._tokens[opp] / 2.0, 1.0)
            idx += 1

        # [115:119] Espionne count (soi + adversaires)
        spy_count_self = self._played_cards.get(agent, []).count(0)
        obs[idx] = min(spy_count_self / 2.0, 1.0)
        idx += 1
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = min(self._played_cards.get(opp, []).count(0) / 2.0, 1.0)
            idx += 1

        # [119:122] Main inchangée (adversaires, 3 dims)
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = min(self._turns_hand_unchanged[opp] / 5.0, 1.0)
            idx += 1

        # [122:125] Comtesse volontaire (adversaires, 3 dims binaires)
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = 1.0 if self._countess_voluntary[opp] else 0.0
            idx += 1

        # [125:155] Knowledge deck : 3 positions × 10 cartes one-hot
        # Positions 0, 1, 2 (top, top-1, top-2)
        my_knowledge = self._deck_knowledge[agent]
        for pos in range(self.MAX_DECK_POS_TRACKED):
            if pos in my_knowledge:
                card = my_knowledge[pos]
                obs[idx + card] = 1.0
            idx += 10

        # [155:158] Min kept card (adversaires, normalisé /9)
        for i in range(1, self.num_players):
            opp_idx = (my_idx + i) % self.num_players
            opp = self.possible_agents[opp_idx]
            obs[idx] = self._min_kept_card[opp] / 9.0
            idx += 1

        # idx doit égaler OBS_DIM
        assert idx == self.OBS_DIM, f"Obs size mismatch: {idx} vs {self.OBS_DIM}"

        # ==========================================
        # GÉNÉRATION DU MASQUE (inchangé)
        # ==========================================
        if self._chancellor_pending and agent == self.agent_selection:
            pool_size = len(self._chancellor_pool)
            valid_perms = 6 if pool_size == 3 else (2 if pool_size == 2 else 1)
            for i in range(valid_perms):
                mask[900 + i] = 1
            return {"observation": obs, "action_mask": mask}

        hand = self._hands.get(agent, [])
        if not hand:
            return {"observation": obs, "action_mask": mask}

        dummy = 9
        valid_opponents = [
            i for i, a in enumerate(self.possible_agents)
            if a in self.agents and not self.terminations.get(a, True)
            and not self._protected.get(a, False) and a != agent
        ]

        must_play_countess = 8 in hand and (5 in hand or 7 in hand)

        for card in set(hand):
            if must_play_countess and card != 8:
                continue

            if card == 1:
                if valid_opponents:
                    for t_idx in valid_opponents:
                        for g in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
                            mask[(card * 100) + (t_idx * 10) + g] = 1
                else:
                    mask[(card * 100) + (dummy * 10) + 0] = 1
            elif card in [2, 3, 7]:
                if valid_opponents:
                    for t_idx in valid_opponents:
                        mask[(card * 100) + (t_idx * 10) + 0] = 1
                else:
                    mask[(card * 100) + (dummy * 10) + 0] = 1
            elif card == 5:
                valid_targets_prince = valid_opponents + [my_idx]
                for t_idx in valid_targets_prince:
                    mask[(card * 100) + (t_idx * 10) + 0] = 1
            else:
                mask[(card * 100) + (dummy * 10) + 0] = 1

        return {"observation": obs, "action_mask": mask}
