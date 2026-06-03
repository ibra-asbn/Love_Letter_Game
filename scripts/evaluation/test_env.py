from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from pettingzoo.test import api_test
from love_letter.engine import LoveLetterRLEnv

# Dictionnaire de traduction pour un affichage humain
CARD_NAMES = {
    0: "Espionne (0)",
    1: "Garde (1)",
    2: "Prêtre (2)",
    3: "Baron (3)",
    4: "Servante (4)",
    5: "Prince (5)",
    6: "Chancelier (6)",
    7: "Roi (7)",
    8: "Comtesse (8)",
    9: "Princesse (9)"
}

def run_random_game():
    print("--- 1. TEST DE L'API PETTINGZOO ---")
    env = LoveLetterRLEnv(num_players=4)
    
    try:
        api_test(env, num_cycles=1000, verbose_progress=False)
        print("✅ Le test de l'API PettingZoo est un succès !\n")
    except Exception as e:
        print("❌ Échec du test de l'API :", e)
        return

    print("=========================================")
    print("      SIMULATION D'UNE PARTIE TYPE       ")
    print("=========================================")
    
    env.reset()
    
    print("\n[ DISTRIBUTION INITIALE ]")
    for agent in env.possible_agents:
        hand = [CARD_NAMES[c] for c in env._hands.get(agent, [])]
        print(f"{agent} reçoit : {hand[0]}")
    print(f"Carte mise de côté (secrète) : {CARD_NAMES[env._set_aside]}")
    print("-----------------------------------------\n")
    
    step_count = 0
    tracked_rewards = {a: 0.0 for a in env.possible_agents}
    
    # RADAR : On enregistre qui est en vie au début de la partie
    joueurs_en_vie = set(env.possible_agents)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        tracked_rewards[agent] += reward

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            
            if len(valid_actions) == 0:
                print(f"⚠️ ERREUR : Le masque de {agent} ne contient aucune action valide !")
                break
                
            action = np.random.choice(valid_actions)
            
            deck_size = len(env._deck)
            current_hand = [CARD_NAMES[c] for c in env._hands[agent]]
            
            if not getattr(env, '_chancellor_pending', False):
                print(f"\n--- Tour {step_count:02d} | {agent} ---")
                print(f"📦 Pioche restante : {deck_size} cartes")
                print(f"🃏 Main actuelle : {current_hand}")
                
                card = action // 100
                target_idx = (action % 100) // 10
                guess = action % 10
                
                card_name = CARD_NAMES.get(card, f"Inconnue ({card})")
                target_str = f"player_{target_idx}" if target_idx < env.num_players else "Personne"
                guess_str = CARD_NAMES.get(guess, str(guess))
                
                if card == 1:
                    print(f"🎯 ACTION : Joue {card_name} sur {target_str} et devine {guess_str}")
                elif card in [2, 3, 5, 7]:
                    print(f"🎯 ACTION : Joue {card_name} sur {target_str}")
                else:
                    print(f"🎯 ACTION : Joue {card_name} (sans cible / sur soi-même)")
            
            else:
                action_idx = action - 900
                keep_idx = action_idx // 2
                pool = [CARD_NAMES[c] for c in env._chancellor_pool]
                kept_card = pool[keep_idx] if keep_idx < len(pool) else "inconnue"
                
                print(f"\n--- Tour {step_count:02d} | {agent} (RÉSOLUTION CHANCELIER) ---")
                print(f"🃏 Cartes examinées : {pool}")
                print(f"🎯 ACTION : Garde {kept_card} et remet le reste sous la pioche.")

        # L'environnement applique l'action
        env.step(action)
        
        # --- TRACKING DES ÉLIMINATIONS EN DIRECT ---
        # On regarde qui est toujours vivant après l'action
        vivants_actuels = set([a for a in env.possible_agents if not env.terminations.get(a, True)])
        
        # On soustrait : si un joueur était en vie avant l'action mais plus maintenant, il est mort
        morts_recents = joueurs_en_vie - vivants_actuels
        
        if morts_recents:
            # On évite le spam de fin de partie quand l'environnement tue tout le monde pour clore le jeu
            if len(vivants_actuels) > 0 and len(vivants_actuels) < len(env.possible_agents): 
                for mort in morts_recents:
                    print(f"   💀 BOUM ! {mort} a été éliminé suite à cette action !")
                print(f"   👥 Participants restants : {', '.join(sorted(vivants_actuels))}")
            
            # Mise à jour du radar
            joueurs_en_vie = vivants_actuels

        if not (termination or truncation):
            step_count += 1

    print("\n=========================================")
    print("             PARTIE TERMINÉE !             ")
    print("=========================================")
    
    print("\nSurviants et leurs mains finales :")
    for agent in env.possible_agents:
        if len(env._hands.get(agent, [])) > 0:
            hand = [CARD_NAMES[c] for c in env._hands[agent]]
            print(f"- {agent} a survécu avec : {hand}")
        else:
            print(f"- {agent} : Éliminé 💀")

    print("\nDétail du score de la manche :")
    winners = getattr(env, "_round_winners", None)
    reason = getattr(env, "_round_win_reason", None)
    if winners:
        if reason == "last_alive":
            w = winners[0]
            print(f"🏆 Point manche : {w} (dernier survivant)")
        else:  # highest_card
            cartes = ", ".join(
                f"{w} ({CARD_NAMES[env._hands[w][0]]})"
                for w in winners if env._hands[w]
            )
            print(f"🏆 Point manche : {cartes} (plus haute carte)")
    if getattr(env, "_round_spy_winner", None):
        print(f"🕵️  Bonus Espionne : {env._round_spy_winner}")

if __name__ == "__main__":
    run_random_game()