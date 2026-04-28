# Diagnostic modele - belief_conditioned_ppo_final.pth

Date: 2026-04-24

Configuration: player_0 joue le checkpoint, player_1/player_2/player_3 jouent HeuristicBot.
Log brut: `logs/evaluations/2026-04-24_belief_conditioned_ppo_vs_3heuristics_detailed.json`

## Resultat

- Parties: 30
- Victoires: 4 (13.3%)
- Reward moyen: 0.227

## Actions jouees par player_0

- Garde: 22
- Pretre: 12
- Chancelier: 10
- Chancelier choice: 10
- Baron: 7
- Prince: 6
- Servante: 6
- Comtesse: 3
- Espionne: 3

## Moments bizarres

- Garde devine une carte moins probable que le top belief.: 10
- Chancelier ne garde pas la meilleure option visible dans le pool.: 2
- Baron perdant: il cible une main plus forte.: 1

## Fulgurances

- Chancelier garde la meilleure carte non-Princesse du pool.: 8
- Baron favorable: la carte gardee bat la cible.: 6
- Garde touche exactement la carte adverse.: 4
- Prince force la Princesse adverse.: 1

## Exemples

### Garde devine une carte moins probable que le top belief.
- Partie 3: main=['Roi', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Garde, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.3791550397872925}, {'card': 3, 'card_name': 'Baron', 'prob': 0.14185020327568054}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.11330137401819229}]
- Partie 3: main=['Roi', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Pretre, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.3780125379562378}, {'card': 9, 'card_name': 'Princesse', 'prob': 0.2264319807291031}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.19190987944602966}]
- Partie 6: main=['Garde', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Baron, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.22428162395954132}, {'card': 4, 'card_name': 'Servante', 'prob': 0.1551554799079895}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.12430185824632645}]

### Garde touche exactement la carte adverse.
- Partie 5: main=['Garde', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.22179976105690002}, {'card': 4, 'card_name': 'Servante', 'prob': 0.13585664331912994}, {'card': 3, 'card_name': 'Baron', 'prob': 0.12151303887367249}]
- Partie 16: main=['Garde', 'Comtesse'], action=Garde -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.25073984265327454}, {'card': 3, 'card_name': 'Baron', 'prob': 0.11681583523750305}, {'card': 2, 'card_name': 'Pretre', 'prob': 0.11437497287988663}]
- Partie 17: main=['Garde', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.2493029087781906}, {'card': 5, 'card_name': 'Prince', 'prob': 0.11825430393218994}, {'card': 4, 'card_name': 'Servante', 'prob': 0.11503316462039948}]

### Chancelier garde la meilleure carte non-Princesse du pool.
- Partie 5: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 8: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 9: main=[], action=choix Chancelier, carte cible=None, belief_top=None

### Baron favorable: la carte gardee bat la cible.
- Partie 9: main=['Comtesse', 'Baron'], action=Baron -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.4436592161655426}, {'card': 4, 'card_name': 'Servante', 'prob': 0.14785872399806976}, {'card': 5, 'card_name': 'Prince', 'prob': 0.10662374645471573}]
- Partie 11: main=['Princesse', 'Baron'], action=Baron -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 0, 'card_name': 'Espionne', 'prob': 0.3953365385532379}, {'card': 4, 'card_name': 'Servante', 'prob': 0.2897613048553467}, {'card': 1, 'card_name': 'Garde', 'prob': 0.1261567324399948}]
- Partie 12: main=['Baron', 'Princesse'], action=Baron -> player_1 guess Espionne, carte cible=Servante, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.19808922708034515}, {'card': 1, 'card_name': 'Garde', 'prob': 0.19558191299438477}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.18811869621276855}]

### Prince force la Princesse adverse.
- Partie 11: main=['Prince', 'Espionne'], action=Prince -> player_3 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.960211992263794}, {'card': 1, 'card_name': 'Garde', 'prob': 0.016916699707508087}, {'card': 2, 'card_name': 'Pretre', 'prob': 0.008359141647815704}]

### Chancelier ne garde pas la meilleure option visible dans le pool.
- Partie 12: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 27: main=[], action=choix Chancelier, carte cible=None, belief_top=None

### Baron perdant: il cible une main plus forte.
- Partie 14: main=['Baron', 'Prince'], action=Baron -> player_1 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.9147746562957764}, {'card': 7, 'card_name': 'Roi', 'prob': 0.07090753316879272}, {'card': 5, 'card_name': 'Prince', 'prob': 0.0031037309672683477}]
