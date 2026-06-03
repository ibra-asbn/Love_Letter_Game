# Diagnostic modele - champion_belief_ppo_attempt2_tactical_best.pth

Date: 2026-04-24

Configuration: player_0 joue le checkpoint, player_1/player_2/player_3 jouent HeuristicBot.
Log brut: `logs/evaluations/2026-04-24_attempt2_tactical_best_vs_3heuristics_diagnostic.json`

## Resultat

- Parties: 30
- Victoires: 4 (13.3%)
- Reward moyen: 0.180

## Actions jouees par player_0

- Garde: 30
- Prince: 8
- Pretre: 7
- Chancelier: 6
- Chancelier choice: 6
- Servante: 5
- Baron: 4
- Comtesse: 2
- Roi: 2
- Espionne: 1

## Moments bizarres

- Garde devine une carte moins probable que le top belief.: 15
- Chancelier ne garde pas la meilleure option visible dans le pool.: 5
- Roi donne une meilleure carte que celle recue.: 2

## Fulgurances

- Baron favorable: la carte gardee bat la cible.: 4
- Garde touche exactement la carte adverse.: 3
- Prince force la Princesse adverse.: 3
- Chancelier garde la meilleure carte non-Princesse du pool.: 1

## Exemples

### Garde devine une carte moins probable que le top belief.
- Partie 3: main=['Roi', 'Garde'], action=Garde -> player_2 guess Prince, carte cible=Baron, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.24198777973651886}, {'card': 1, 'card_name': 'Garde', 'prob': 0.2182125747203827}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.2026720643043518}]
- Partie 5: main=['Garde', 'Garde'], action=Garde -> player_3 guess Prince, carte cible=Garde, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.5467849969863892}, {'card': 5, 'card_name': 'Prince', 'prob': 0.3082844018936157}, {'card': 7, 'card_name': 'Roi', 'prob': 0.060990579426288605}]
- Partie 10: main=['Servante', 'Garde'], action=Garde -> player_2 guess Chancelier, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.526094913482666}, {'card': 6, 'card_name': 'Chancelier', 'prob': 0.23489174246788025}, {'card': 7, 'card_name': 'Roi', 'prob': 0.19152510166168213}]

### Garde touche exactement la carte adverse.
- Partie 5: main=['Garde', 'Garde'], action=Garde -> player_1 guess Baron, carte cible=Baron, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.4529964327812195}, {'card': 3, 'card_name': 'Baron', 'prob': 0.32038816809654236}, {'card': 7, 'card_name': 'Roi', 'prob': 0.07411307096481323}]
- Partie 8: main=['Garde', 'Chancelier'], action=Garde -> player_2 guess Prince, carte cible=Prince, belief_top=[{'card': 5, 'card_name': 'Prince', 'prob': 0.3392889201641083}, {'card': 1, 'card_name': 'Garde', 'prob': 0.2131013572216034}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.11319070309400558}]
- Partie 23: main=['Servante', 'Garde'], action=Garde -> player_1 guess Comtesse, carte cible=Comtesse, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.6831413507461548}, {'card': 5, 'card_name': 'Prince', 'prob': 0.1642381101846695}, {'card': 7, 'card_name': 'Roi', 'prob': 0.05376248061656952}]

### Chancelier garde la meilleure carte non-Princesse du pool.
- Partie 8: main=[], action=choix Chancelier, carte cible=None, belief_top=None

### Chancelier ne garde pas la meilleure option visible dans le pool.
- Partie 9: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 12: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 13: main=[], action=choix Chancelier, carte cible=None, belief_top=None

### Baron favorable: la carte gardee bat la cible.
- Partie 11: main=['Princesse', 'Baron'], action=Baron -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 0, 'card_name': 'Espionne', 'prob': 0.46354177594184875}, {'card': 4, 'card_name': 'Servante', 'prob': 0.25993213057518005}, {'card': 1, 'card_name': 'Garde', 'prob': 0.14746695756912231}]
- Partie 22: main=['Baron', 'Comtesse'], action=Baron -> player_2 guess Espionne, carte cible=Roi, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.38181811571121216}, {'card': 9, 'card_name': 'Princesse', 'prob': 0.17792440950870514}, {'card': 7, 'card_name': 'Roi', 'prob': 0.1638576090335846}]
- Partie 28: main=['Prince', 'Baron'], action=Baron -> player_3 guess Espionne, carte cible=Baron, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.45657581090927124}, {'card': 3, 'card_name': 'Baron', 'prob': 0.1617790013551712}, {'card': 7, 'card_name': 'Roi', 'prob': 0.09399180114269257}]

### Prince force la Princesse adverse.
- Partie 11: main=['Prince', 'Espionne'], action=Prince -> player_3 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.8271041512489319}, {'card': 4, 'card_name': 'Servante', 'prob': 0.04672299325466156}, {'card': 2, 'card_name': 'Pretre', 'prob': 0.044056784361600876}]
- Partie 14: main=['Baron', 'Prince'], action=Prince -> player_1 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.9149739146232605}, {'card': 7, 'card_name': 'Roi', 'prob': 0.06240759417414665}, {'card': 5, 'card_name': 'Prince', 'prob': 0.007704995572566986}]
- Partie 17: main=['Prince', 'Servante'], action=Prince -> player_2 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.48243799805641174}, {'card': 3, 'card_name': 'Baron', 'prob': 0.25267940759658813}, {'card': 5, 'card_name': 'Prince', 'prob': 0.1948799192905426}]

### Roi donne une meilleure carte que celle recue.
- Partie 13: main=['Princesse', 'Roi'], action=Roi -> player_1 guess Espionne, carte cible=Garde, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.3066897690296173}, {'card': 1, 'card_name': 'Garde', 'prob': 0.2427632063627243}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.1599571257829666}]
- Partie 16: main=['Roi', 'Princesse'], action=Roi -> player_1 guess Espionne, carte cible=Prince, belief_top=[{'card': 5, 'card_name': 'Prince', 'prob': 0.2939397990703583}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.28477242588996887}, {'card': 3, 'card_name': 'Baron', 'prob': 0.16279754042625427}]
