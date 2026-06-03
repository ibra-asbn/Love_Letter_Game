# Diagnostic modele - champion_belief_ppo_attempt1_best.pth

Date: 2026-04-24

Configuration: player_0 joue le checkpoint, player_1/player_2/player_3 jouent HeuristicBot.
Log brut: `logs/evaluations/2026-04-24_attempt1_best_vs_3heuristics_diagnostic.json`

## Resultat

- Parties: 30
- Victoires: 4 (13.3%)
- Reward moyen: 0.207

## Actions jouees par player_0

- Garde: 24
- Pretre: 10
- Chancelier: 9
- Chancelier choice: 9
- Prince: 7
- Baron: 4
- Comtesse: 4
- Servante: 4
- Espionne: 2
- Roi: 1

## Moments bizarres

- Garde devine une carte moins probable que le top belief.: 12
- Chancelier ne garde pas la meilleure option visible dans le pool.: 8
- Baron perdant: il cible une main plus forte.: 1
- Roi donne une meilleure carte que celle recue.: 1

## Fulgurances

- Garde touche exactement la carte adverse.: 9
- Baron favorable: la carte gardee bat la cible.: 3
- Chancelier garde la meilleure carte non-Princesse du pool.: 1
- Prince force la Princesse adverse.: 1

## Exemples

### Garde devine une carte moins probable que le top belief.
- Partie 3: main=['Roi', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Garde, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.40970635414123535}, {'card': 3, 'card_name': 'Baron', 'prob': 0.13668109476566315}, {'card': 5, 'card_name': 'Prince', 'prob': 0.09072592109441757}]
- Partie 3: main=['Roi', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Pretre, belief_top=[{'card': 8, 'card_name': 'Comtesse', 'prob': 0.29385313391685486}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.2337544709444046}, {'card': 9, 'card_name': 'Princesse', 'prob': 0.17977292835712433}]
- Partie 10: main=['Servante', 'Garde'], action=Garde -> player_2 guess Chancelier, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.4610440135002136}, {'card': 6, 'card_name': 'Chancelier', 'prob': 0.3231755793094635}, {'card': 7, 'card_name': 'Roi', 'prob': 0.18287380039691925}]

### Garde touche exactement la carte adverse.
- Partie 5: main=['Garde', 'Garde'], action=Garde -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.23013150691986084}, {'card': 4, 'card_name': 'Servante', 'prob': 0.14679180085659027}, {'card': 3, 'card_name': 'Baron', 'prob': 0.12522760033607483}]
- Partie 6: main=['Garde', 'Garde'], action=Garde -> player_1 guess Baron, carte cible=Baron, belief_top=[{'card': 3, 'card_name': 'Baron', 'prob': 0.2632094621658325}, {'card': 4, 'card_name': 'Servante', 'prob': 0.17476032674312592}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.13629001379013062}]
- Partie 6: main=['Garde', 'Princesse'], action=Garde -> player_3 guess Espionne, carte cible=Espionne, belief_top=[{'card': 0, 'card_name': 'Espionne', 'prob': 0.38695481419563293}, {'card': 8, 'card_name': 'Comtesse', 'prob': 0.1871335357427597}, {'card': 4, 'card_name': 'Servante', 'prob': 0.17973555624485016}]

### Chancelier ne garde pas la meilleure option visible dans le pool.
- Partie 5: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 8: main=[], action=choix Chancelier, carte cible=None, belief_top=None
- Partie 9: main=[], action=choix Chancelier, carte cible=None, belief_top=None

### Baron favorable: la carte gardee bat la cible.
- Partie 11: main=['Princesse', 'Baron'], action=Baron -> player_1 guess Espionne, carte cible=Espionne, belief_top=[{'card': 4, 'card_name': 'Servante', 'prob': 0.33731985092163086}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.3288785219192505}, {'card': 1, 'card_name': 'Garde', 'prob': 0.17986269295215607}]
- Partie 22: main=['Baron', 'Comtesse'], action=Baron -> player_1 guess Espionne, carte cible=Pretre, belief_top=[{'card': 2, 'card_name': 'Pretre', 'prob': 0.25237730145454407}, {'card': 5, 'card_name': 'Prince', 'prob': 0.2116852104663849}, {'card': 6, 'card_name': 'Chancelier', 'prob': 0.145537331700325}]
- Partie 30: main=['Prince', 'Baron'], action=Baron -> player_1 guess Espionne, carte cible=Servante, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.37411049008369446}, {'card': 3, 'card_name': 'Baron', 'prob': 0.15437182784080505}, {'card': 4, 'card_name': 'Servante', 'prob': 0.13694055378437042}]

### Prince force la Princesse adverse.
- Partie 11: main=['Prince', 'Espionne'], action=Prince -> player_3 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.8404577374458313}, {'card': 4, 'card_name': 'Servante', 'prob': 0.05324959754943848}, {'card': 2, 'card_name': 'Pretre', 'prob': 0.04617862403392792}]

### Roi donne une meilleure carte que celle recue.
- Partie 13: main=['Princesse', 'Roi'], action=Roi -> player_1 guess Espionne, carte cible=Garde, belief_top=[{'card': 1, 'card_name': 'Garde', 'prob': 0.3172171413898468}, {'card': 3, 'card_name': 'Baron', 'prob': 0.2849762737751007}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.12344387918710709}]

### Baron perdant: il cible une main plus forte.
- Partie 14: main=['Baron', 'Prince'], action=Baron -> player_1 guess Espionne, carte cible=Princesse, belief_top=[{'card': 9, 'card_name': 'Princesse', 'prob': 0.908036470413208}, {'card': 7, 'card_name': 'Roi', 'prob': 0.07250464707612991}, {'card': 0, 'card_name': 'Espionne', 'prob': 0.0045125470496714115}]

### Chancelier garde la meilleure carte non-Princesse du pool.
- Partie 27: main=[], action=choix Chancelier, carte cible=None, belief_top=None
