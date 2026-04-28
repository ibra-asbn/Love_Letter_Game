# Diagnostic modele Love Letter - 2026-04-24

Modele diagnostique: `curriculum_phase1.pth`
Checkpoint: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/models/checkpoints/curriculum_phase1.pth`

## Verdict court

`curriculum_phase1.pth` est le meilleur champion polyvalent observe dans les tests actuels, mais il n'est pas encore un joueur solide contre des heuristiques humaines: il domine surtout les randoms et perd beaucoup de robustesse quand les adversaires deviennent plus structures.

## Scores de reference

Evaluation 100 parties par configuration:

| Configuration | Winrate | Reward moyen |
|---|---:|---:|
| vs 3 randoms | 55% | 0.840 |
| vs 1 heuristique + 2 randoms | 43% | 0.648 |
| vs 2 heuristiques + 1 random | 33% | 0.490 |
| vs 3 heuristiques | 20% | 0.322 |

Run detaille du diagnostic, 30 parties vs 3 heuristiques:

- Winrate: 9/30 = 30.0%
- Reward moyen: 0.507
- Log brut: `logs/evaluations/2026-04-24_curriculum_phase1_vs_3heuristics_detailed.json`

Run separe ou ma politique tactique ad hoc jouait contre 3 copies de ce modele:
- Winrate de ma politique: 3/10 = 30%
- Reward moyen: 0.440

## Etat actuel du modele

Forces actuelles:

- Il a une base tactique reelle: il gagne souvent contre des randoms et reste competitif jusqu'a deux heuristiques.
- Il semble avoir appris a exploiter des erreurs grossieres: les rewards contre randoms sont nettement superieurs a une baseline aleatoire.
- Il ne s'effondre pas totalement contre 3 heuristiques, mais son winrate autour de 20% montre qu'il est encore derriere un style humain regle.

Limites actuelles:

- Le modele est sensible a la qualite des adversaires: de 55% vs randoms a 20% vs full heuristique.
- Le curriculum n'a pas donne une progression monotone: phase2 et phase3 ne battent pas phase1 dans les tests recents.
- Son score semble mieux refleter une competence opportuniste qu'une comprehension strategique stable de la manche.

## Incoherences et moments bizarres releves

- Roi potentiellement couteux avec carte gardee haute (Chancelier (6)): 1 occurrence(s) dans le run detaille.
- Prince offensif sans information certaine: 1 occurrence(s) dans le run detaille.
- Roi potentiellement couteux avec carte gardee haute (Princesse (9)): 1 occurrence(s) dans le run detaille.

Points structurels bizarres observes dans les evaluations:

- `curriculum_phase3_final.pth` est plus recent mais moins bon que `curriculum_phase1.pth` sur plusieurs configs, ce qui suggere un sur-apprentissage, une degradation PPO, ou une evaluation d'entrainement pas alignee avec le vrai objectif.
- `model_score_0.94_epoch_0.pth` a un nom tres flatteur, mais son score ne correspond pas a un winrate direct vs 3 heuristiques; il est probablement lie a une autre metrique/config.
- Le modele de l'app Streamlit et `play_vs_agent` etait `curriculum_phase2.pth`, pas le meilleur checkpoint observe.
- La difference entre 20% vs 3 heuristiques et 30% pour une politique tactique ad hoc contre 3 copies du modele indique que le champion reste exploitable.

## Fulgurances observees

- Partie 1: Fulgurance: Garde touche player_3 (Chancelier (6)) via `joue Garde (1) sur player_3, devine Chancelier (6)`.
- Partie 2: Fulgurance: Baron gagne contre player_1 via `joue Baron (3) sur player_1`.
- Partie 7: Fulgurance: Garde touche player_2 (Prince (5)) via `joue Garde (1) sur player_2, devine Prince (5)`.
- Partie 8: Fulgurance: Garde touche player_2 (Prince (5)) via `joue Garde (1) sur player_2, devine Prince (5)`.
- Partie 8: Fulgurance: Garde touche player_1 (Baron (3)) via `joue Garde (1) sur player_1, devine Baron (3)`.
- Partie 13: Fulgurance: Baron gagne contre player_1 via `joue Baron (3) sur player_1`.
- Partie 14: Fulgurance: Garde touche player_3 (Princesse (9)) via `joue Garde (1) sur player_3, devine Princesse (9)`.
- Partie 19: Fulgurance: Garde touche player_1 (Comtesse (8)) via `joue Garde (1) sur player_1, devine Comtesse (8)`.
- Partie 20: Fulgurance: Baron gagne contre player_1 via `joue Baron (3) sur player_1`.
- Partie 21: Fulgurance: Garde touche player_1 (Servante (4)) via `joue Garde (1) sur player_1, devine Servante (4)`.

## Distribution des cartes jouees par le modele

- Espionne (0): 6
- Garde (1): 32
- Prêtre (2): 10
- Baron (3): 6
- Servante (4): 10
- Prince (5): 1
- Chancelier (6): 7
- Roi (7): 3
- ChancellorResolution: 7

## Recommandations

1. Utiliser `curriculum_phase1.pth` comme champion par defaut dans l'app pour l'instant, ou comparer sur 1000 parties avant de trancher definitivement.
2. Ajouter un script d'evaluation canonique qui teste tous les checkpoints sur les memes seeds, 0H/1H/2H/3H, et produit un tableau automatiquement.
3. Revoir le curriculum: phase2/phase3 devraient etre acceptes seulement s'ils battent phase1 sur un set de validation fixe.
4. Ajouter des tests de comportement ciblés: Garde avec info certaine, Baron avec carte faible, Prince sur Princesse connue, Roi avec carte forte.
5. Pour progresser contre heuristique, entrainer explicitement contre plusieurs variantes heuristiques plutot qu'une seule politique fixe.

## Niveau de confiance

Moyen. Les tendances sont coherentes sur plusieurs runs de 100 parties, mais Love Letter reste variance-heavy. Pour figer un champion, je recommande 1000 parties par config avec seeds fixes.
