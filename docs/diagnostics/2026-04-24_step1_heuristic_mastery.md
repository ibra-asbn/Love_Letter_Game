# Diagnostic Step 1 - Heuristic Mastery

Date: 2026-04-24, Europe/Paris.

## Objectif

Creer une premiere etape propre pour apprendre tout ce que le modele peut apprendre de `HeuristicBot`, sans overfitting evident, avant de repartir vers du RL ambitieux.

Le dossier cree pour cette etape est:

```text
step1_heuristic_mastery/
```

## Constat Initial

Le projet avait deja un champion historique (`curriculum_phase1.pth`) et plusieurs essais actor/belief. Le probleme principal observe n'etait pas seulement "le belief est mauvais": les tests contre-factuels montraient que le belief contenait un signal utile, mais que l'actor ne l'exploitait pas assez proprement.

Avant de chercher a battre l'heuristique avec PPO, il fallait verifier une chose plus fondamentale: est-ce qu'un reseau peut au moins apprendre l'heuristique de reference de facon fiable ?

## Ce Qui A Ete Fait

- Creation du sous-dossier `step1_heuristic_mastery/`.
- Ajout d'un collecteur de sequences supervisees par `HeuristicBot`.
- Ajout d'un entrainement recurrent avec split par parties entieres, validation, test et early stopping.
- Ajout d'un comparateur action par action entre student et teacher.
- Ajout d'une evaluation arena student vs randoms/heuristiques.
- Correction de l'observation du Chancelier dans `love_letter/engine.py`.

## Point Important Decouvert

La premiere intuition "collecter toutes les places de table" etait mauvaise telle quelle.

L'observation est relative au joueur courant, mais certaines actions et certains tie-breaks de l'heuristique dependent encore de l'identite absolue des joueurs. En collectant toutes les places sans signal de siege explicite, on fabrique des labels contradictoires pour un meme type d'observation.

Decision retenue pour cette etape:

- entrainer le student sur `player_0`, qui est le siege utilise par l'app et par nos evaluations principales;
- garder les autres sieges pour une etape future, soit avec un encodage de siege explicite, soit avec une heuristique rendue strictement permutation-invariant.

## Correction Chancelier

Avant correction, pendant la phase de choix du Chancelier, l'observation ne contenait pas correctement les cartes disponibles et leur ordre. Le reseau devait donc predire `900/902/904` avec une information partiellement absente.

Correction:

- le bloc de main expose le pool du Chancelier pendant cette phase;
- un bloc d'observation encode l'ordre exact des cartes du pool.

Effet mesure:

- ancien attempt player0: `ChancellorChoice` exact a 74.01%;
- attempt final apres correction: `ChancellorChoice` exact a 99.78%.

## Run Final

Dataset:

```text
step1_heuristic_mastery/data/teacher_sequences_attempt4_player0_chancellor_order.pkl
```

Checkpoint:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Rapports JSON locaux:

```text
step1_heuristic_mastery/reports/heuristic_student_attempt4_player0_chancellor_order_train.json
step1_heuristic_mastery/reports/heuristic_student_attempt4_player0_chancellor_order_action_compare.json
step1_heuristic_mastery/reports/heuristic_student_attempt4_player0_chancellor_order_eval_1000.json
step1_heuristic_mastery/reports/heuristic_student_attempt4_player0_chancellor_order_eval_5000_primary.json
```

Logs locaux:

```text
step1_heuristic_mastery/logs/2026-04-24_step1_collect_attempt4_player0_chancellor_order.md
step1_heuristic_mastery/logs/2026-04-24_step1_train_attempt4_player0_chancellor_order.md
step1_heuristic_mastery/logs/2026-04-24_step1_compare_attempt4_player0_chancellor_order.md
step1_heuristic_mastery/logs/2026-04-24_step1_eval_attempt4_player0_chancellor_order.md
step1_heuristic_mastery/logs/2026-04-24_step1_eval_attempt4_player0_chancellor_order_5000_primary.md
```

## Resultats Supervise

| Split | Action accuracy | Action loss | Belief accuracy |
|---|---:|---:|---:|
| Train | 99.92% | 0.0048 | 32.93% |
| Validation | 98.93% | 0.0313 | 31.43% |
| Test | 98.94% | 0.0292 | 30.62% |

Gap train-validation action accuracy: `0.99` point.

Lecture: le modele generalise tres bien sur des parties non vues. Il y a un petit gap, normal pour une imitation quasi parfaite, mais pas de sur-apprentissage bloquant.

## Comparaison Action Par Action

Sur `129 858` decisions:

| Mesure | Resultat |
|---|---:|
| Exact action accuracy | 99.72% |
| Meme carte jouee | 99.95% |
| Garde exact | 99.76% |
| Baron exact | 99.80% |
| Prince exact | 99.98% |
| Chancelier exact | 99.97% |
| Choix Chancelier exact | 99.78% |

Erreurs restantes:

- surtout des choix de cible avec le Roi;
- quelques devinettes de Garde;
- tres rarement une carte differente.

Ce sont principalement des tie-breaks ou des situations peu frequentes, pas une incomprehension massive des regles.

## Arena 1000 Parties

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student | 48.20% | 35.30% | 22.90% | 12.70% | 0.2383 |
| HeuristicBot | 48.60% | 35.20% | 22.90% | 12.40% | 0.2373 |

## Arena 5000 Parties

Test de reference, plus significatif:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student | 47.68% | 32.86% | 21.40% | 11.72% | 0.22448 |
| HeuristicBot | 47.86% | 32.78% | 21.62% | 11.82% | 0.22556 |

Ecart composite: `-0.00108` pour le student.

Lecture: l'ecart est trop petit pour conclure a une vraie inferiorite strategique. Le student est, en pratique, au niveau de l'heuristique qu'il imite.

## Verdict

Succes pour l'etape 1.

Le modele a appris quasiment tout ce que l'on pouvait raisonnablement extraire de l'heuristique dans le cadre player_0 actuel:

- imitation test a 98.94%;
- comparaison action par action a 99.72%;
- arena 5000 quasiment egale a `HeuristicBot`;
- pas de signal d'overfitting grave;
- correction d'un vrai bug d'observation sur le Chancelier.

Ce modele ne bat pas encore l'heuristique. C'est normal: il est entraine a la copier, pas a l'exploiter. Le prochain saut doit venir du RL/league training a partir de ce checkpoint, avec garde-fou pour ne pas oublier l'heuristique.

## Suite Recommandee

1. Repartir du checkpoint `heuristic_student_attempt4_player0_chancellor_order.pth`.
2. Lancer une phase RL courte avec regularisation imitation pour conserver les fondamentaux.
3. Evaluer chaque candidat contre randoms, heuristiques et champion historique.
4. Introduire ensuite le belief comme avantage tactique explicite, pas comme simple feature passive.
5. Ne declarer un nouveau champion que sur une evaluation longue, au moins `5000` parties par configuration.
