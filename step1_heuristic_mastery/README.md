# Step 1 - Heuristic Mastery

Objectif: obtenir un modele qui a appris tout ce qu'on peut raisonnablement extraire de `HeuristicBot`, avant de refaire du RL.

Regle de cette etape:

- pas de nouveau module "fancy";
- pas de PPO tant que l'imitation de l'heuristique n'est pas propre;
- split par parties entieres pour eviter les fuites train/validation;
- early stopping sur validation;
- evaluation contre `HeuristicBot`, pas seulement contre randoms.

## Fichiers

```text
collect_teacher_sequences.py   # collecte des sequences labellisees par HeuristicBot
train_heuristic_student.py     # imitation recurrente avec validation/test et early stopping
compare_student_teacher.py     # comparaison action par action contre l'heuristique
evaluate_heuristic_mastery.py  # arena student vs heuristic/random + tables miroir
common.py                      # logs, chemins et helpers communs
```

Les artefacts de run restent locaux:

```text
data/
checkpoints/
logs/
reports/
```

## Pipeline

Collecte player_0 finale:

```bash
python3 -m step1_heuristic_mastery.collect_teacher_sequences \
  --games 40000 \
  --teacher-seat-mix all \
  --record-seats player0 \
  --output teacher_sequences_attempt4_player0_chancellor_order.pkl
```

Entrainement:

```bash
python3 -m step1_heuristic_mastery.train_heuristic_student \
  --dataset teacher_sequences_attempt4_player0_chancellor_order.pkl \
  --start champion_belief_retarget_distilled_attempt1.pth \
  --output heuristic_student_attempt4_player0_chancellor_order.pth
```

Comparaison action par action:

```bash
python3 -m step1_heuristic_mastery.compare_student_teacher \
  --checkpoint heuristic_student_attempt4_player0_chancellor_order.pth \
  --dataset teacher_sequences_attempt4_player0_chancellor_order.pkl
```

Evaluation arena:

```bash
python3 -m step1_heuristic_mastery.evaluate_heuristic_mastery \
  --checkpoint heuristic_student_attempt4_player0_chancellor_order.pth \
  --games 5000 \
  --skip-mirrors
```

## Resultat Actuel

Checkpoint:

```text
checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Training:

| Split | Action accuracy | Action loss |
|---|---:|---:|
| Train | 99.92% | 0.0048 |
| Validation | 98.93% | 0.0313 |
| Test | 98.94% | 0.0292 |

Comparaison action par action:

| Mesure | Resultat |
|---|---:|
| Exact action accuracy | 99.72% |
| Meme carte jouee | 99.95% |
| Choix Chancelier exact | 99.78% |

Arena player_0, `5000` parties par configuration:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student | 47.68% | 32.86% | 21.40% | 11.72% | 0.22448 |
| HeuristicBot | 47.86% | 32.78% | 21.62% | 11.82% | 0.22556 |

Verdict: l'etape est reussie. Le student est au niveau pratique de l'heuristique, avec un ecart composite de seulement `-0.00108` sur le test long. Il ne faut pas attendre qu'il la batte par imitation pure; le depassement doit venir de l'etape RL suivante.

## Critere De Reussite

Le modele est considere comme ayant absorbe l'heuristique si:

- l'action accuracy validation/test est stable et ne monte plus;
- le gap train-validation reste faible;
- les erreurs action-par-action sont comprises;
- le score arena est proche de `HeuristicBot`;
- le modele ne gagne pas seulement contre randoms, il tient aussi contre les tables avec heuristiques.

Statut: atteint pour `player_0`.

Limite connue: les resultats miroir sur d'autres sieges ne sont pas le critere principal de cette etape. Pour entrainer tous les sieges proprement, il faudra ajouter une information de siege ou rendre l'heuristique strictement permutation-invariant.
