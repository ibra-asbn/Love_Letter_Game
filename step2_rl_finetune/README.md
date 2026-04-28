# Step 2 - RL Fine-Tuning

Objectif: partir du student heuristique de l'etape 1 et lui apprendre a depasser `HeuristicBot` par reward reel, sans oublier les fondamentaux acquis par imitation.

Checkpoint de depart:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

## Idee

L'etape 1 a donne un excellent imitateur. L'etape 2 doit le transformer en joueur meilleur que son professeur.

La boucle d'entrainement:

- collecte des parties contre plusieurs compositions random/heuristic;
- PPO sur le reward reel;
- imitation de `HeuristicBot` comme garde-fou au debut;
- coefficient imitation qui decroit progressivement;
- KL anchor vers le checkpoint de depart au debut, puis relache;
- selection du meilleur checkpoint par score composite arena.

## Scripts

```text
common.py            # chemins, logs, score composite
evaluate_step2.py    # baseline et evaluation des candidats
train_step2_ppo.py   # PPO depuis le student heuristique
```

Les artefacts locaux restent dans:

```text
checkpoints/
logs/
reports/
```

## Baseline

```bash
python3 -m step2_rl_finetune.evaluate_step2 \
  --checkpoint heuristic_student_attempt4_player0_chancellor_order.pth \
  --games 3000 \
  --seed-start 700000 \
  --output step2_baseline_student_3000.json \
  --run-log step2_rl_finetune/logs/2026-04-24_step2_baseline_student_3000.md
```

## Entrainement

```bash
python3 -m step2_rl_finetune.train_step2_ppo \
  --start heuristic_student_attempt4_player0_chancellor_order.pth \
  --output-prefix step2_ppo_attempt1
```

Le premier PPO a ete utile comme test mais pas comme succes final: il a eu un succes court sur `600` parties, puis la confirmation longue est retombee sous `HeuristicBot`. Le vrai succes de cette etape vient de la distillation du retarget belief:

```bash
python3 scripts/training/distill_belief_retarget.py \
  --start step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth \
  --output step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth \
  --games 7000 \
  --epochs 10
```

## Resultat Actuel

Checkpoint retenu:

```text
checkpoints/step2_retarget_distilled_attempt1.pth
```

Pourquoi ce chemin a marche:

- le student heuristique brut copiait `HeuristicBot`;
- son belief contenait deja un signal utile;
- le mode contre-factuel `retarget` montait fortement le score;
- on a donc distille les actions retarget dans l'actor.

Baseline step2 sur `3000` parties par configuration:

| Modele player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student etape 1 | 45.47% | 33.83% | 20.70% | 12.37% | 0.22470 |
| HeuristicBot | 45.77% | 34.07% | 20.90% | 12.43% | 0.22633 |

Confirmation longue du checkpoint step2 sur `5000` parties par configuration:

| Modele player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.54% | 36.80% | 26.36% | 16.04% | 0.26738 |
| HeuristicBot | 46.02% | 33.48% | 21.36% | 11.30% | 0.22226 |

Gain confirme:

- `+4.51` points de composite vs `HeuristicBot`;
- `+4.52` points vs 3 randoms;
- `+3.32` points vs 1H+2R;
- `+5.00` points vs 2H+1R;
- `+4.74` points vs 3H.

Diagnostic actor/belief apres distillation sur `1000` parties:

| Mode | Composite |
|---|---:|
| Actor brut distille | 0.2634 |
| Retarget encore force | 0.2695 |
| Tactical force | 0.2777 |

Lecture: l'actor a internalise l'essentiel du retarget. Il reste encore du potentiel, surtout tactique, mais l'etape 2 est un succes valide.

## Critere De Succes

Un run court est un succes provisoire si le score composite bat:

- le student de depart;
- `HeuristicBot`;
- avec une petite marge configuree dans `train_step2_ppo.py`.

Un vrai succes demande ensuite une confirmation longue, idealement `5000` parties par configuration.

Statut: atteint avec `step2_retarget_distilled_attempt1.pth`.

