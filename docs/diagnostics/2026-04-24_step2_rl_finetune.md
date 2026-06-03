# Diagnostic Step 2 - RL Fine-Tuning / Belief Retarget

Date: 2026-04-24, Europe/Paris.

## Objectif

Demarrer une deuxieme etape dans un dossier separe pour passer du student qui copie `HeuristicBot` a un modele qui bat vraiment cette heuristique.

Sous-dossier:

```text
step2_rl_finetune/
```

## Point De Depart

Checkpoint de depart:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Baseline step2 sur `3000` parties par configuration:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student etape 1 | 45.47% | 33.83% | 20.70% | 12.37% | 0.22470 |
| HeuristicBot | 45.77% | 34.07% | 20.90% | 12.43% | 0.22633 |

Lecture: le student etape 1 est bien colle au teacher, mais ne le bat pas encore.

## Plan Initial

Plan annonce:

- creer une pipeline separee;
- evaluer le point de depart;
- lancer un fine-tuning PPO depuis le student;
- garder une contrainte imitation au debut;
- reduire cette contrainte pour autoriser le modele a depasser le teacher;
- documenter les resultats avant/apres.

Attendu: le PPO devait produire un checkpoint qui depasse le student de depart et `HeuristicBot` au score composite.

## Tentative PPO

Script:

```text
step2_rl_finetune/train_step2_ppo.py
```

Run:

```text
step2_rl_finetune/logs/2026-04-24_step2_ppo_attempt1.md
step2_rl_finetune/reports/step2_ppo_attempt1_train.json
```

Resultat:

- succes court a `600` parties: best `u6`, composite `0.23083`;
- seuil court atteint contre le baseline interne;
- confirmation longue a `5000` parties: composite `0.22586` contre `0.22800` pour `HeuristicBot`;
- donc pas un succes final.

Conclusion PPO: utile pour tester l'infrastructure, mais trop faible / trop bruité pour etre retenu comme checkpoint champion. Il a tendance a rester tres proche de l'heuristique et les petits gains courts ne survivent pas a l'evaluation longue.

## Diagnostic Belief Retarget

On a ensuite teste si le belief du student etape 1 contenait un signal exploitable.

Run:

```text
step2_rl_finetune/logs/2026-04-24_step2_student_belief_counterfactual_1000.md
step2_rl_finetune/reports/step2_student_belief_counterfactual_1000.json
```

Resultat sur `1000` parties par configuration:

| Mode | Composite |
|---|---:|
| Actor brut | 0.2225 |
| Retarget belief force | 0.2672 |
| Tactical force | 0.2523 |

Details importants:

- le retarget change `29.7%` des decisions;
- sur les Gardes, le choix du top belief passe de `36.1%` a `100%`;
- le hit rate des Gardes passe de `26.8%` a `34.4%`;
- le gain composite est `+4.47` points.

Conclusion: le probleme principal etait bien que l'actor n'utilisait pas assez son belief pour les cibles/devinettes.

## Distillation Retarget

Run:

```text
step2_rl_finetune/logs/2026-04-24_step2_retarget_distillation_attempt1.md
step2_rl_finetune/reports/step2_retarget_distillation_attempt1_train.json
```

Checkpoint produit:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Dataset:

- `7000` parties;
- `22335` decisions;
- `6855` decisions corrigees par retarget;
- taux de correction: `30.69%`.

Corrections principales:

| Type | Nombre |
|---|---:|
| Garde | 4437 |
| Prince | 1324 |
| Baron | 771 |
| Roi | 307 |
| Chancelier visible | 16 |

Validation finale:

- target accuracy globale: `90.77%`;
- accuracy sur les cas changes: `82.52%`;
- le modele ne copie plus aveuglement l'heuristique sur les cibles importantes.

## Confirmation Longue

Run:

```text
step2_rl_finetune/logs/2026-04-24_step2_retarget_distilled_attempt1_eval_5000.md
step2_rl_finetune/reports/step2_retarget_distilled_attempt1_eval_5000.json
```

Evaluation sur `5000` parties par configuration:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.54% | 36.80% | 26.36% | 16.04% | 0.26738 |
| HeuristicBot | 46.02% | 33.48% | 21.36% | 11.30% | 0.22226 |

Gain:

- composite: `+0.04512`;
- vs 3 randoms: `+4.52` points;
- vs 1H+2R: `+3.32` points;
- vs 2H+1R: `+5.00` points;
- vs 3H: `+4.74` points.

Verdict: succes confirme.

## Diagnostic Post-Distillation

Run:

```text
step2_rl_finetune/logs/2026-04-24_step2_retarget_distilled_attempt1_counterfactual_1000.md
step2_rl_finetune/reports/step2_retarget_distilled_attempt1_counterfactual_1000.json
```

Sur `1000` parties:

| Mode | Composite |
|---|---:|
| Actor brut distille | 0.2634 |
| Retarget encore force | 0.2695 |
| Tactical force | 0.2777 |

Lecture:

- l'actor a internalise la majeure partie du retarget;
- il reste un petit ecart retarget de `+0.0061`;
- il reste un potentiel tactique plus large de `+0.0143`;
- prochaine etape: distiller ou apprendre proprement le mode tactical, puis relancer du RL depuis ce checkpoint plus fort.

## Conclusion

L'etape 2 est un succes.

On a maintenant un modele DL qui bat `HeuristicBot` de maniere confirmee sur une evaluation longue. Le succes n'est pas venu du PPO seul, mais du diagnostic correct: le belief savait des choses utiles, l'actor ne les utilisait pas assez. La distillation retarget a transforme ce signal en comportement actor brut.

Checkpoint a conserver pour la suite:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Statut du projet:

- etape 1: apprendre l'heuristique, reussie;
- etape 2: battre l'heuristique via actor qui utilise mieux le belief, reussie;
- etape suivante: exploiter le potentiel tactical restant et comparer au champion historique `curriculum_phase1.pth`.

