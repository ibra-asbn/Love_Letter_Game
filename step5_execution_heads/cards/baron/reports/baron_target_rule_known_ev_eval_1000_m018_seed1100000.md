# Step5 - Evaluation Tete Baron

Date: 2026-04-26 14:31:08 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 53.10% | 42.60% | 38.60% | 33.90% | 0.38970 |
| Baron random | 53.60% | 42.70% | 38.20% | 32.50% | 0.38360 |
| Step3 + regle Baron | 53.20% | 42.70% | 38.50% | 33.80% | 0.38930 |

Delta tete vs Step3: `-0.00040`.
Delta random vs Step3: `-0.00610`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 56.75% (n=363) | 52.32% (n=323) | 44.44% (n=306) | 40.14% (n=284) | 48.98% (n=1276) |
| Baron random | 58.13% (n=363) | 52.63% (n=323) | 43.14% (n=306) | 35.21% (n=284) | 48.04% (n=1276) |
| Step3 + regle Baron | 57.02% (n=363) | 52.63% (n=323) | 44.12% (n=306) | 39.79% (n=284) | 48.98% (n=1276) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 49.23% (n=260) | 43.61% (n=227) | 36.57% (n=216) | 37.61% (n=218) | 42.13% (n=921) |
| Baron random | 51.15% (n=260) | 44.05% (n=227) | 34.72% (n=216) | 31.19% (n=218) | 40.83% (n=921) |
| Step3 + regle Baron | 49.62% (n=260) | 44.05% (n=227) | 36.11% (n=216) | 37.16% (n=218) | 42.13% (n=921) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 947 | 0 | 0.00% | 0.0000 |
| Baron random | 948 | 549 | 57.91% | 0.0000 |
| Step3 + regle Baron | 946 | 38 | 4.02% | 0.0444 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 42.05% | 18.80% | 16.00% | 11.10% | 12.05% |
| Baron random | 41.75% | 19.15% | 16.45% | 10.67% | 11.97% |
| Step3 + regle Baron | 42.05% | 18.73% | 16.02% | 11.10% | 12.10% |
