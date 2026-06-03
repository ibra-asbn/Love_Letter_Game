# Step5 - Evaluation Tete Baron

Date: 2026-04-26 14:26:49 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 52.12% | 43.28% | 38.34% | 32.90% | 0.38530 |
| Baron random | 51.58% | 42.76% | 37.90% | 32.64% | 0.38136 |
| Step3 + regle Baron | 52.28% | 43.32% | 38.30% | 32.98% | 0.38574 |

Delta tete vs Step3: `+0.00044`.
Delta random vs Step3: `-0.00394`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 59.14% (n=1691) | 51.39% (n=1582) | 45.75% (n=1506) | 38.09% (n=1457) | 49.02% (n=6236) |
| Baron random | 57.54% (n=1691) | 49.75% (n=1582) | 44.29% (n=1506) | 37.20% (n=1457) | 47.61% (n=6236) |
| Step3 + regle Baron | 59.61% (n=1691) | 51.52% (n=1582) | 45.62% (n=1506) | 38.37% (n=1457) | 49.21% (n=6236) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 53.07% (n=1221) | 46.49% (n=1181) | 40.48% (n=1124) | 34.06% (n=1145) | 43.72% (n=4671) |
| Baron random | 50.86% (n=1221) | 44.28% (n=1181) | 38.52% (n=1124) | 32.93% (n=1145) | 41.83% (n=4671) |
| Step3 + regle Baron | 53.73% (n=1221) | 46.66% (n=1181) | 40.30% (n=1124) | 34.41% (n=1145) | 43.97% (n=4671) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 4838 | 0 | 0.00% | 0.0000 |
| Baron random | 4830 | 2751 | 56.96% | 0.0000 |
| Step3 + regle Baron | 4837 | 216 | 4.47% | 0.0476 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 41.66% | 18.77% | 16.69% | 10.87% | 12.01% |
| Baron random | 41.22% | 19.00% | 17.16% | 10.75% | 11.88% |
| Step3 + regle Baron | 41.72% | 18.71% | 16.62% | 10.87% | 12.08% |
