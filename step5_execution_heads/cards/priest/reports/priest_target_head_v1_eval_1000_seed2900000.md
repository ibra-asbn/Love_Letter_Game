# Step5 - Evaluation Tete Pretre

Date: 2026-04-26 16:37:42 CEST.

Kind: `priest_target`
Checkpoint: `step5_execution_heads/cards/priest/checkpoints/priest_target_head_v1.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 52.40% | 44.00% | 37.60% | 32.70% | 0.38400 |
| Pretre random | 52.10% | 43.00% | 37.70% | 32.40% | 0.38080 |
| Step3 + tete Pretre | 52.30% | 44.40% | 37.50% | 33.00% | 0.38560 |

Delta tete vs Step3: `+0.00160`.
Delta random vs Step3: `-0.00320`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 60.06% (n=343) | 48.76% (n=402) | 44.71% (n=340) | 41.64% (n=353) | 48.75% (n=1438) |
| Pretre random | 59.18% (n=343) | 46.27% (n=402) | 45.00% (n=340) | 40.79% (n=353) | 47.71% (n=1438) |
| Step3 + tete Pretre | 59.77% (n=343) | 49.75% (n=402) | 44.41% (n=340) | 42.49% (n=353) | 49.10% (n=1438) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 53.42% (n=234) | 44.37% (n=293) | 40.52% (n=269) | 37.26% (n=263) | 43.63% (n=1059) |
| Pretre random | 52.14% (n=234) | 40.96% (n=293) | 40.89% (n=269) | 36.12% (n=263) | 42.21% (n=1059) |
| Step3 + tete Pretre | 52.99% (n=234) | 45.73% (n=293) | 40.15% (n=269) | 38.40% (n=263) | 44.10% (n=1059) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 1111 | 0 | 0.00% | 0.0000 |
| Pretre random | 1116 | 639 | 57.26% | 0.0000 |
| Step3 + tete Pretre | 1111 | 226 | 20.34% | 0.0288 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 41.68% | 18.27% | 16.20% | 11.22% | 12.62% |
| Pretre random | 41.30% | 18.65% | 16.55% | 10.78% | 12.72% |
| Step3 + tete Pretre | 41.80% | 18.25% | 16.15% | 11.03% | 12.78% |
