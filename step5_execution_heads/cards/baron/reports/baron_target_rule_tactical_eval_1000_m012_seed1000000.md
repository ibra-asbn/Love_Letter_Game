# Step5 - Evaluation Tete Baron

Date: 2026-04-26 14:12:01 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 50.50% | 42.30% | 39.70% | 35.10% | 0.39460 |
| Baron random | 50.90% | 41.50% | 38.30% | 34.40% | 0.38640 |
| Step3 + regle Baron | 51.10% | 41.60% | 40.00% | 34.80% | 0.39350 |

Delta tete vs Step3: `-0.00110`.
Delta random vs Step3: `-0.00820`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 58.97% (n=329) | 51.82% (n=330) | 46.44% (n=295) | 44.37% (n=311) | 50.59% (n=1265) |
| Baron random | 60.18% (n=329) | 49.39% (n=330) | 41.69% (n=295) | 42.12% (n=311) | 48.62% (n=1265) |
| Step3 + regle Baron | 60.79% (n=329) | 49.70% (n=330) | 47.46% (n=295) | 43.41% (n=311) | 50.51% (n=1265) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 54.39% (n=239) | 48.18% (n=247) | 41.15% (n=226) | 39.68% (n=252) | 45.85% (n=964) |
| Baron random | 56.07% (n=239) | 44.94% (n=247) | 34.96% (n=226) | 36.90% (n=252) | 43.26% (n=964) |
| Step3 + regle Baron | 56.90% (n=239) | 45.34% (n=247) | 42.48% (n=226) | 38.49% (n=252) | 45.75% (n=964) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 995 | 0 | 0.00% | 0.0000 |
| Baron random | 991 | 561 | 56.61% | 0.0000 |
| Step3 + regle Baron | 997 | 130 | 13.04% | 0.0592 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 41.90% | 19.18% | 16.43% | 10.90% | 11.60% |
| Baron random | 41.27% | 19.70% | 17.22% | 10.62% | 11.18% |
| Step3 + regle Baron | 41.88% | 19.30% | 16.38% | 10.93% | 11.53% |
