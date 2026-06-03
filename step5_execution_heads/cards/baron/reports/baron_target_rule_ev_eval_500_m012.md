# Step5 - Evaluation Tete Baron

Date: 2026-04-26 14:00:42 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 54.00% | 43.80% | 37.80% | 30.80% | 0.37820 |
| Baron random | 53.60% | 44.00% | 37.60% | 31.60% | 0.38080 |
| Step3 + regle Baron | 54.60% | 43.80% | 37.80% | 31.00% | 0.37960 |

Delta tete vs Step3: `+0.00140`.
Delta random vs Step3: `+0.00260`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 58.70% (n=184) | 49.68% (n=155) | 48.95% (n=143) | 32.39% (n=142) | 48.24% (n=624) |
| Baron random | 57.61% (n=184) | 50.32% (n=155) | 48.25% (n=143) | 35.21% (n=142) | 48.56% (n=624) |
| Step3 + regle Baron | 60.33% (n=184) | 49.68% (n=155) | 48.95% (n=143) | 33.10% (n=142) | 48.88% (n=624) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.80% (n=139) | 39.64% (n=111) | 41.90% (n=105) | 30.58% (n=121) | 41.39% (n=476) |
| Baron random | 50.36% (n=139) | 40.54% (n=111) | 40.95% (n=105) | 33.88% (n=121) | 41.81% (n=476) |
| Step3 + regle Baron | 53.96% (n=139) | 39.64% (n=111) | 41.90% (n=105) | 31.40% (n=121) | 42.23% (n=476) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 498 | 0 | 0.00% | 0.0000 |
| Baron random | 494 | 284 | 57.49% | 0.0000 |
| Step3 + regle Baron | 499 | 20 | 4.01% | 0.0281 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 41.60% | 17.40% | 17.15% | 11.25% | 12.60% |
| Baron random | 41.70% | 18.15% | 17.00% | 10.95% | 12.20% |
| Step3 + regle Baron | 41.80% | 17.35% | 16.90% | 11.35% | 12.60% |
