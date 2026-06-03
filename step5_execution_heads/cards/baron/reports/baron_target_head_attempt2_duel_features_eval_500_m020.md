# Step5 - Evaluation Tete Baron

Date: 2026-04-26 13:59:06 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 54.00% | 43.80% | 37.80% | 30.80% | 0.37820 |
| Baron random | 53.60% | 44.00% | 37.60% | 31.60% | 0.38080 |
| Step3 + tete Baron | 53.80% | 43.80% | 38.20% | 30.80% | 0.37920 |

Delta tete vs Step3: `+0.00100`.
Delta random vs Step3: `+0.00260`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 58.70% (n=184) | 49.68% (n=155) | 48.95% (n=143) | 32.39% (n=142) | 48.24% (n=624) |
| Baron random | 57.61% (n=184) | 50.32% (n=155) | 48.25% (n=143) | 35.21% (n=142) | 48.56% (n=624) |
| Step3 + tete Baron | 58.15% (n=184) | 49.68% (n=155) | 50.35% (n=143) | 32.39% (n=142) | 48.40% (n=624) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.80% (n=139) | 39.64% (n=111) | 41.90% (n=105) | 30.58% (n=121) | 41.39% (n=476) |
| Baron random | 50.36% (n=139) | 40.54% (n=111) | 40.95% (n=105) | 33.88% (n=121) | 41.81% (n=476) |
| Step3 + tete Baron | 51.08% (n=139) | 39.64% (n=111) | 43.81% (n=105) | 30.58% (n=121) | 41.60% (n=476) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 498 | 0 | 0.00% | 0.0000 |
| Baron random | 494 | 284 | 57.49% | 0.0000 |
| Step3 + tete Baron | 498 | 31 | 6.22% | 0.0482 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 41.60% | 17.40% | 17.15% | 11.25% | 12.60% |
| Baron random | 41.70% | 18.15% | 17.00% | 10.95% | 12.20% |
| Step3 + tete Baron | 41.65% | 17.35% | 17.20% | 11.15% | 12.65% |
