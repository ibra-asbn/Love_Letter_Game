# Step5 - Evaluation Tete Baron

Date: 2026-04-26 13:47:40 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt1.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.80% | 43.40% | 38.90% | 35.20% | 0.39610 |
| Baron random | 51.50% | 43.90% | 39.20% | 35.20% | 0.39770 |
| Step3 + tete Baron | 51.30% | 42.80% | 39.60% | 35.20% | 0.39650 |

Delta tete vs Step3: `+0.00040`.
Delta random vs Step3: `+0.00160`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 59.29% (n=339) | 48.25% (n=315) | 44.12% (n=272) | 42.22% (n=270) | 49.08% (n=1196) |
| Baron random | 58.41% (n=339) | 49.84% (n=315) | 45.22% (n=272) | 42.22% (n=270) | 49.50% (n=1196) |
| Step3 + tete Baron | 57.82% (n=339) | 46.35% (n=315) | 46.69% (n=272) | 42.22% (n=270) | 48.75% (n=1196) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 52.71% (n=258) | 41.42% (n=239) | 39.34% (n=211) | 38.03% (n=213) | 43.32% (n=921) |
| Baron random | 51.55% (n=258) | 43.51% (n=239) | 40.76% (n=211) | 38.03% (n=213) | 43.87% (n=921) |
| Step3 + tete Baron | 50.78% (n=258) | 38.91% (n=239) | 42.65% (n=211) | 38.03% (n=213) | 42.89% (n=921) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 959 | 0 | 0.00% | 0.0000 |
| Baron random | 950 | 578 | 60.84% | 0.0000 |
| Step3 + tete Baron | 956 | 119 | 12.45% | 0.0314 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 42.33% | 19.57% | 15.28% | 10.95% | 11.88% |
| Baron random | 42.45% | 19.53% | 15.60% | 10.22% | 12.20% |
| Step3 + tete Baron | 42.23% | 19.60% | 15.43% | 10.90% | 11.85% |
