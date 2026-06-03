# Step5 - Evaluation Tete Baron

Date: 2026-04-26 14:17:19 CEST.

Kind: `baron_target`
Checkpoint: `step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt2_duel_features.pth`

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 56.40% | 41.70% | 38.80% | 33.00% | 0.38820 |
| Baron random | 55.30% | 42.10% | 39.10% | 32.40% | 0.38640 |
| Step3 + regle Baron | 56.30% | 42.00% | 39.20% | 33.10% | 0.39030 |

Delta tete vs Step3: `+0.00210`.
Delta random vs Step3: `-0.00180`.

## Winrates Conditionnels - Carte Jouee

Ces lignes ne gardent que les parties ou la politique evaluee a effectivement joue la carte cible au moins une fois.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 63.10% (n=355) | 49.84% (n=317) | 45.07% (n=304) | 36.88% (n=263) | 49.72% (n=1239) |
| Baron random | 60.00% (n=355) | 51.10% (n=317) | 46.05% (n=304) | 34.60% (n=263) | 48.91% (n=1239) |
| Step3 + regle Baron | 62.82% (n=355) | 50.79% (n=317) | 46.38% (n=304) | 37.26% (n=263) | 50.28% (n=1239) |

## Winrates Conditionnels - Cible Randomisable

Ces lignes gardent seulement les parties ou la carte cible a ete jouee avec au moins deux executions legales.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Winrate pondere |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 58.33% (n=240) | 45.61% (n=239) | 39.33% (n=239) | 32.20% (n=205) | 44.31% (n=923) |
| Baron random | 53.75% (n=240) | 47.28% (n=239) | 40.59% (n=239) | 29.27% (n=205) | 43.23% (n=923) |
| Step3 + regle Baron | 57.92% (n=240) | 46.86% (n=239) | 41.00% (n=239) | 32.68% (n=205) | 45.07% (n=923) |

## Interventions

| Politique | Checks | Overrides | Override rate | Mean predicted margin |
|---|---:|---:|---:|---:|
| Step3 rapide | 966 | 0 | 0.00% | 0.0000 |
| Baron random | 962 | 545 | 56.65% | 0.0000 |
| Step3 + regle Baron | 967 | 44 | 4.55% | 0.0434 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 42.48% | 19.18% | 15.57% | 10.88% | 11.90% |
| Baron random | 42.23% | 19.60% | 16.23% | 10.12% | 11.82% |
| Step3 + regle Baron | 42.65% | 19.15% | 15.38% | 10.93% | 11.90% |
