# Seat-Rotated Fair Tactical Arena

Date: 2026-04-25 14:59:37 CEST.

Le joueur evalue tourne entre `player_0`, `player_1`, `player_2`, `player_3`. Les heuristiques adverses utilisent `HeuristicBot(shuffle_targets=True)`.

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 37.50% | 25.00% | 31.25% | 50.00% | 0.38125 |
| Step2 retarget | 56.25% | 37.50% | 12.50% | 43.75% | 0.34375 |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 35.94% | 18.75% | 9.38% | 17.19% | 18.75% |
| Step2 retarget | 37.50% | 18.75% | 25.00% | 12.50% | 6.25% |

## Metriques Tactiques Globales

| Politique | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier pioche connue gagne | Espionne bonus / Espionne |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 23.08% | 83.33% | 60.00% | 69.57% | 30.43% | 50.00% | 42.86% |
| Step2 retarget | 37.93% | 90.00% | 83.33% | 76.19% | 23.81% | 100.00% | 46.67% |

## Note Interne - Comtesse Volontaire

| Politique | Comtesses volontaires | Winrate apres Comtesse volontaire |
|---|---:|---:|
| Fair HeuristicBot | 2 | 50.00% |
| Step2 retarget | 6 | 33.33% |
