# Seat-Rotated Fair Tactical Arena

Date: 2026-04-25 15:02:37 CEST.

Le joueur evalue tourne entre `player_0`, `player_1`, `player_2`, `player_3`. Les heuristiques adverses utilisent `HeuristicBot(shuffle_targets=True)`.

Definitions rapides:

- `Gagnant` utilise la meme definition que nos arenas historiques: reward final `>= 1.0`.
- `Finaliste perdant` signifie que le joueur n'a pas ete elimine par effet de carte, mais perd a la resolution finale.
- Les metriques tactiques ne mesurent que les decisions du joueur evalue, pas celles des adversaires.

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 49.80% | 42.90% | 32.00% | 32.70% | 0.36240 |
| Step2 retarget | 49.40% | 44.10% | 38.30% | 33.30% | 0.38570 |
| Step3 rapide DAgger | 54.40% | 43.00% | 38.30% | 36.70% | 0.40210 |
| Step3 hybride verify16 | 52.90% | 44.20% | 39.50% | 35.10% | 0.40020 |

## Winrates Par Siege En Full Heuristique

| Politique | player_0 | player_1 | player_2 | player_3 |
|---|---:|---:|---:|---:|
| Fair HeuristicBot | 37.60% | 30.00% | 31.20% | 32.00% |
| Step2 retarget | 36.00% | 30.40% | 32.00% | 34.80% |
| Step3 rapide DAgger | 40.00% | 34.00% | 39.60% | 33.20% |
| Step3 hybride verify16 | 37.20% | 32.00% | 37.20% | 34.00% |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 39.35% | 19.30% | 16.90% | 10.78% | 13.68% |
| Step2 retarget | 41.27% | 19.00% | 16.45% | 11.88% | 11.40% |
| Step3 rapide DAgger | 43.10% | 18.57% | 15.82% | 11.10% | 11.40% |
| Step3 hybride verify16 | 42.93% | 18.85% | 14.77% | 11.72% | 11.72% |

## Metriques Tactiques Globales

| Politique | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier pioche connue gagne | Espionne bonus / Espionne |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 25.68% | 92.54% | 90.69% | 73.42% | 24.07% | 54.70% | 60.10% |
| Step2 retarget | 35.62% | 91.15% | 88.94% | 74.12% | 23.92% | 58.45% | 53.94% |
| Step3 rapide DAgger | 30.65% | 86.00% | 92.24% | 74.46% | 23.38% | 61.63% | 53.52% |
| Step3 hybride verify16 | 35.19% | 91.13% | 93.16% | 76.08% | 21.41% | 59.73% | 56.86% |

Definitions tactiques:

- `Garde juste`: devinette exacte sur tous les Gardes joues.
- `Garde connu juste`: Garde exact quand la carte de la cible etait connue dans l'etat du joueur.
- `Pretre->Garde juste`: Garde exact sur une information obtenue precedemment par Pretre.
- `Baron gagne/perdu`: comparaison favorable/defavorable au moment de jouer Baron.
- `Chancelier pioche connue gagne`: proportion des pioches connues via Chancelier qui finissent dans une partie gagnee.
- `Espionne bonus / Espionne`: bonus Espionne obtenu par Espionne jouee.

## Note Interne - Comtesse Volontaire

| Politique | Comtesses volontaires | Winrate apres Comtesse volontaire |
|---|---:|---:|
| Fair HeuristicBot | 153 | 75.82% |
| Step2 retarget | 132 | 65.15% |
| Step3 rapide DAgger | 155 | 68.39% |
| Step3 hybride verify16 | 147 | 67.35% |
