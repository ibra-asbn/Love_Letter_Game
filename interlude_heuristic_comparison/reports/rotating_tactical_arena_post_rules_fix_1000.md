# Seat-Rotated Fair Tactical Arena

Date: 2026-04-25 16:05:19 CEST.

Le joueur evalue tourne entre `player_0`, `player_1`, `player_2`, `player_3`. Les heuristiques adverses utilisent `HeuristicBot(shuffle_targets=True)`.

Definitions rapides:

- `Gagnant` utilise la meme definition que nos arenas historiques: reward final `>= 1.0`.
- `Finaliste perdant` signifie que le joueur n'a pas ete elimine par effet de carte, mais perd a la resolution finale.
- Les metriques tactiques ne mesurent que les decisions du joueur evalue, pas celles des adversaires.

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 51.80% | 39.20% | 31.50% | 29.20% | 0.34150 |
| Step2 retarget | 51.70% | 43.00% | 35.70% | 33.30% | 0.37800 |
| Step3 rapide DAgger | 51.90% | 46.20% | 38.40% | 34.50% | 0.39750 |
| Step3 hybride verify16 | 51.20% | 45.30% | 39.30% | 34.30% | 0.39690 |

## Winrates Par Siege En Full Heuristique

| Politique | player_0 | player_1 | player_2 | player_3 |
|---|---:|---:|---:|---:|
| Fair HeuristicBot | 29.20% | 27.20% | 31.20% | 29.20% |
| Step2 retarget | 30.40% | 34.40% | 32.80% | 35.60% |
| Step3 rapide DAgger | 33.20% | 33.60% | 37.60% | 33.60% |
| Step3 hybride verify16 | 31.60% | 32.80% | 34.40% | 38.40% |

## Sorties Moyennes

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 37.92% | 20.60% | 16.80% | 10.90% | 13.78% |
| Step2 retarget | 40.92% | 19.62% | 15.25% | 12.53% | 11.68% |
| Step3 rapide DAgger | 42.75% | 18.90% | 15.97% | 10.78% | 11.60% |
| Step3 hybride verify16 | 42.52% | 19.20% | 15.60% | 11.25% | 11.43% |

## Metriques Tactiques Globales

| Politique | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier pioche connue gagne | Espionne bonus / Espionne |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 26.43% | 94.54% | 91.35% | 69.97% | 27.61% | 50.48% | 59.67% |
| Step2 retarget | 35.57% | 90.68% | 93.49% | 75.56% | 22.66% | 63.08% | 55.05% |
| Step3 rapide DAgger | 30.50% | 87.02% | 92.12% | 73.56% | 24.12% | 65.03% | 55.37% |
| Step3 hybride verify16 | 34.14% | 92.95% | 94.26% | 74.86% | 23.23% | 61.74% | 57.01% |

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
| Fair HeuristicBot | 154 | 72.73% |
| Step2 retarget | 140 | 67.14% |
| Step3 rapide DAgger | 158 | 72.78% |
| Step3 hybride verify16 | 171 | 72.51% |
