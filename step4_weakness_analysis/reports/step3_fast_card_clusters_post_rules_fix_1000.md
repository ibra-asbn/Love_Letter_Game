# Step4 - Analyse Des Faiblesses Par Cartes

Date: 2026-04-25 16:07:07 CEST.

Modele analyse: `Step3 rapide DAgger`.

Checkpoint: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth`.

Taxonomie: `step4_weakness_analysis/CARD_TAXONOMY.md`.

## Resultat Global

| Games | Composite | Winrate moyen | Reward moyen |
|---:|---:|---:|---:|
| 4000 | 0.39750 | 42.75% | 0.6447 |

## Arena Fair Seat-Rotated

| Composition | Games | Winrate | Reward moyen | 1er sorti parmi pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|---:|
| vs 3 randoms | 1000 | 51.90% | 0.7888 | 30.35% | 25.57% | 19.75% | 24.32% |
| vs 1H+2R | 1000 | 46.20% | 0.6790 | 30.67% | 28.81% | 19.89% | 20.63% |
| vs 2H+1R | 1000 | 38.40% | 0.5744 | 33.77% | 28.41% | 19.32% | 18.51% |
| vs 3H | 1000 | 34.50% | 0.5364 | 36.18% | 28.40% | 16.79% | 18.63% |

## Positions De Defaite Globales

Les pourcentages ci-dessous sont conditionnels aux defaites: leur somme vaut 100%.

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 33.01% | 27.90% | 18.82% | 20.26% |

## Archetypes De Parties

| Archetype | Games | Winrate | 1er sorti / pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|
| Main riche en information | 2562 | 55.78% | 9.44% | 25.15% | 26.83% | 38.57% |
| Beaucoup de cartes pieges | 2036 | 60.85% | 6.90% | 24.47% | 30.99% | 37.64% |
| Controle en fin de manche | 1717 | 57.37% | 5.74% | 20.63% | 27.05% | 46.58% |
| Main riche en controle | 1595 | 56.87% | 7.99% | 23.55% | 28.49% | 39.97% |
| Beaucoup de Gardes | 1546 | 60.54% | 7.87% | 19.34% | 31.48% | 41.31% |
| Partie avec Servante | 1482 | 56.21% | 11.25% | 23.73% | 26.96% | 38.06% |
| Pression Baron | 1297 | 52.58% | 17.40% | 33.33% | 28.29% | 20.98% |
| Partie avec Prince | 1249 | 52.84% | 12.39% | 27.50% | 23.43% | 36.67% |
| Grosse carte tot | 1194 | 49.75% | 21.00% | 30.33% | 25.33% | 23.33% |
| Partie avec Chancelier | 1142 | 51.14% | 17.56% | 23.66% | 22.76% | 36.02% |
| Partie avec Espionne | 1103 | 58.11% | 24.03% | 28.57% | 27.06% | 20.35% |
| Baron avec petite carte | 1010 | 47.13% | 26.97% | 28.09% | 20.04% | 24.91% |
| Pretre puis Garde | 504 | 66.47% | 5.92% | 20.71% | 33.73% | 39.64% |
| Partie avec Roi | 464 | 44.61% | 10.51% | 28.79% | 29.57% | 31.13% |
| Princesse tot | 445 | 48.76% | 21.05% | 31.58% | 27.19% | 20.18% |
| Comtesse tot | 405 | 55.56% | 17.78% | 22.78% | 28.33% | 31.11% |
| Comtesse volontaire | 158 | 72.78% | 9.30% | 23.26% | 41.86% | 25.58% |

## Archetypes A Surveiller

Filtre: au moins 120 parties.

| Archetype | Games | Winrate | Lecture rapide |
|---|---:|---:|---|
| Partie avec Roi | 464 | 44.61% | Candidat de faiblesse si confirme par seed independant. |
| Baron avec petite carte | 1010 | 47.13% | Candidat de faiblesse si confirme par seed independant. |
| Princesse tot | 445 | 48.76% | Candidat de faiblesse si confirme par seed independant. |
| Grosse carte tot | 1194 | 49.75% | Candidat de faiblesse si confirme par seed independant. |
| Partie avec Chancelier | 1142 | 51.14% | Candidat de faiblesse si confirme par seed independant. |
| Pression Baron | 1297 | 52.58% | Candidat de faiblesse si confirme par seed independant. |
| Partie avec Prince | 1249 | 52.84% | Candidat de faiblesse si confirme par seed independant. |
| Comtesse tot | 405 | 55.56% | Candidat de faiblesse si confirme par seed independant. |

## Familles De Cartes Par Phase

Chaque case indique: `presence dans les parties / winrate de ces parties`.

| Famille | Cartes | Early | Mid | Late |
|---|---|---:|---:|---:|
| Information active | Garde, Pretre, Baron, Chancelier, Roi | 84.05% / 44.71% | 73.58% / 50.73% | 57.23% / 58.10% |
| Hypothese / ciblage | Garde, Baron, Prince, Roi | 79.92% / 44.67% | 71.67% / 50.51% | 56.43% / 58.22% |
| Pression elimination | Garde, Baron, Prince, Princesse | 79.62% / 45.02% | 71.12% / 51.56% | 57.75% / 59.09% |
| Controle main/pioche | Prince, Chancelier, Roi, Comtesse | 51.40% / 44.94% | 52.73% / 52.30% | 42.93% / 57.37% |
| Tempo sur | Espionne, Servante, Comtesse | 44.70% / 47.20% | 41.95% / 55.01% | 36.27% / 63.40% |
| Valeur passive / contrainte | Espionne, Comtesse, Princesse | 37.23% / 47.62% | 40.08% / 57.02% | 39.48% / 66.81% |
| Risque fort | Baron, Prince, Roi, Comtesse, Princesse | 61.30% / 46.04% | 62.90% / 52.98% | 52.92% / 60.27% |
| Revelation publique | Garde, Baron, Prince, Princesse | 79.62% / 45.02% | 71.12% / 51.56% | 57.75% / 59.09% |

## Cartes Individuelles Par Phase

Chaque case indique: `presence dans les parties / winrate / coups joues`.

| Carte | Early | Mid | Late |
|---|---:|---:|---:|
| Espionne | 19.78% / 44.25% / 348 | 18.73% / 52.34% / 391 | 19.15% / 62.66% / 453 |
| Garde | 52.02% / 45.84% / 1549 | 43.73% / 52.37% / 1476 | 37.10% / 61.59% / 1366 |
| Pretre | 20.75% / 44.70% / 683 | 16.70% / 52.54% / 511 | 14.92% / 56.62% / 405 |
| Baron | 21.85% / 44.51% / 382 | 23.20% / 50.22% / 497 | 19.85% / 53.65% / 474 |
| Servante | 20.42% / 47.25% / 623 | 17.60% / 54.97% / 566 | 14.27% / 64.10% / 469 |
| Prince | 21.93% / 42.87% / 499 | 19.85% / 49.75% / 477 | 16.15% / 59.91% / 426 |
| Chancelier | 20.10% / 43.66% / 411 | 21.10% / 50.12% / 415 | 17.52% / 53.92% / 424 |
| Roi | 10.85% / 44.47% / 67 | 13.78% / 52.09% / 126 | 12.95% / 53.86% / 271 |
| Comtesse | 10.12% / 55.56% / 100 | 13.45% / 63.75% / 170 | 12.15% / 70.37% / 183 |
| Princesse | 11.12% / 48.76% / 0 | 15.28% / 58.92% / 0 | 19.98% / 73.09% / 0 |

## Notes

- Les clusters sont multi-label: une meme partie peut etre `Princesse tot` et `Controle main/pioche`.
- Les phases sont basees sur la taille de pioche au moment ou le modele agit.
- Ce rapport sert a trouver des hypotheses de faiblesse, pas a conclure seul. Les plus gros signaux devront etre retestes avec un seed independant.
