# Step4 - Analyse Des Faiblesses Par Cartes

Date: 2026-04-25 16:06:19 CEST.

Modele analyse: `Step3 rapide DAgger`.

Checkpoint: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth`.

Taxonomie: `step4_weakness_analysis/CARD_TAXONOMY.md`.

## Resultat Global

| Games | Composite | Winrate moyen | Reward moyen |
|---:|---:|---:|---:|
| 4000 | 0.37240 | 40.62% | 0.6148 |

## Arena Fair Seat-Rotated

| Composition | Games | Winrate | Reward moyen | 1er sorti parmi pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|---:|
| vs 3 randoms | 1000 | 51.50% | 0.7774 | 35.26% | 24.95% | 15.26% | 24.54% |
| vs 1H+2R | 1000 | 44.90% | 0.6742 | 27.95% | 24.86% | 24.86% | 22.32% |
| vs 2H+1R | 1000 | 33.30% | 0.4954 | 31.48% | 26.99% | 22.34% | 19.19% |
| vs 3H | 1000 | 32.80% | 0.5122 | 34.82% | 31.10% | 16.07% | 18.01% |

## Positions De Defaite Globales

Les pourcentages ci-dessous sont conditionnels aux defaites: leur somme vaut 100%.

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 32.38% | 27.24% | 19.71% | 20.67% |

## Archetypes De Parties

| Archetype | Games | Winrate | 1er sorti / pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|
| Main riche en information | 2576 | 51.94% | 8.80% | 25.44% | 28.43% | 37.32% |
| Beaucoup de cartes pieges | 2054 | 56.38% | 7.37% | 22.32% | 32.37% | 37.95% |
| Controle en fin de manche | 1694 | 55.37% | 7.28% | 19.58% | 26.98% | 46.16% |
| Main riche en controle | 1554 | 54.12% | 6.45% | 22.58% | 29.45% | 41.51% |
| Beaucoup de Gardes | 1535 | 56.09% | 8.46% | 24.63% | 29.67% | 37.24% |
| Partie avec Servante | 1503 | 53.09% | 10.21% | 23.26% | 25.53% | 40.99% |
| Pression Baron | 1290 | 48.99% | 18.24% | 29.03% | 32.83% | 19.91% |
| Partie avec Prince | 1249 | 51.40% | 14.83% | 22.73% | 26.36% | 36.08% |
| Grosse carte tot | 1196 | 48.08% | 19.32% | 32.21% | 23.51% | 24.96% |
| Partie avec Chancelier | 1130 | 48.32% | 12.84% | 23.29% | 26.03% | 37.84% |
| Partie avec Espionne | 1091 | 54.81% | 19.68% | 29.21% | 27.79% | 23.33% |
| Baron avec petite carte | 985 | 41.73% | 25.78% | 26.31% | 21.25% | 26.66% |
| Partie avec Roi | 489 | 42.74% | 12.50% | 28.93% | 28.21% | 30.36% |
| Pretre puis Garde | 476 | 64.50% | 7.10% | 23.08% | 34.32% | 35.50% |
| Princesse tot | 453 | 49.45% | 17.03% | 30.13% | 29.69% | 23.14% |
| Comtesse tot | 421 | 53.68% | 15.90% | 33.85% | 16.92% | 33.33% |
| Comtesse volontaire | 170 | 67.06% | 5.36% | 44.64% | 35.71% | 14.29% |

## Archetypes A Surveiller

Filtre: au moins 120 parties.

| Archetype | Games | Winrate | Lecture rapide |
|---|---:|---:|---|
| Baron avec petite carte | 985 | 41.73% | Candidat de faiblesse si confirme par seed independant. |
| Partie avec Roi | 489 | 42.74% | Candidat de faiblesse si confirme par seed independant. |
| Grosse carte tot | 1196 | 48.08% | Candidat de faiblesse si confirme par seed independant. |
| Partie avec Chancelier | 1130 | 48.32% | Candidat de faiblesse si confirme par seed independant. |
| Pression Baron | 1290 | 48.99% | Candidat de faiblesse si confirme par seed independant. |
| Princesse tot | 453 | 49.45% | Candidat de faiblesse si confirme par seed independant. |
| Partie avec Prince | 1249 | 51.40% | Candidat de faiblesse si confirme par seed independant. |
| Main riche en information | 2576 | 51.94% | Candidat de faiblesse si confirme par seed independant. |

## Familles De Cartes Par Phase

Chaque case indique: `presence dans les parties / winrate de ces parties`.

| Famille | Cartes | Early | Mid | Late |
|---|---|---:|---:|---:|
| Information active | Garde, Pretre, Baron, Chancelier, Roi | 84.12% / 41.52% | 73.72% / 48.02% | 57.33% / 55.43% |
| Hypothese / ciblage | Garde, Baron, Prince, Roi | 80.20% / 41.58% | 71.00% / 47.89% | 56.40% / 55.81% |
| Pression elimination | Garde, Baron, Prince, Princesse | 79.55% / 42.30% | 71.12% / 48.58% | 57.55% / 56.78% |
| Controle main/pioche | Prince, Chancelier, Roi, Comtesse | 51.20% / 43.95% | 51.70% / 48.89% | 42.35% / 55.37% |
| Tempo sur | Espionne, Servante, Comtesse | 46.23% / 46.67% | 42.73% / 52.14% | 35.33% / 59.66% |
| Valeur passive / contrainte | Espionne, Comtesse, Princesse | 37.85% / 46.90% | 40.98% / 53.51% | 39.57% / 63.11% |
| Risque fort | Baron, Prince, Roi, Comtesse, Princesse | 60.88% / 43.49% | 64.40% / 48.60% | 52.83% / 57.74% |
| Revelation publique | Garde, Baron, Prince, Princesse | 79.55% / 42.30% | 71.12% / 48.58% | 57.55% / 56.78% |

## Cartes Individuelles Par Phase

Chaque case indique: `presence dans les parties / winrate / coups joues`.

| Carte | Early | Mid | Late |
|---|---:|---:|---:|
| Espionne | 19.78% / 43.24% / 332 | 18.55% / 50.54% / 381 | 19.25% / 57.66% / 465 |
| Garde | 52.50% / 41.76% / 1549 | 43.50% / 50.34% / 1421 | 37.18% / 59.05% / 1390 |
| Pretre | 20.38% / 41.96% / 650 | 17.75% / 50.42% / 542 | 15.28% / 54.66% / 410 |
| Baron | 21.52% / 39.95% / 393 | 23.15% / 44.71% / 463 | 20.70% / 52.42% / 483 |
| Servante | 21.73% / 47.53% / 670 | 18.45% / 52.71% / 597 | 13.40% / 60.45% / 436 |
| Prince | 21.88% / 42.17% / 471 | 20.88% / 45.75% / 502 | 16.05% / 57.48% / 416 |
| Chancelier | 19.78% / 40.71% / 423 | 19.00% / 49.21% / 407 | 16.50% / 52.58% / 427 |
| Roi | 10.62% / 41.65% / 87 | 13.90% / 46.94% / 124 | 13.18% / 47.82% / 278 |
| Comtesse | 10.53% / 53.68% / 107 | 14.00% / 57.68% / 187 | 12.32% / 68.76% / 184 |
| Princesse | 11.33% / 49.45% / 0 | 16.40% / 54.88% / 0 | 19.95% / 70.05% / 0 |

## Notes

- Les clusters sont multi-label: une meme partie peut etre `Princesse tot` et `Controle main/pioche`.
- Les phases sont basees sur la taille de pioche au moment ou le modele agit.
- Ce rapport sert a trouver des hypotheses de faiblesse, pas a conclure seul. Les plus gros signaux devront etre retestes avec un seed independant.
