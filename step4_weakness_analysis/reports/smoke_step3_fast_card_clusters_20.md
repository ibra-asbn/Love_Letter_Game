# Step4 - Analyse Des Faiblesses Par Cartes

Date: 2026-04-25 16:05:30 CEST.

Modele analyse: `Step3 rapide DAgger`.

Checkpoint: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth`.

Taxonomie: `step4_weakness_analysis/CARD_TAXONOMY.md`.

## Resultat Global

| Games | Composite | Winrate moyen | Reward moyen |
|---:|---:|---:|---:|
| 80 | 0.39000 | 43.75% | 0.5875 |

## Arena Fair Seat-Rotated

| Composition | Games | Winrate | Reward moyen | 1er sorti parmi pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|---:|
| vs 3 randoms | 20 | 45.00% | 0.7200 | 54.55% | 9.09% | 27.27% | 9.09% |
| vs 1H+2R | 20 | 65.00% | 0.7800 | 71.43% | 28.57% | 0.00% | 0.00% |
| vs 2H+1R | 20 | 45.00% | 0.5900 | 9.09% | 36.36% | 18.18% | 36.36% |
| vs 3H | 20 | 20.00% | 0.2600 | 31.25% | 50.00% | 6.25% | 12.50% |

## Positions De Defaite Globales

Les pourcentages ci-dessous sont conditionnels aux defaites: leur somme vaut 100%.

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 37.78% | 33.33% | 13.33% | 15.56% |

## Archetypes De Parties

| Archetype | Games | Winrate | 1er sorti / pertes | 2e | 3e | Finaliste perdant |
|---|---:|---:|---:|---:|---:|---:|
| Main riche en information | 53 | 50.94% | 11.54% | 38.46% | 23.08% | 26.92% |
| Beaucoup de cartes pieges | 42 | 59.52% | 0.00% | 35.29% | 23.53% | 41.18% |
| Beaucoup de Gardes | 33 | 66.67% | 27.27% | 9.09% | 27.27% | 36.36% |
| Partie avec Servante | 33 | 54.55% | 20.00% | 40.00% | 13.33% | 26.67% |
| Controle en fin de manche | 32 | 53.12% | 0.00% | 40.00% | 26.67% | 33.33% |
| Main riche en controle | 31 | 54.84% | 0.00% | 35.71% | 35.71% | 28.57% |
| Partie avec Prince | 31 | 58.06% | 30.77% | 23.08% | 23.08% | 23.08% |
| Grosse carte tot | 20 | 60.00% | 12.50% | 50.00% | 12.50% | 25.00% |
| Partie avec Espionne | 20 | 40.00% | 41.67% | 33.33% | 16.67% | 8.33% |
| Pression Baron | 19 | 26.32% | 7.14% | 50.00% | 14.29% | 28.57% |
| Baron avec petite carte | 19 | 36.84% | 25.00% | 25.00% | 8.33% | 41.67% |
| Partie avec Chancelier | 17 | 47.06% | 0.00% | 33.33% | 33.33% | 33.33% |
| Pretre puis Garde | 14 | 64.29% | 40.00% | 0.00% | 40.00% | 20.00% |
| Partie avec Roi | 10 | 70.00% | 0.00% | 33.33% | 33.33% | 33.33% |
| Comtesse tot | 9 | 44.44% | 0.00% | 60.00% | 0.00% | 40.00% |
| Princesse tot | 8 | 50.00% | 25.00% | 50.00% | 25.00% | 0.00% |
| Comtesse volontaire | 3 | 33.33% | 0.00% | 100.00% | 0.00% | 0.00% |

## Archetypes A Surveiller

Filtre: au moins 100 parties.

| Archetype | Games | Winrate | Lecture rapide |
|---|---:|---:|---|

## Familles De Cartes Par Phase

Chaque case indique: `presence dans les parties / winrate de ces parties`.

| Famille | Cartes | Early | Mid | Late |
|---|---|---:|---:|---:|
| Information active | Garde, Pretre, Baron, Chancelier, Roi | 80.00% / 37.50% | 81.25% / 47.69% | 56.25% / 57.78% |
| Hypothese / ciblage | Garde, Baron, Prince, Roi | 70.00% / 41.07% | 75.00% / 53.33% | 56.25% / 57.78% |
| Pression elimination | Garde, Baron, Prince, Princesse | 73.75% / 42.37% | 75.00% / 55.00% | 55.00% / 56.82% |
| Controle main/pioche | Prince, Chancelier, Roi, Comtesse | 52.50% / 45.24% | 53.75% / 48.84% | 40.00% / 53.12% |
| Tempo sur | Espionne, Servante, Comtesse | 47.50% / 50.00% | 35.00% / 50.00% | 32.50% / 46.15% |
| Valeur passive / contrainte | Espionne, Comtesse, Princesse | 40.00% / 50.00% | 37.50% / 56.67% | 32.50% / 57.69% |
| Risque fort | Baron, Prince, Roi, Comtesse, Princesse | 62.50% / 44.00% | 70.00% / 55.36% | 52.50% / 57.14% |
| Revelation publique | Garde, Baron, Prince, Princesse | 73.75% / 42.37% | 75.00% / 55.00% | 55.00% / 56.82% |

## Cartes Individuelles Par Phase

Chaque case indique: `presence dans les parties / winrate / coups joues`.

| Carte | Early | Mid | Late |
|---|---:|---:|---:|
| Espionne | 21.25% / 52.94% / 7 | 16.25% / 46.15% / 7 | 13.75% / 36.36% / 7 |
| Garde | 42.50% / 38.24% / 25 | 52.50% / 57.14% / 39 | 31.25% / 76.00% / 24 |
| Pretre | 31.25% / 40.00% / 17 | 21.25% / 35.29% / 15 | 21.25% / 58.82% / 11 |
| Baron | 18.75% / 20.00% / 7 | 20.00% / 43.75% / 5 | 21.25% / 29.41% / 10 |
| Servante | 20.00% / 50.00% / 13 | 16.25% / 53.85% / 11 | 16.25% / 69.23% / 11 |
| Prince | 30.00% / 45.83% / 15 | 23.75% / 57.89% / 12 | 18.75% / 60.00% / 8 |
| Chancelier | 10.00% / 12.50% / 2 | 16.25% / 23.08% / 8 | 16.25% / 69.23% / 7 |
| Roi | 6.25% / 100.00% / 0 | 17.50% / 64.29% / 1 | 12.50% / 70.00% / 9 |
| Comtesse | 11.25% / 44.44% / 4 | 7.50% / 50.00% / 2 | 10.00% / 25.00% / 3 |
| Princesse | 10.00% / 50.00% / 0 | 16.25% / 69.23% / 0 | 20.00% / 62.50% / 0 |

## Notes

- Les clusters sont multi-label: une meme partie peut etre `Princesse tot` et `Controle main/pioche`.
- Les phases sont basees sur la taille de pioche au moment ou le modele agit.
- Ce rapport sert a trouver des hypotheses de faiblesse, pas a conclure seul. Les plus gros signaux devront etre retestes avec un seed independant.
