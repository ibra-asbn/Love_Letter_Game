# Step6 - Matchups Asymetriques

Date: 2026-04-26 17:44:15 CEST.

Parties par matchup: `5000`.

Evaluation uniquement: un singleton joue contre trois copies d'un autre profil. Le siege du singleton tourne a chaque manche.

## Synthese

| Matchup | Singleton | Score >=1 | Victoire manche | Reward | Opposants/copie | Score >=1 | Victoire manche | Reward |
|---|---|---:|---:|---:|---|---:|---:|---:|
| champion_vs_3_step3 | Champion CBP | 31.36% | 27.64% | 0.4830 | Step3 seul | 28.06% | 24.43% | 0.4347 |
| champion_vs_3_step2 | Champion CBP | 30.06% | 26.42% | 0.4743 | Step2 | 27.56% | 24.93% | 0.4320 |
| champion_vs_3_heuristic | Champion CBP | 37.38% | 33.14% | 0.5730 | Heuristique fair | 26.58% | 22.61% | 0.4005 |
| step3_vs_3_champions | Step3 seul | 27.64% | 23.42% | 0.4135 | Champion CBP | 29.95% | 25.97% | 0.4522 |
| step2_vs_3_champions | Step2 | 25.36% | 22.38% | 0.3892 | Champion CBP | 30.13% | 26.25% | 0.4642 |

## Details Par Matchup

### champion_vs_3_step3

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 5000 | 31.36% | +/- 1.29% | 27.64% | 12.26% | 0.4830 | 23.28% | 21.02% | 13.84% | 10.50% |
| Opposants/copie | Step3 seul | 15000 | 28.06% | +/- 0.72% | 24.43% | 10.83% | 0.4347 | 24.92% | 21.47% | 14.61% | 10.95% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 27.69% | 77.29% | 21.67% | 88.45% | 90.70% |
| Opposants/copie | 28.56% | 69.05% | 29.41% | 71.48% | 94.09% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 1451,
    "forced": 56,
    "overrides": 496,
    "sum_best_margin": 129
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 3368,
    "base_baron_plays": 1573,
    "overrides": 1580
  },
  "singleton_champion_prince": {
    "base_prince_plays": 1579,
    "overrides": 637,
    "prince_hand_checks": 2384
  }
}
```

### champion_vs_3_step2

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 5000 | 30.06% | +/- 1.27% | 26.42% | 12.74% | 0.4743 | 24.12% | 21.02% | 16.62% | 8.18% |
| Opposants/copie | Step2 | 15000 | 27.56% | +/- 0.72% | 24.93% | 9.23% | 0.4320 | 24.70% | 22.21% | 15.91% | 9.62% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 28.21% | 77.81% | 21.50% | 89.17% | 96.05% |
| Opposants/copie | 33.66% | 70.99% | 27.37% | 74.42% | 93.35% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 1396,
    "forced": 54,
    "overrides": 458,
    "sum_best_margin": 123
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 3196,
    "base_baron_plays": 1438,
    "overrides": 1489
  },
  "singleton_champion_prince": {
    "base_prince_plays": 1485,
    "overrides": 598,
    "prince_hand_checks": 2314
  }
}
```

### champion_vs_3_heuristic

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 5000 | 37.38% | +/- 1.34% | 33.14% | 14.80% | 0.5730 | 21.42% | 18.72% | 10.42% | 12.06% |
| Opposants/copie | Heuristique fair | 15000 | 26.58% | +/- 0.71% | 22.61% | 10.34% | 0.4005 | 25.29% | 21.06% | 13.79% | 13.27% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 30.63% | 78.15% | 20.83% | 87.41% | 93.91% |
| Opposants/copie | 25.00% | 67.34% | 31.11% | 75.19% | 92.95% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 1546,
    "forced": 59,
    "overrides": 529,
    "sum_best_margin": 135
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 3371,
    "base_baron_plays": 1505,
    "overrides": 1606
  },
  "singleton_champion_prince": {
    "base_prince_plays": 1618,
    "overrides": 670,
    "prince_hand_checks": 2564
  }
}
```

### step3_vs_3_champions

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Step3 seul | 5000 | 27.64% | +/- 1.24% | 23.42% | 10.20% | 0.4135 | 23.90% | 20.86% | 13.22% | 14.38% |
| Opposants/copie | Champion CBP | 15000 | 29.95% | +/- 0.73% | 25.97% | 11.27% | 0.4522 | 24.15% | 19.37% | 12.85% | 13.68% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 28.88% | 63.47% | 34.59% | 71.29% | 95.70% |
| Opposants/copie | 26.61% | 76.58% | 22.35% | 89.07% | 95.46% |

Activite des tetes champion:

```json
{
  "opponent_champion_chancellor": {
    "checks": 4367,
    "forced": 237,
    "overrides": 1344,
    "sum_best_margin": 356
  },
  "opponent_champion_baron": {
    "baron_hand_checks": 10290,
    "base_baron_plays": 4476,
    "overrides": 4906
  },
  "opponent_champion_prince": {
    "base_prince_plays": 4692,
    "overrides": 1978,
    "prince_hand_checks": 7242
  }
}
```

### step2_vs_3_champions

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Step2 | 5000 | 25.36% | +/- 1.21% | 22.38% | 8.06% | 0.3892 | 24.44% | 21.92% | 15.38% | 12.90% |
| Opposants/copie | Champion CBP | 15000 | 30.13% | +/- 0.73% | 26.25% | 11.95% | 0.4642 | 23.87% | 19.73% | 13.51% | 12.75% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 32.44% | 67.46% | 31.38% | 75.92% | 95.24% |
| Opposants/copie | 27.86% | 76.08% | 22.53% | 89.16% | 95.57% |

Activite des tetes champion:

```json
{
  "opponent_champion_chancellor": {
    "checks": 4232,
    "forced": 250,
    "overrides": 1357,
    "sum_best_margin": 364
  },
  "opponent_champion_baron": {
    "baron_hand_checks": 10148,
    "base_baron_plays": 4413,
    "overrides": 4840
  },
  "opponent_champion_prince": {
    "base_prince_plays": 4686,
    "overrides": 1846,
    "prince_hand_checks": 7174
  }
}
```

## Lecture

Ces matchups ne remplacent pas l'arena de lignage complete. Ils mesurent plutot si un profil singleton tient quand les trois autres joueurs ont tous le meme niveau/style. C'est un bon test d'exploitabilite avant d'entrainer en self-play.