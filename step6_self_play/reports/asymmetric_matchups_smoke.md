# Step6 - Matchups Asymetriques

Date: 2026-04-26 17:35:55 CEST.

Parties par matchup: `24`.

Evaluation uniquement: un singleton joue contre trois copies d'un autre profil. Le siege du singleton tourne a chaque manche.

## Synthese

| Matchup | Singleton | Score >=1 | Victoire manche | Reward | Opposants/copie | Score >=1 | Victoire manche | Reward |
|---|---|---:|---:|---:|---|---:|---:|---:|
| champion_vs_3_step3 | Champion CBP | 37.50% | 37.50% | 0.5750 | Step3 seul | 27.78% | 20.83% | 0.3833 |
| champion_vs_3_step2 | Champion CBP | 45.83% | 41.67% | 0.6833 | Step2 | 25.00% | 19.44% | 0.4111 |
| champion_vs_3_heuristic | Champion CBP | 33.33% | 29.17% | 0.4083 | Heuristique fair | 27.78% | 23.61% | 0.3889 |
| step3_vs_3_champions | Step3 seul | 41.67% | 37.50% | 0.6750 | Champion CBP | 27.78% | 20.83% | 0.3917 |
| step2_vs_3_champions | Step2 | 20.83% | 20.83% | 0.3167 | Champion CBP | 26.39% | 26.39% | 0.4667 |

## Details Par Matchup

### champion_vs_3_step3

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 24 | 37.50% | +/- 19.37% | 37.50% | 12.50% | 0.5750 | 4.17% | 29.17% | 20.83% | 8.33% |
| Opposants/copie | Step3 seul | 72 | 27.78% | +/- 10.35% | 20.83% | 11.11% | 0.3833 | 30.56% | 20.83% | 9.72% | 11.11% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 24.14% | 66.67% | 33.33% | 100.00% | 100.00% |
| Opposants/copie | 28.36% | 57.14% | 38.10% | 70.83% | 66.67% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 6,
    "forced": 1,
    "overrides": 1,
    "sum_best_margin": 0
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 22,
    "base_baron_plays": 7,
    "overrides": 8
  },
  "singleton_champion_prince": {
    "base_prince_plays": 9,
    "overrides": 5,
    "prince_hand_checks": 11
  }
}
```

### champion_vs_3_step2

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 24 | 45.83% | +/- 19.93% | 41.67% | 16.67% | 0.6833 | 20.83% | 20.83% | 8.33% | 4.17% |
| Opposants/copie | Step2 | 72 | 25.00% | +/- 10.00% | 19.44% | 12.50% | 0.4111 | 26.39% | 26.39% | 19.44% | 2.78% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 35.71% | 66.67% | 33.33% | 80.00% | 0.00% |
| Opposants/copie | 28.17% | 76.00% | 24.00% | 64.29% | 0.00% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 5,
    "overrides": 4,
    "sum_best_margin": 0
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 20,
    "base_baron_plays": 7,
    "overrides": 10
  },
  "singleton_champion_prince": {
    "base_prince_plays": 8,
    "overrides": 4,
    "prince_hand_checks": 16
  }
}
```

### champion_vs_3_heuristic

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 24 | 33.33% | +/- 18.86% | 29.17% | 8.33% | 0.4083 | 29.17% | 33.33% | 0.00% | 4.17% |
| Opposants/copie | Heuristique fair | 72 | 27.78% | +/- 10.35% | 23.61% | 9.72% | 0.3889 | 23.61% | 22.22% | 16.67% | 9.72% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 21.74% | 42.86% | 57.14% | 85.71% | 0.00% |
| Opposants/copie | 25.68% | 52.00% | 48.00% | 77.78% | 100.00% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 7,
    "overrides": 3,
    "sum_best_margin": 0
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 18,
    "base_baron_plays": 9,
    "overrides": 11
  },
  "singleton_champion_prince": {
    "base_prince_plays": 7,
    "prince_hand_checks": 8
  }
}
```

### step3_vs_3_champions

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Step3 seul | 24 | 41.67% | +/- 19.72% | 37.50% | 20.83% | 0.6750 | 20.83% | 16.67% | 8.33% | 12.50% |
| Opposants/copie | Champion CBP | 72 | 27.78% | +/- 10.35% | 20.83% | 11.11% | 0.3917 | 26.39% | 19.44% | 11.11% | 15.28% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 29.63% | 71.43% | 28.57% | 66.67% | 100.00% |
| Opposants/copie | 20.90% | 80.00% | 20.00% | 91.67% | 100.00% |

Activite des tetes champion:

```json
{
  "opponent_champion_chancellor": {
    "checks": 22,
    "forced": 2,
    "overrides": 4,
    "sum_best_margin": 1
  },
  "opponent_champion_baron": {
    "baron_hand_checks": 46,
    "base_baron_plays": 22,
    "overrides": 19
  },
  "opponent_champion_prince": {
    "base_prince_plays": 21,
    "overrides": 12,
    "prince_hand_checks": 39
  }
}
```

### step2_vs_3_champions

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Step2 | 24 | 20.83% | +/- 16.25% | 20.83% | 4.17% | 0.3167 | 25.00% | 25.00% | 16.67% | 12.50% |
| Opposants/copie | Champion CBP | 72 | 26.39% | +/- 10.18% | 26.39% | 11.11% | 0.4667 | 20.83% | 18.06% | 18.06% | 16.67% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 35.29% | 75.00% | 25.00% | 83.33% | 100.00% |
| Opposants/copie | 28.57% | 78.95% | 21.05% | 86.96% | 100.00% |

Activite des tetes champion:

```json
{
  "opponent_champion_chancellor": {
    "checks": 21,
    "forced": 2,
    "overrides": 8,
    "sum_best_margin": 2
  },
  "opponent_champion_baron": {
    "baron_hand_checks": 50,
    "base_baron_plays": 25,
    "overrides": 27
  },
  "opponent_champion_prince": {
    "base_prince_plays": 28,
    "overrides": 7,
    "prince_hand_checks": 37
  }
}
```

## Lecture

Ces matchups ne remplacent pas l'arena de lignage complete. Ils mesurent plutot si un profil singleton tient quand les trois autres joueurs ont tous le meme niveau/style. C'est un bon test d'exploitabilite avant d'entrainer en self-play.