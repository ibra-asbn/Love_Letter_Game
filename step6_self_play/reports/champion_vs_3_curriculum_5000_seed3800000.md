# Step6 - Matchups Asymetriques

Date: 2026-04-26 17:48:49 CEST.

Parties par matchup: `5000`.

Evaluation uniquement: un singleton joue contre trois copies d'un autre profil. Le siege du singleton tourne a chaque manche.

## Synthese

| Matchup | Singleton | Score >=1 | Victoire manche | Reward | Opposants/copie | Score >=1 | Victoire manche | Reward |
|---|---|---:|---:|---:|---|---:|---:|---:|
| champion_vs_3_curriculum | Champion CBP | 28.62% | 25.10% | 0.4367 | Curriculum phase1 | 29.14% | 25.31% | 0.4572 |

## Details Par Matchup

### champion_vs_3_curriculum

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 5000 | 28.62% | +/- 1.25% | 25.10% | 10.32% | 0.4367 | 23.94% | 19.88% | 14.00% | 13.56% |
| Opposants/copie | Curriculum phase1 | 15000 | 29.14% | +/- 0.73% | 25.31% | 12.30% | 0.4572 | 24.39% | 21.05% | 14.10% | 11.31% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 26.82% | 76.86% | 21.25% | 89.61% | 99.04% |
| Opposants/copie | 25.68% | 70.73% | 26.72% | 68.63% | 99.01% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 1567,
    "forced": 79,
    "overrides": 495,
    "sum_best_margin": 129
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 3535,
    "base_baron_plays": 1645,
    "overrides": 1675
  },
  "singleton_champion_prince": {
    "base_prince_plays": 1593,
    "overrides": 716,
    "prince_hand_checks": 2507
  }
}
```

## Lecture

Ces matchups ne remplacent pas l'arena de lignage complete. Ils mesurent plutot si un profil singleton tient quand les trois autres joueurs ont tous le meme niveau/style. C'est un bon test d'exploitabilite avant d'entrainer en self-play.