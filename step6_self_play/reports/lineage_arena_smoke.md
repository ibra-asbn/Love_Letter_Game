# Step6 - Arena De Lignage

Date: 2026-04-26 17:27:00 CEST.

Parties: `24`.

## Lecture

Evaluation uniquement: ce benchmark ne met pas encore a jour les modeles. Il sert a verifier si le champion courant domine son lignage direct avant de construire une ligue de self-play.

## Winrates

| Politique | Winrate | IC95 | Reward moyen | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---:|---:|---:|---:|---:|---:|---:|
| Champion CBP | 29.17% | +/- 18.18% | 0.4750 | 29.17% | 25.00% | 8.33% | 8.33% |
| Step3 seul | 29.17% | +/- 18.18% | 0.4667 | 25.00% | 16.67% | 20.83% | 8.33% |
| Step2 | 20.83% | +/- 16.25% | 0.3750 | 25.00% | 16.67% | 16.67% | 20.83% |
| Heuristique fair | 33.33% | +/- 18.86% | 0.4500 | 16.67% | 29.17% | 16.67% | 4.17% |

## Winrate Par Siege

| Politique | player_0 | player_1 | player_2 | player_3 |
|---|---:|---:|---:|---:|
| Champion CBP | 16.67% | 50.00% | 16.67% | 33.33% |
| Step3 seul | 16.67% | 16.67% | 33.33% | 50.00% |
| Step2 | 0.00% | 50.00% | 16.67% | 16.67% |
| Heuristique fair | 33.33% | 50.00% | 33.33% | 16.67% |

## Tactiques

| Politique | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Champion CBP | 28.57% | 60.00% | 40.00% | 80.00% | 0.00% |
| Step3 seul | 30.30% | 57.14% | 42.86% | 57.14% | 100.00% |
| Step2 | 23.81% | 72.73% | 27.27% | 87.50% | 100.00% |
| Heuristique fair | 30.00% | 71.43% | 28.57% | 100.00% | 100.00% |

## Activite Des Tetes Champion

```json
{
  "champion_chancellor": {
    "checks": 5,
    "overrides": 2,
    "sum_best_margin": 0
  },
  "champion_baron": {
    "baron_hand_checks": 9,
    "base_baron_plays": 5,
    "overrides": 5
  },
  "champion_prince": {
    "base_prince_plays": 8,
    "overrides": 2,
    "prince_hand_checks": 14
  }
}
```

## Conclusion Provisoire

Cette arena est le pont entre evaluation classique et self-play. Si le champion domine proprement cette table, la prochaine etape logique est de transformer ce lignage en population d'entrainement, avec conservation des anciens checkpoints pour eviter l'overfitting a un seul adversaire.