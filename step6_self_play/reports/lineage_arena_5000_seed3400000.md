# Step6 - Arena De Lignage

Date: 2026-04-26 17:28:31 CEST.

Parties: `5000`.

## Lecture

Evaluation uniquement: ce benchmark ne met pas encore a jour les modeles. Il sert a verifier si le champion courant domine son lignage direct avant de construire une ligue de self-play.

## Winrates

| Politique | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward moyen | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Champion CBP | 32.28% | +/- 1.30% | 28.36% | 13.08% | 0.5010 | 23.54% | 18.90% | 14.84% | 10.44% |
| Step3 seul | 28.96% | +/- 1.26% | 25.08% | 11.92% | 0.4567 | 23.94% | 20.64% | 14.64% | 11.82% |
| Step2 | 28.50% | +/- 1.25% | 26.02% | 8.98% | 0.4416 | 24.58% | 21.34% | 14.44% | 11.14% |
| Heuristique fair | 25.28% | +/- 1.20% | 21.58% | 9.80% | 0.3811 | 25.58% | 23.52% | 14.54% | 11.08% |

## Winrate Par Siege

| Politique | player_0 | player_1 | player_2 | player_3 |
|---|---:|---:|---:|---:|
| Champion CBP | 32.06% | 30.32% | 33.41% | 33.33% |
| Step3 seul | 23.84% | 28.72% | 30.72% | 32.56% |
| Step2 | 25.08% | 26.88% | 31.10% | 30.94% |
| Heuristique fair | 23.48% | 23.76% | 24.94% | 28.94% |

## Tactiques

| Politique | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Champion CBP | 28.51% | 79.58% | 19.38% | 90.46% | 91.37% |
| Step3 seul | 29.54% | 71.38% | 27.16% | 70.38% | 95.32% |
| Step2 | 34.84% | 69.07% | 29.25% | 74.34% | 92.52% |
| Heuristique fair | 22.89% | 68.97% | 29.74% | 76.99% | 89.97% |

## Activite Des Tetes Champion

```json
{
  "champion_chancellor": {
    "checks": 1395,
    "forced": 72,
    "overrides": 471,
    "sum_best_margin": 126
  },
  "champion_baron": {
    "baron_hand_checks": 3293,
    "base_baron_plays": 1454,
    "overrides": 1558
  },
  "champion_prince": {
    "base_prince_plays": 1608,
    "overrides": 617,
    "prince_hand_checks": 2482
  }
}
```

## Conclusion Provisoire

Cette arena est le pont entre evaluation classique et self-play. Si le champion domine proprement cette table, la prochaine etape logique est de transformer ce lignage en population d'entrainement, avec conservation des anciens checkpoints pour eviter l'overfitting a un seul adversaire.
