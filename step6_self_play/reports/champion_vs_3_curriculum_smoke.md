# Step6 - Matchups Asymetriques

Date: 2026-04-26 17:46:57 CEST.

Parties par matchup: `24`.

Evaluation uniquement: un singleton joue contre trois copies d'un autre profil. Le siege du singleton tourne a chaque manche.

## Synthese

| Matchup | Singleton | Score >=1 | Victoire manche | Reward | Opposants/copie | Score >=1 | Victoire manche | Reward |
|---|---|---:|---:|---:|---|---:|---:|---:|
| champion_vs_3_curriculum | Champion CBP | 37.50% | 29.17% | 0.5083 | Curriculum phase1 | 31.94% | 25.00% | 0.5083 |

## Details Par Matchup

### champion_vs_3_curriculum

| Groupe | Politique | N | Score >=1 | IC95 | Victoire manche | Bonus Espionne | Reward | 1er sorti | 2e sorti | 3e sorti | Perd final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Singleton | Champion CBP | 24 | 37.50% | +/- 19.37% | 29.17% | 12.50% | 0.5083 | 16.67% | 29.17% | 8.33% | 8.33% |
| Opposants/copie | Curriculum phase1 | 72 | 31.94% | +/- 10.77% | 25.00% | 16.67% | 0.5083 | 27.78% | 16.67% | 12.50% | 11.11% |

| Groupe | Guard hit | Baron gagne | Baron perdu | Chancelier garde max | Priest->Guard |
|---|---:|---:|---:|---:|---:|
| Singleton | 28.57% | 60.00% | 40.00% | 87.50% | 100.00% |
| Opposants/copie | 25.97% | 87.50% | 6.25% | 68.75% | 100.00% |

Activite des tetes champion:

```json
{
  "singleton_champion_chancellor": {
    "checks": 5,
    "forced": 3,
    "sum_best_margin": 0
  },
  "singleton_champion_baron": {
    "baron_hand_checks": 16,
    "base_baron_plays": 7,
    "overrides": 9
  },
  "singleton_champion_prince": {
    "base_prince_plays": 6,
    "overrides": 5,
    "prince_hand_checks": 12
  }
}
```

## Lecture

Ces matchups ne remplacent pas l'arena de lignage complete. Ils mesurent plutot si un profil singleton tient quand les trois autres joueurs ont tous le meme niveau/style. C'est un bon test d'exploitabilite avant d'entrainer en self-play.