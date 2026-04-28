# Step5 Combine - Chancelier V1 + Baron V1

Date: 2026-04-26.

## Decision

**Succes valide.**

Le Step3 rapide a ete evalue seul, puis avec les deux modules Step5 valides:

- Chancelier V1: tete rapide de choix de carte/ordre.
- Baron V1: specialiste action-value `jouer Baron vs jouer l'autre carte`.

Rapport principal:

```text
step5_execution_heads/reports/combined_chancellor_baron_eval_5000_seed2300000.md
```

## Resultats

Evaluation: 5000 parties par composition d'arene.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.68% | 44.16% | 38.76% | 34.00% | 0.39228 |
| Step3 + Chancelier V1 | 52.62% | 45.46% | 39.80% | 35.72% | 0.40582 |
| Step3 + Baron V1 | 52.48% | 45.74% | 39.00% | 35.16% | 0.40160 |
| Step3 + Chancelier + Baron | 53.36% | 46.86% | 39.96% | 37.00% | 0.41496 |

## Deltas Vs Step3 Rapide

- Chancelier V1: `+0.01354` composite.
- Baron V1: `+0.00932` composite.
- Chancelier V1 + Baron V1: `+0.02268` composite.

## Metriques Tactiques

| Politique | Guard hit | Baron win | Baron loss | Chancellor keep highest |
|---|---:|---:|---:|---:|
| Step3 rapide | 30.24% | 73.33% | 24.29% | 70.22% |
| Step3 + Chancelier V1 | 30.37% | 73.86% | 23.82% | 88.99% |
| Step3 + Baron V1 | 28.93% | 79.93% | 18.68% | 71.29% |
| Step3 + Chancelier + Baron | 29.10% | 80.30% | 18.33% | 88.98% |

## Conclusion

Les deux modules s'additionnent bien:

- Chancelier ameliore fortement les decisions de pool.
- Baron reduit fortement les duels perdus.
- La combinaison ameliore toutes les compositions, y compris `vs 3H`.

Le joueur Step5 actuel de reference devient donc:

```text
Step3 rapide + Chancelier V1 + Baron V1
```
