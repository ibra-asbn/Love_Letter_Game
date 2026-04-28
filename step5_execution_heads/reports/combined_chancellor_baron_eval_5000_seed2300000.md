# Step5 - Evaluation Combinee Chancelier + Baron

Date: 2026-04-26 15:48:17 CEST.

Parties: `5000` par composition.
Chancelier: `step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth`.

## Winrates

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.68% | 44.16% | 38.76% | 34.00% | 0.39228 |
| Step3 + Chancelier V1 | 52.62% | 45.46% | 39.80% | 35.72% | 0.40582 |
| Step3 + Baron V1 | 52.48% | 45.74% | 39.00% | 35.16% | 0.40160 |
| Step3 + Chancelier + Baron | 53.36% | 46.86% | 39.96% | 37.00% | 0.41496 |

## Deltas Vs Base

- Step3 + Chancelier V1: `+0.01354` composite.
- Step3 + Baron V1: `+0.00932` composite.
- Step3 + Chancelier + Baron: `+0.02268` composite.

## Tactique Agregee

| Politique | Guard hit | Baron win | Baron loss | Chancellor keep highest |
|---|---:|---:|---:|---:|
| Step3 rapide | 30.24% | 73.33% | 24.29% | 70.22% |
| Step3 + Chancelier V1 | 30.37% | 73.86% | 23.82% | 88.99% |
| Step3 + Baron V1 | 28.93% | 79.93% | 18.68% | 71.29% |
| Step3 + Chancelier + Baron | 29.10% | 80.30% | 18.33% | 88.98% |