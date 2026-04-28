# Interlude - Heuristique vs Step2 vs Step3

Date: 2026-04-25 14:27:59 CEST.

Objectif: verifier proprement si les modeles actuels sont vraiment moins bons que `HeuristicBot`.

Protocole: 1000 parties par composition et par seed, seeds [134000, 135000, 136000].
La Step3 hybride `verify16` reutilise les rapports officiels deja calcules sur ces memes seeds.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| HeuristicBot | 46.40% | 32.57% | 21.50% | 11.83% | 0.22337 | +0.00000 |
| Step2 retarget | 51.57% | 37.50% | 24.73% | 16.17% | 0.26543 | +0.04207 |
| Step3 rapide DAgger | 53.87% | 38.27% | 26.97% | 16.63% | 0.27783 | +0.05447 |
| Step3 hybride verify16 | 53.30% | 38.87% | 26.47% | 17.40% | 0.28003 | +0.05667 |

## Lecture

- Un delta est considere clair seulement s'il depasse grossierement son intervalle d'incertitude approximatif.
- La question centrale est donc: les modeles Step2/Step3 sont-ils sous `HeuristicBot`, au meme niveau, ou au-dessus ?

- `Step2 retarget`: delta composite vs HeuristicBot = `+0.04207` ; IC95 approx du delta = `+/- 0.01094`.
- `Step3 rapide DAgger`: delta composite vs HeuristicBot = `+0.05447` ; IC95 approx du delta = `+/- 0.01103`.
- `Step3 hybride verify16`: delta composite vs HeuristicBot = `+0.05667` ; IC95 approx du delta = `+/- 0.01107`.

## Conclusion

`step2_retarget` est au-dessus de l'heuristique de facon nette: delta composite `+0.04207`, IC95 approx `+/- 0.01094`. `step3_fast_dagger` est au-dessus de l'heuristique de facon nette: delta composite `+0.05447`, IC95 approx `+/- 0.01103`. `step3_hybrid_verify16` est au-dessus de l'heuristique de facon nette: delta composite `+0.05667`, IC95 approx `+/- 0.01107`.
