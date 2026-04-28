# Step5 Baron V1 - Conclusion

Date: 2026-04-26.

## Decision

**Succes valide.**

Le Step5 Baron V1 devient un specialiste action-value local. Il ne corrige pas
seulement la cible du Baron: il decide aussi si jouer Baron est meilleur que
jouer l'autre carte disponible.

Script:

```text
step5_execution_heads/cards/baron/evaluate_baron_specialist.py
```

Rapport principal:

```text
step5_execution_heads/cards/baron/reports/baron_specialist_v1_eval_5000_seed2100000.md
```

Log:

```text
step5_execution_heads/cards/baron/logs/2026-04-26_baron_specialist_v1_eval_5000_seed2100000.md
```

## Resultats Principaux

Evaluation: 5000 parties par composition d'arene, soit 20000 parties par
politique.

| Politique | Composite | Baron en main | Baron joue | Duel gagne | Duel perdu |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38564 | 46.51% (n=8443) | 48.89% (n=6083) | 71.65% | 24.44% |
| Baron target random | 0.38230 | 45.66% (n=8443) | 47.71% (n=6083) | 69.92% | 26.22% |
| Step5 Baron specialist | 0.39504 | 49.02% (n=8443) | 54.88% (n=4891) | 79.83% | 17.01% |

Gains vs Step3 rapide:

- Composite: `+0.00940`.
- Winrate quand Baron est en main: `+2.51 points`.
- Winrate quand Baron est joue: `+5.99 points`.
- Duels gagnes: `+8.18 points`.
- Duels perdus: `-7.43 points`.

## Lecture Par Carte Accompagnante

Le specialiste corrige exactement le probleme identifie:

| Carte avec Baron | Step3 winrate main | Specialist winrate main | Step3 duel perdu | Specialist duel perdu |
|---|---:|---:|---:|---:|
| Prince | 36.25% | 48.59% | 40.10% | 26.80% |
| Chancelier | 46.11% | 47.47% | 27.19% | 21.91% |
| Roi | 48.47% | 50.63% | 22.70% | 18.63% |
| Comtesse | 60.54% | 62.39% | 10.35% | 10.53% |
| Princesse | 74.43% | 73.09% | 0.00% | 0.00% |

Le cas `Baron + Prince` est la correction majeure: Step3 jouait Baron dans
94.78% des cas, alors que le specialiste descend a 24.23% et gagne beaucoup
plus souvent quand Baron est en main.

## Conclusion

Baron V1 est valide parce qu'il ameliore le composite global tout en corrigeant
la faiblesse tactique precise qui avait ete mesuree: les duels trop risques avec
cartes moyennes-fortes.

La prochaine etape possible sera de distiller cette regle action-value dans une
petite tete neuronale, mais ce n'est plus necessaire pour prouver la pertinence
du correctif Baron.
