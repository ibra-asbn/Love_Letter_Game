# Step5 - Baron

Date: 2026-04-26.

## Objectif

Construire un specialiste Baron qui aide le joueur Step3 uniquement dans les
etats ou Baron est en main.

Le point important est que Baron n'est pas seulement un probleme de cible. Avec
`Baron + Prince` ou `Baron + Chancelier`, la vraie decision est:

1. jouer Baron maintenant, et choisir une cible;
2. ou jouer l'autre carte et garder Baron / changer la main.

Step5 Baron part donc directement d'une logique **action-value locale**:

> evaluer `Baron -> cible` contre l'action non-Baron disponible.

## Diagnostic Initial

Rapport source:

```text
reports/baron_companion_audit_step3_5000_seed1800000.md
```

Sur 5000 parties par composition d'arene, Step3 montre deja une vraie
comprehension de `quand` Baron est naturellement fort:

| Carte avec Baron | Pct joue Baron | Winrate si joue | Duel gagne | Duel perdu |
|---|---:|---:|---:|---:|
| Espionne | 0.36% | 0.00% | 0.00% | 60.00% |
| Garde | 0.11% | 50.00% | 0.00% | 50.00% |
| Pretre | 2.36% | 25.00% | 32.14% | 67.86% |
| Servante | 1.53% | 66.67% | 94.44% | 5.56% |
| Prince | 94.41% | 35.93% | 55.01% | 39.83% |
| Chancelier | 95.92% | 41.21% | 65.87% | 29.12% |
| Roi | 99.33% | 52.09% | 78.27% | 17.94% |
| Comtesse | 100.00% | 62.71% | 84.48% | 11.88% |
| Princesse | 100.00% | 72.11% | 97.79% | 0.00% |

Lecture:

- Step3 evite presque toujours Baron avec les tres petites cartes.
- Step3 joue presque toujours Baron avec les grosses cartes.
- La faiblesse principale concerne les cartes moyennes-fortes:
  `Prince` et `Chancelier`.

## Strategie Retenue

La premiere version validee est simple, explicable et mesurable:

- ne jamais toucher aux tours sans Baron en main;
- conserver l'agressivite avec `Roi`, `Comtesse`, `Princesse`;
- etre plus prudent avec `Prince` et `Chancelier` si le risque de mourir est
  trop haut;
- ne jouer Baron avec une petite carte que si l'information rend le duel
  quasiment certain;
- mesurer le succes sur les parties ou Baron apparait en main, pas seulement
  sur les parties ou Baron est joue.

## Resultat V1

Rapport principal:

```text
reports/baron_specialist_v1_eval_5000_seed2100000.md
```

Conclusion:

```text
reports/2026-04-26_baron_specialist_v1_success.md
```

Evaluation: 5000 parties par composition d'arene.

| Politique | Composite | Baron en main | Baron joue | Duel gagne | Duel perdu |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38564 | 46.51% | 48.89% | 71.65% | 24.44% |
| Baron target random | 0.38230 | 45.66% | 47.71% | 69.92% | 26.22% |
| Step5 Baron specialist | 0.39504 | 49.02% | 54.88% | 79.83% | 17.01% |

Statut: **V1 validee**.

Le gain principal vient de `Baron + Prince`: Step3 jouait Baron dans 94.78% des
cas et gagnait 36.25% des manches ou Baron etait en main. Le specialiste ne joue
Baron que dans 24.23% de ces cas, mais monte a 48.59% de winrate quand Baron est
en main.

## Fichiers

- `analyze_baron_companion.py`: audit de l'usage de Baron selon la carte
  accompagnante.
- `evaluate_baron_specialist.py`: evaluation du specialiste Baron action-value.
- `reports/baron_companion_audit_step3_5000_seed1800000.md`: diagnostic de
  depart.
- `reports/2026-04-26_baron_v2_action_value_plan.md`: plan methodologique.
- `reports/2026-04-26_baron_specialist_v1_success.md`: conclusion de validation.
