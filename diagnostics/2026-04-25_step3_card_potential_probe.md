# Probe Step3 - Potentiel Action-Value Par Carte

Date: 25 avril 2026.

Objectif: identifier rapidement les cartes sur lesquelles le search
action-value pourrait apporter le plus de gain au modele Step2.

Checkpoint teste:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Commande:

```bash
python3 -m step3_action_value.card_potential_probe \
  --checkpoint step2_retarget_distilled_attempt1.pth \
  --opponent-config vs_3H \
  --states-per-card 14 \
  --rollouts-per-action 12 \
  --max-actions 14 \
  --high-margin 0.12 \
  --seed 9600
```

Le probe collecte des etats ou chaque carte est en main, evalue plusieurs
actions candidates par rollouts determinises, puis mesure le regret entre
l'action Step2 et la meilleure action estimee.

Correction importante ajoutee apres discussion: le regret est maintenant
attribue en deux temps.

Exemple: main `Comtesse + Garde`, Step2 joue `Garde`.

- Pour mesurer "mal jouer Garde", on compare l'action Garde de Step2 a la
  meilleure action Garde.
- Pour dire "il fallait jouer Comtesse", on compare Comtesse a la meilleure
  action Garde, pas a la mauvaise action Garde choisie par Step2.

Cela evite de donner artificiellement du regret a une carte presente en main
alors que le vrai probleme est seulement la cible/devinette de l'autre carte.

## Ranking

Version corrigee par attribution carte-vs-carte:

| Carte | Potentiel attribue | Gain freq. par decision | Etats a forte marge | Execution | Missed | Avoid |
|---|---:|---:|---:|---:|---:|---:|
| Garde | 17.26 pts | 1.95 pts | 50.00% | 3.57 pts | 8.33 pts | 5.36 pts |
| Pretre | 15.48 pts | 1.75 pts | 42.86% | 2.98 pts | 5.95 pts | 6.55 pts |
| Espionne | 10.12 pts | 1.14 pts | 21.43% | 0.00 pts | 8.93 pts | 1.19 pts |
| Roi | 8.33 pts | 0.94 pts | 21.43% | 1.19 pts | 1.19 pts | 5.95 pts |
| Prince | 7.74 pts | 0.87 pts | 14.29% | 4.76 pts | 0.60 pts | 2.38 pts |
| Chancelier | 7.74 pts | 0.87 pts | 28.57% | 0.00 pts | 1.79 pts | 5.95 pts |
| Baron | 6.55 pts | 0.74 pts | 35.71% | 5.36 pts | 1.19 pts | 0.00 pts |
| Servante | 2.98 pts | 0.34 pts | 7.14% | 0.00 pts | 0.00 pts | 2.98 pts |
| Comtesse | 0.00 pts | 0.00 pts | 0.00% | 0.00 pts | 0.00 pts | 0.00 pts |
| Princesse | 0.00 pts | 0.00 pts | 0.00% | 0.00 pts | 0.00 pts | 0.00 pts |

Definitions:

- `Potentiel attribue`: winrate estime perdu et attribuable a cette carte apres
  correction carte-vs-carte.
- `Gain freq. par decision`: regret cumule rapporte a toutes les decisions
  player_0 scannees.
- `Execution`: Step2 joue cette carte, mais pas avec la meilleure cible/devinette.
- `Missed`: cette carte aurait battu la meilleure version de la carte jouee.
- `Avoid`: Step2 joue cette carte, mais une autre carte bat sa meilleure version.

## Lecture

Le Prêtre est bien interessant. Il sort deuxieme du ranking, tres proche du
Garde. Le signal n'est pas seulement "jouer ou ne pas jouer Pretre": il y a
aussi de la cible a apprendre. C'est logique: l'information prise par Pretre
peut transformer les decisions suivantes, surtout Garde et Baron.

La Comtesse tombe a `0.00` apres correction. C'est exactement le comportement
attendu: les regrets observes dans les mains contenant Comtesse venaient des
autres cartes, par exemple une mauvaise cible de Garde, et pas d'un mauvais
choix de Comtesse.

Le Baron reste prioritaire aussi, mais le type de correction est different:
le plus gros signal est `Refine`, donc choisir la bonne cible de Baron plus que
decider de jouer Baron ou non.

Le Roi ressort haut, mais on doit le traiter prudemment. Comme deja discute, le
Roi est difficile a juger localement: sa valeur depend beaucoup de la fin de
round, de l'information revelee et du risque que l'adversaire connaisse notre
nouvelle carte.

## Conclusion

Pour la prochaine etape de search/distillation, les cartes a prioriser sont:

1. Garde
2. Pretre
3. Baron, surtout pour la cible/raffinement
4. Espionne / Roi a confirmer, mais avec metriques dediees plus prudentes

Prince reste utile, mais ce mini-probe dit qu'il n'est pas la premiere source
de gain marginal sur cette distribution d'etats.

Rapport brut:

```text
step3_action_value/reports/step3_card_potential_probe_vs3H_14x12_seed9600_adjusted.json
```
