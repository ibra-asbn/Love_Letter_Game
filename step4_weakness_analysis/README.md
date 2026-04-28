# Step 4 - Identification Des Faiblesses

Date de creation: 2026-04-25.

Cette etape ne cree pas encore un nouveau modele. Elle sert a comprendre le
dernier Step3 avec un protocole propre:

- verifier les winrates dans l'arene fair seat-rotated apres correction des
  regles;
- figer une taxonomie des cartes et des phases de partie;
- analyser le dernier Step3 rapide par archetypes de mains/cartes;
- mesurer non seulement s'il perd, mais comment il perd: premier sorti,
  deuxieme sorti, troisieme sorti, ou finaliste perdant.

Modele analyse par defaut:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Taxonomie:

```text
step4_weakness_analysis/CARD_TAXONOMY.md
```

Script principal:

```bash
python3 -m step4_weakness_analysis.cluster_step3_card_archetypes \
  --games 1000 \
  --seed-start 260000 \
  --output step3_fast_card_clusters_post_rules_fix_1000.json \
  --markdown step3_fast_card_clusters_post_rules_fix_1000.md
```

Les rapports produits vont dans:

```text
step4_weakness_analysis/reports/
step4_weakness_analysis/logs/
```

Script d'ablation conditionnelle:

```bash
python3 -m step4_weakness_analysis.evaluate_card_ablation_arena \
  --games 1000 \
  --seed-start 260000
```

## Resultats Actuels

Run principal aligne avec l'arene interlude post-fix:

```text
step4_weakness_analysis/reports/step3_fast_card_clusters_post_rules_fix_1000.md
```

Recap avance lisible:

```text
step4_weakness_analysis/reports/2026-04-25_step4_advanced_model_state.md
```

Analyse des ablations conditionnelles:

```text
step4_weakness_analysis/reports/2026-04-25_step4_card_ablation_analysis.md
```

Resume:

| Mesure | Resultat |
|---|---:|
| Parties | 4000 |
| Composite | 0.39750 |
| Winrate moyen | 42.75% |
| Reward moyen | 0.6447 |

Quand le modele perd, sa position de sortie est:

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 33.01% | 27.90% | 18.82% | 20.26% |

Archetypes faibles a surveiller en priorite:

| Archetype | Games | Winrate |
|---|---:|---:|
| Partie avec Roi | 464 | 44.61% |
| Baron avec petite carte | 1010 | 47.13% |
| Princesse tot | 445 | 48.76% |
| Grosse carte tot | 1194 | 49.75% |
| Partie avec Chancelier | 1142 | 51.14% |

Run secondaire, seed independant:

```text
step4_weakness_analysis/reports/step3_fast_card_clusters_1000.md
```

Il descend a `0.37240` composite. Ce n'est pas une contradiction: cela donne
un premier signal que les conclusions Step4 doivent etre confirmees sur
plusieurs seeds avant d'en faire une nouvelle regle d'entrainement.

## Ablations Conditionnelles

Principe: on laisse Step3 rapide choisir naturellement la carte jouee, puis on
randomise seulement l'execution fine de cette carte.

Rapport brut:

```text
step4_weakness_analysis/reports/step3_fast_card_ablation_1000.md
```

Resume:

| Ablation | Composite | Delta vs normal | Lecture |
|---|---:|---:|---|
| Normal | 0.39750 | +0.00000 | Reference |
| Garde cible random | 0.38330 | -0.01420 | Ciblage Garde utile |
| Garde guess random | 0.33320 | -0.06430 | Guess Garde tres maitrise |
| Garde cible+guess random | 0.33220 | -0.06530 | Execution Garde tres forte |
| Pretre cible random | 0.39900 | +0.00150 | Pas d'effet clair |
| Baron cible random | 0.39100 | -0.00650 | Ciblage Baron un peu utile |
| Prince cible random | 0.37470 | -0.02280 | Ciblage Prince net |
| Roi cible random | 0.39750 | +0.00000 | Faiblesse Roi pas expliquee par cible seule |
| Chancelier choix random | 0.39550 | -0.00200 | Choix Chancelier peu converti en winrate |

Conclusion courte: Garde et Prince sont des competences a proteger. Roi,
Baron faible et Chancelier demandent maintenant un audit decisionnel avec
`forcedness` et regret rollout avant tout entrainement cible.

Decision de passage vers Step5:

```text
step4_weakness_analysis/reports/2026-04-25_step4_to_step5_decision.md
```
