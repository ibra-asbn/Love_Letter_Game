# Diagnostic Step 3 - Action-Value / Search

Date de session: 24 avril 2026, logs termines apres minuit CEST.

## Etat De Depart

Checkpoint de depart:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Ce modele bat deja `HeuristicBot`, mais il reste derriere le champion historique
`curriculum_phase1.pth`. Les diagnostics precedents montraient un probleme
clair: le modele sait mieux utiliser le belief qu'avant, mais il prend encore
des decisions tactiques localement mauvaises, notamment autour de Garde, Baron
et Prince.

Objectif Step3: passer de "regles tactiques / retarget belief" a une estimation
de valeur par action dans le contexte.

## Resultats Avant Tentative

Evaluation Step2 brute, `1000` parties par configuration, seed `783000`:

| Policy | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 brut | 51.00% | 35.10% | 22.80% | 15.50% | 0.25160 |

## Tentatives Faites

### Tentative 1 - Distillation Directe

Fichier:

```text
step3_action_value/checkpoints/step3_rollout_distilled_attempt1.pth
```

Dataset collecte:

- 200 etats critiques;
- 113 labels changes;
- change rate: 56.5%;
- regret moyen: 11.0 points de winrate.

Resultat sur `1000` parties/config, seed `780000`:

| Policy | Composite |
|---|---:|
| Step2 brut | 0.26410 |
| Attempt1 distille | 0.25800 |

Conclusion: echec. Les labels etaient trop nombreux et trop bruites, surtout
avec Roi/Chancelier inclus. Le modele a modifie trop de choses et a regresse.

### Tentative 2 - Distillation Tactique Plus Prudente

Fichier:

```text
step3_action_value/checkpoints/step3_rollout_distilled_attempt2_tactical.pth
```

Dataset collecte:

- categories limitees a `baron`, `guard`, `prince`;
- 240 etats critiques;
- 54 labels changes;
- change rate: 22.5%;
- regret moyen des changements: 18.1 points.

Resultat sur `1000` parties/config, seed `780000`:

| Policy | Composite |
|---|---:|
| Step2 brut | 0.26410 |
| Attempt2 tactique | 0.25840 |

Conclusion: echec aussi. Les labels etaient plus propres, mais l'actor n'a pas
su apprendre assez fortement les corrections rares sans perdre de comportement
utile.

## Succes - Rollout-Guided Policy

Fichier:

```text
step3_action_value/evaluate_rollout_guided.py
```

Principe:

- Step2 reste l'action par defaut;
- sur `baron`, `guard`, `prince`, on evalue quelques actions legales par
  rollouts determinises;
- on remplace l'action du modele seulement si le meilleur coup a au moins
  `0.12` de marge de winrate estimee;
- continuation player0 et adversaires de rollout: heuristique;
- `12` rollouts par action, `10` actions candidates maximum.

Commande de confirmation:

```bash
python3 -m step3_action_value.evaluate_rollout_guided \
  --checkpoint step2_retarget_distilled_attempt1.pth \
  --games 1000 \
  --seed-start 783000 \
  --rollouts-per-action 12 \
  --max-actions 10 \
  --override-margin 0.12 \
  --player0-continuation heuristic
```

Resultat confirme:

| Policy | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 brut | 51.00% | 35.10% | 22.80% | 15.50% | 0.25160 |
| Step3 rollout-guided | 50.90% | 36.80% | 26.00% | 18.00% | 0.27450 |

Gain composite: `+2.29` points.

Details de guidance sur `1000` parties/config:

| Config | Checks tactiques | Overrides | Override/decision |
|---|---:|---:|---:|
| vs 3 randoms | 1966 | 854 | 22.02% |
| vs 1H+2R | 1747 | 826 | 23.77% |
| vs 2H+1R | 1499 | 720 | 23.63% |
| vs 3H | 1330 | 587 | 22.44% |

## Conclusion

Succes pour l'etape 3, mais pas encore sous la forme finale.

Ce qui est valide:

- l'action-value/search ajoute un vrai edge par rapport au Step2 brut;
- le gain est surtout visible contre les heuristiques, donc sur les adversaires
  qui punissent les erreurs tactiques;
- Baron/Garde/Prince sont bien les zones prioritaires.

Ce qui n'est pas encore valide:

- la distillation offline naive ne suffit pas;
- la policy rollout-guided est trop lente pour l'interface utilisateur;
- il faut transformer ce teacher lent en acteur rapide.

Suite recommandee:

1. Collecter un dataset plus large depuis `evaluate_rollout_guided.py`, mais en
   ne gardant que les overrides robustes.
2. Ajouter une tete action-value/Q ou une tete "override gate" plutot que
   remplacer directement toute la distribution de l'acteur.
3. Distiller seulement les deltas tactiques: "quand dois-je ignorer Step2 ?",
   pas "reapprendre toute la policy".
4. Reconfirmer sur `3000` a `5000` parties/config avant promotion champion.
