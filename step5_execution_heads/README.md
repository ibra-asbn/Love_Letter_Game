# Step5 - Execution Heads

Date de lancement: 2026-04-25.

## Pourquoi Step5 ?

Step4 a montre que le Step3 rapide n'est pas un mauvais joueur global: il bat
l'heuristique fair au composite et possede de vraies competences, notamment
sur le Garde et le Prince.

Mais Step4 a aussi montre une limite precise:

> Quand le modele decide deja de jouer Roi, Baron, Pretre ou Chancelier, on n'a
> pas encore la preuve qu'il execute ces cartes mieux que le hasard.

La bonne direction n'est donc pas de relancer un PPO global, ni de reecrire des
regles a la main. On veut apprendre des corrections locales d'execution.

## Hypothese

Le Step3 rapide sait souvent **quand jouer** une carte. Step5 cherche a lui
apprendre les corrections locales que le modele global ne maitrise pas encore:

- Roi: choisir la bonne cible d'echange.
- Baron: comparer localement `jouer Baron` contre `jouer l'autre carte`, puis
  choisir une cible sure.
- Pretre: choisir la cible dont l'information a le plus de valeur.
- Chancelier: choisir quelle carte garder et dans quel ordre remettre les autres.

## Plan

### Phase A - Teacher/Audit D'Execution

Collecter des etats naturels du Step3 rapide. On ne force jamais une carte:
on garde uniquement les moments ou le modele choisit deja Roi, Baron, Pretre ou
Chancelier.

Pour chaque etat, on evalue toutes les executions legales par rollouts CRN:

```text
state
card_or_subdecision
legal_executions
model_execution
oracle_scores
best_execution
regret
confidence
forcedness
```

### Phase B - Dataset Filtre

On garde seulement les etats ou:

- plusieurs executions etaient legalement possibles;
- l'oracle trouve une alternative clairement meilleure;
- le signal est assez stable pour ne pas apprendre du bruit.

Les etats ou toutes les options se valent ne doivent pas entrainer le modele.

### Phase C - Tetes D'Execution

Entrainer de petites tetes specialisees:

- `king_target_head`;
- `baron_target_head`;
- `priest_target_head`;
- `chancellor_choice_head`.

Elles ne choisissent pas la carte a jouer. Elles corrigent seulement
l'execution lorsque le Step3 rapide a deja choisi la carte.

### Phase D - Integration Prudente

Une tete ne peut override l'execution du Step3 que si:

- la carte correspond;
- l'action proposee est legale;
- le score/marge est clair;
- la correction ne sort pas de la region de confiance.

## Criteres De Succes

Step5 sera considere comme un succes si:

1. Le teacher trouve un regret exploitable sur au moins une des cartes ciblees.
2. Une tete rapide autonome ameliore le Step3 rapide sur arena fair
   seat-rotated.
3. Le gain composite est positif sur seed independant, cible minimale:
   `+0.010` composite vs Step3 rapide.
4. Pas de regression majeure contre `3H`: tolerance maximale `-0.010`.
5. Les competences deja acquises sont preservees:
   - Garde guess ne doit pas s'effondrer;
   - Prince cible ne doit pas s'effondrer.
6. L'inference reste rapide: pas de rollouts pendant le jeu.

## Criteres D'Echec

Step5 est un echec si:

- le teacher ne trouve pas de regret clair;
- les labels sont trop bruites;
- la tete ameliore une carte mais degrade le composite global;
- la tete casse Garde/Prince;
- le gain n'est visible que sur un seed de `1000` parties et disparait ensuite.

## Premiere Brique

Script teacher/audit:

```bash
python3 -m step5_execution_heads.collect_execution_teacher \
  --games 300 \
  --max-states-per-kind 40 \
  --rollouts-per-action 12
```

Rapports:

```text
step5_execution_heads/reports/
step5_execution_heads/logs/
step5_execution_heads/datasets/
```

## Phase A - Resultat Initial

Run principal du 2026-04-25:

```text
step5_execution_heads/reports/execution_teacher_initial_40x12_report.md
```

Conclusion detaillee:

```text
step5_execution_heads/reports/2026-04-25_step5_phase_a_execution_teacher_conclusion.md
```

Synthese:

| Type | Etats | Best != modele | Regret clair | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|
| Chancelier - choix carte/ordre | 40 | 67.50% | 37.50% | 0.1199 | 0.1104 |
| Baron avec carte faible - cible | 40 | 65.00% | 20.00% | 0.0715 | 0.0667 |
| Roi - cible | 40 | 30.00% | 17.50% | 0.0615 | 0.0583 |
| Baron - cible | 40 | 32.50% | 15.00% | 0.0562 | 0.0521 |
| Pretre - cible | 40 | 40.00% | 12.50% | 0.0382 | 0.0354 |

Lecture: la phase A est reussie. Le teacher trouve bien du regret exploitable,
surtout sur Chancelier, puis Baron faible, Roi et Baron. Pretre reste candidat
secondaire.

La suite Step5 ne doit donc pas etre un entrainement global: elle doit commencer
par des tetes d'execution ciblees, en priorite Chancelier, avec protection des
competences Garde et Prince acquises aux etapes precedentes.

## Phase B/C - Tete Chancelier V1

Run principal du 2026-04-26:

```text
step5_execution_heads/reports/2026-04-26_step5_chancellor_execution_head_v1.md
```

Dataset Chancelier equilibre:

```text
step5_execution_heads/datasets/chancellor_teacher_attempt1_balanced_240x12.json
```

Checkpoint valide:

```text
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
```

Cette tete corrige uniquement l'execution du Chancelier lorsque Step3 rapide a
deja decide de jouer Chancelier. Elle ne fait aucun rollout a l'inference.

| Validation | Step3 rapide | Step3 + tete Chancelier | Delta | Chancelier random |
|---|---:|---:|---:|---:|
| 500/config seed 850000 | 0.39920 | 0.40700 | +0.00780 | non teste |
| 1000/config seed 860000 | 0.39160 | 0.40710 | +0.01550 | 0.39200 |
| 1000/config seed 870000 | 0.39340 | 0.40830 | +0.01490 | 0.39820 |

Conclusion: succes Step5 Chancelier V1. Le gain est superieur au controle
`Chancelier random`, ne degrade pas `vs 3H`, et reste rapide. La limite observee
est claire: la tete apprend surtout a garder la meilleure carte, mais pas encore
la planification fine de la pioche.

## Phase B/C - Baron Specialist V1

Run principal du 2026-04-26:

```text
step5_execution_heads/cards/baron/reports/baron_specialist_v1_eval_5000_seed2100000.md
```

Conclusion:

```text
step5_execution_heads/cards/baron/reports/2026-04-26_baron_specialist_v1_success.md
```

Contrairement au Chancelier, Baron n'est pas seulement un choix d'execution. Le
diagnostic a montre que Step3 joue deja Baron avec les bonnes grosses cartes,
mais perd trop de duels avec `Baron + Prince` et `Baron + Chancelier`.

Le correctif V1 compare donc localement:

```text
jouer Baron sur la meilleure cible
vs
jouer l'autre carte de la main
```

Evaluation: 5000 parties par composition d'arene.

| Politique | Composite | Baron en main | Baron joue | Duel gagne | Duel perdu |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38564 | 46.51% | 48.89% | 71.65% | 24.44% |
| Baron target random | 0.38230 | 45.66% | 47.71% | 69.92% | 26.22% |
| Step5 Baron specialist | 0.39504 | 49.02% | 54.88% | 79.83% | 17.01% |

Conclusion: succes Step5 Baron V1. Le gain composite est positif et la faiblesse
mesuree sur `Baron + Prince` est fortement corrigee.

## Evaluation Combinee - Chancelier V1 + Baron V1

Run principal du 2026-04-26:

```text
step5_execution_heads/reports/combined_chancellor_baron_eval_5000_seed2300000.md
```

Conclusion:

```text
step5_execution_heads/reports/2026-04-26_combined_chancellor_baron_success.md
```

Evaluation: 5000 parties par composition d'arene.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.68% | 44.16% | 38.76% | 34.00% | 0.39228 |
| Step3 + Chancelier V1 | 52.62% | 45.46% | 39.80% | 35.72% | 0.40582 |
| Step3 + Baron V1 | 52.48% | 45.74% | 39.00% | 35.16% | 0.40160 |
| Step3 + Chancelier + Baron | 53.36% | 46.86% | 39.96% | 37.00% | 0.41496 |

Conclusion: les deux modules s'additionnent bien. Le joueur Step5 de reference
devient `Step3 rapide + Chancelier V1 + Baron V1`.

## Phase B/C - Prince V1

Run principal du 2026-04-26:

```text
step5_execution_heads/cards/prince/reports/prince_specialist_v1_eval_5000_seed2500000.md
```

Conclusion:

```text
step5_execution_heads/cards/prince/reports/2026-04-26_prince_v1_first_eval.md
```

Evaluation: 5000 parties par composition d'arene, avec lecture principale sur
les parties ou le joueur evalue a eu un Prince en main.

| Politique | Composite global | Prince en main | Prince joue | Hit Princesse | Suicide soi |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.39162 | 48.31% | 52.02% | 7.64% | 0.51% |
| Step3 + Prince V1 | 0.39428 | 48.97% | 51.03% | 9.38% | 0.35% |

Conclusion: succes leger et signal exploitable. Prince V1 ameliore le winrate
conditionnel avec Prince en main et touche plus souvent la Princesse, mais force
trop souvent le Prince. Il reste candidat, pas encore module de reference.
