# Step5 - Tete D'Execution Chancelier V1

Date: 2026-04-26.

## Contexte

Step4 a montre que le dernier Step3 rapide sait globalement jouer fort, mais
que certaines cartes etaient mal diagnostiquees par les ablations. Pour
Chancelier, randomiser le choix des cartes a garder/remettre ne degradait presque
pas le composite. Cela suggerait que le modele savait parfois **quand** jouer
Chancelier, mais qu'il n'avait pas encore une expertise claire sur **comment**
l'executer.

Step5 commence donc par des tetes locales d'execution. Elles ne changent pas la
carte jouee par Step3. Elles n'interviennent que lorsque Step3 a deja choisi la
carte cible.

## Objectif

Entrainer une tete rapide autonome pour l'effet Chancelier:

- choisir quelle carte garder;
- choisir quelles cartes remettre sous la pioche;
- choisir l'ordre de remise;
- sans rollout a l'inference;
- sans casser le reste du jeu.

Le succes demande:

- un gain composite d'au moins `+0.010` contre Step3 rapide;
- pas de regression majeure contre `3H`;
- un controle contre `Chancelier random`, pour verifier que le gain vient d'une
  competence apprise et non d'un simple bruit de remplacement.

## Teacher Et Donnees

Dataset principal:

```text
step5_execution_heads/datasets/chancellor_teacher_attempt1_balanced_240x12.json
```

Collecte:

```bash
python3 -m step5_execution_heads.collect_execution_teacher \
  --games 1200 \
  --states-per-config-kind 60 \
  --max-states-per-kind 240 \
  --rollouts-per-action 12 \
  --seed-start 640000 \
  --kinds chancellor_choice
```

Le dataset est equilibre sur les quatre compositions d'arena.

| Mesure | Valeur |
|---|---:|
| Etats Chancelier | 240 |
| Etats par composition | 60 |
| Regret clair | 102 / 240 = 42.50% |
| Best action differente du modele | 174 / 240 = 72.50% |
| Etats forces | 6 / 240 = 2.50% |
| Mean score regret | 0.1721 |
| Mean win regret | 0.1601 |

Lecture: le signal teacher est fort. Il existe beaucoup d'etats ou l'execution
Chancelier de Step3 rapide laisse une marge exploitable.

## Modeles Testes

Trois variantes ont ete essayees:

| Candidat | Idee | Resultat court |
|---|---|---|
| `attempt1_balanced` | MLP 192, dataset complet equilibre | Positif mais trop d'overrides |
| `attempt2_clear_only` | Entrainer uniquement sur regrets clairs | Trop peu de no-override appris |
| `attempt3_small_regularized` | MLP plus petit, dropout, regularisation, moins d'epochs | Meilleur compromis |

Checkpoint retenu:

```text
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
```

Parametres d'evaluation retenus:

```text
chancellor_margin = 0.12
verify_rollouts = 0
```

Donc cette tete est bien rapide et autonome: aucun rollout pendant le jeu.

## Resultats Arena

### Validation 500 Parties/Composition

Seed start: `850000`.

| Politique | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 56.80% | 46.40% | 39.20% | 33.00% | 0.39920 |
| Step3 + tete Chancelier | 57.40% | 48.20% | 40.40% | 33.00% | 0.40700 |

Delta composite: `+0.00780`.

Lecture: positif, mais pas encore suffisant pour declarer un succes.

### Validation 1000 Parties/Composition - Bloc 1

Seed start: `860000`.

| Politique | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 52.70% | 45.50% | 38.90% | 32.80% | 0.39160 |
| Step3 + tete Chancelier | 55.80% | 47.20% | 39.50% | 34.60% | 0.40710 |

Delta composite: `+0.01550`.

Metriques Chancelier:

| Politique | Keep highest | Known draw win | Checks | Overrides | Override rate |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 70.34% | 61.80% | 0 | 0 | 0.00% |
| Step3 + tete Chancelier | 89.61% | 61.04% | 1163 | 337 | 28.98% |

### Validation 1000 Parties/Composition - Bloc 2

Seed start: `870000`.

| Politique | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 54.30% | 45.90% | 38.70% | 32.80% | 0.39340 |
| Step3 + tete Chancelier | 54.50% | 46.00% | 41.80% | 34.10% | 0.40830 |

Delta composite: `+0.01490`.

Metriques Chancelier:

| Politique | Keep highest | Known draw win | Checks | Overrides | Override rate |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 70.33% | 60.22% | 0 | 0 | 0.00% |
| Step3 + tete Chancelier | 89.55% | 56.77% | 1121 | 328 | 29.26% |

## Controle Contre Chancelier Random

Le controle random garde la decision principale du Step3 rapide. Il intervient
seulement quand Step3 a deja choisi Chancelier, puis choisit au hasard parmi les
executions Chancelier legales.

### Bloc 1 - Seed 860000

| Politique | Composite | Delta vs Step3 |
|---|---:|---:|
| Step3 rapide | 0.39160 | +0.00000 |
| Chancelier random | 0.39200 | +0.00040 |
| Tete Chancelier apprise | 0.40710 | +0.01550 |

### Bloc 2 - Seed 870000

| Politique | Composite | Delta vs Step3 |
|---|---:|---:|
| Step3 rapide | 0.39340 | +0.00000 |
| Chancelier random | 0.39820 | +0.00480 |
| Tete Chancelier apprise | 0.40830 | +0.01490 |

Lecture: randomiser Chancelier peut parfois bouger un peu le score, mais reste
nettement sous la tete apprise. Le gain principal vient bien d'une competence
d'execution apprise, pas seulement d'un remplacement aleatoire.

## Conclusion

La tete Chancelier V1 est un succes Step5.

Elle respecte les criteres:

- gain `+0.01550` puis `+0.01490` sur deux validations independantes de `1000`
  parties par composition;
- pas de regression contre `3H`; au contraire, `+1.8` pt puis `+1.3` pt;
- inference rapide, sans rollout;
- controle random inferieur a la tete apprise;
- comportement interpretable: le taux de conservation de la meilleure carte
  passe d'environ `70%` a presque `90%`.

Limite importante: la tete apprend surtout le tri local "garder la meilleure
carte". Elle ne semble pas encore ameliorer la planification fine de la pioche:
le taux `known draw win` ne progresse pas et baisse meme sur le deuxieme bloc.

## Suite Logique

Pour Step5, il faut maintenant:

1. Conserver `chancellor_head_attempt3_small_regularized.pth` comme premiere
   tete d'execution validee.
2. Brancher cette tete dans le joueur composite Step5.
3. Repliquer la meme methode sur la prochaine faiblesse:
   - Baron faible d'abord, car le regret teacher est plus fort;
   - puis Roi;
   - puis Pretre si le signal reste utile.
4. Prevoir une V2 Chancelier plus tard pour apprendre la planification de pioche
   et pas seulement la conservation de la meilleure carte.

## Fichiers

Train:

```text
step5_execution_heads/train_chancellor_head.py
step5_execution_heads/chancellor_head.py
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
step5_execution_heads/reports/chancellor_head_attempt3_small_regularized_train.json
```

Evaluations:

```text
step5_execution_heads/reports/chancellor_head_attempt3_small_regularized_eval_500_m012.md
step5_execution_heads/reports/chancellor_head_attempt3_small_regularized_eval_1000_m012.md
step5_execution_heads/reports/chancellor_head_attempt3_small_regularized_eval_1000_m012_seed870000.md
step4_weakness_analysis/reports/step5_compare_chancellor_random_1000_seed860000.md
step4_weakness_analysis/reports/step5_compare_chancellor_random_1000_seed870000.md
```
