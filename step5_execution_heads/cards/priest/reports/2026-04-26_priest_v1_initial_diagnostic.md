# Step5 Pretre V1 - Diagnostic Initial

Date: 2026-04-26.

## Objectif

Demarrer la carte Pretre avec une premiere tete locale de ciblage.

Hypothese testee:

> Quand Step3 decide deja de jouer Pretre, une correction de cible peut augmenter
> la valeur de l'information vue.

Cette V1 ne change donc pas le choix `jouer Pretre ou non`. Elle ne fait que
retarget le Pretre.

## Run Principal

```bash
python3 step5_execution_heads/cards/priest/evaluate_priest_specialist.py \
  --games 1000 \
  --seed-start 2700000 \
  --output priest_specialist_v1_eval_1000_seed2700000.json \
  --markdown priest_specialist_v1_eval_1000_seed2700000.md
```

## Resultats

| Politique | Composite | Pretre en main | Pretre joue | Pretre sur inconnu | Carte 7+ vue | Princesse vue | Pretre->Garde hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38460 | 47.99% | 48.39% | 96.79% | 26.45% | 9.38% | 94.66% |
| Pretre target random | 0.39140 | 49.50% | 50.07% | 95.83% | 26.61% | 9.86% | 92.93% |
| Step3 + Pretre V1 | 0.38590 | 48.30% | 48.74% | 97.23% | 27.30% | 9.77% | 95.28% |

Lecture:

- Pretre V1 ameliore legerement les signaux tactiques:
  - plus de Pretres sur cible inconnue;
  - plus de cartes 7+ vues;
  - plus de Princesse vues;
  - Pretre->Garde hit legerement meilleur.
- Mais le gain arena est quasi nul: `+0.00130` composite seulement.
- Le controle `target random` fait mieux sur ce seed, ce qui interdit de valider
  Pretre V1.

## Test Agressif

On a aussi teste une variante plus agressive avec `--retarget-margin 0.00`.

Resultat:

| Politique | Composite | Pretre en main | Pretre joue | Carte 7+ vue | Pretre->Garde hit |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38520 | 48.06% | 48.39% | 28.20% | 88.89% |
| Pretre target random | 0.39020 | 49.78% | 50.39% | 27.91% | 86.89% |
| Pretre V1 margin 0 | 0.38200 | 47.45% | 47.68% | 28.90% | 90.12% |

La version agressive voit encore plus de grosses cartes, mais elle perd en
arena. Donc le probleme n'est pas seulement un seuil trop prudent.

## Diagnostic

Pretre V1 n'est **pas encore un succes**.

La conclusion importante est subtile: choisir une cible qui a probablement une
grosse carte ne suffit pas. Le Pretre ne marque pas de point directement; sa
valeur depend de ce que le modele fera ensuite.

Nos signaux montrent que Step3 exploite deja tres bien certaines infos:

- `Pretre->Garde hit` est deja tres haut, autour de 90-95%;
- les Gardes sur carte connue sont deja tres souvent corrects.

Cela suggere que le blocage n'est pas seulement:

```text
"sur qui jouer Pretre ?"
```

mais plutot:

```text
"quelle information restera exploitable jusqu'a mon prochain coup ?"
"quand faut-il jouer Pretre plutot que l'autre carte ?"
"comment convertir l'info Pretre en Prince/Baron/Roi, pas seulement en Garde ?"
```

## Decision

On garde les scripts et les logs, mais on ne valide pas la V1 heuristique comme
module Step5.

## V1 Apprise Par Teacher CRN

Le dossier contenait deja un dataset teacher CRN:

```text
step5_execution_heads/cards/priest/datasets/priest_target_teacher_160x12.json
```

Synthese du dataset exploitable:

| Etats utilisables | Best != Step3 | Regret clair | Mean score regret | Mean win regret |
|---:|---:|---:|---:|---:|
| 121 | 41.88% | 10.62% | 0.0359 | 0.0333 |

On a donc entraine une vraie tete rapide:

```bash
python3 step5_execution_heads/train_target_head.py \
  --kind priest_target \
  --dataset step5_execution_heads/cards/priest/datasets/priest_target_teacher_160x12.json \
  --output step5_execution_heads/cards/priest/checkpoints/priest_target_head_v1.pth
```

Validation interne de la tete:

| Split | Top1 | Override agreement | Pred override |
|---|---:|---:|---:|
| train | 72.16% | 75.26% | 23.71% |
| validation | 37.50% | 54.17% | 25.00% |

La validation est faible: le dataset est petit et le signal Pretre est bruite.

Evaluation arena 1000/config:

| Politique | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 52.40% | 44.00% | 37.60% | 32.70% | 0.38400 |
| Pretre random | 52.10% | 43.00% | 37.70% | 32.40% | 0.38080 |
| Step3 + tete Pretre | 52.30% | 44.40% | 37.50% | 33.00% | 0.38560 |

Conditionnel sur les parties ou Pretre est joue:

| Politique | Winrate pondere |
|---|---:|
| Step3 rapide | 48.75% |
| Pretre random | 47.71% |
| Step3 + tete Pretre | 49.10% |

Conditionnel sur les parties ou la cible du Pretre etait vraiment
randomisable:

| Politique | Winrate pondere |
|---|---:|
| Step3 rapide | 43.63% |
| Pretre random | 42.21% |
| Step3 + tete Pretre | 44.10% |

Cette tete apprise bat Step3 et le controle random, mais seulement de maniere
tres legere: `+0.00160` composite.

Decision: **candidat positif mais non valide**. On le garde comme checkpoint de
travail, pas comme tete Step5 officielle.

La prochaine version doit etre plus proche d'une tete de valeur:

1. Collecter des etats ou Step3 joue Pretre.
2. Evaluer les cibles par rollouts CRN, pas uniquement par score heuristique.
3. Mesurer la valeur future de l'info:
   - Guard exact plus tard;
   - Prince sur Princesse ou grosse carte;
   - Baron evite/gagne grace a l'information;
   - Roi evite ou choisi grace a l'information.
4. Entrainer/brancher uniquement si l'oracle trouve un regret clair.

Conclusion courte: **on a commence le Pretre, la tete apprise donne un petit
signal positif, mais le Pretre demande une V2 orientee valeur future/exploitation
de l'information.**

## Variante Pre-Garde

Suite a l'hypothese suivante:

> utiliser le Pretre comme si on preparait un Garde, uniquement sur des cartes
> pas encore connues, en priorisant les soupcons de grosses valeurs.

On a ajoute la variante `Pretre pre-Garde` dans:

```text
step5_execution_heads/cards/priest/evaluate_priest_specialist.py
```

Run:

```bash
python3 step5_execution_heads/cards/priest/evaluate_priest_specialist.py \
  --games 1000 \
  --seed-start 3000000 \
  --retarget-margin 0.00 \
  --include-guard-probe \
  --output priest_guard_probe_eval_1000_seed3000000.json \
  --markdown priest_guard_probe_eval_1000_seed3000000.md
```

Resultats:

| Politique | Composite | Pretre en main | Pretre joue | Pretre sur inconnu | Carte 7+ vue | Princesse vue | Pretre->Garde hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step3 rapide | 0.39620 | 52.27% | 52.76% | 96.48% | 27.21% | 9.09% | 96.74% |
| Pretre target random | 0.39900 | 51.75% | 52.18% | 95.32% | 27.09% | 9.10% | 93.57% |
| Pretre pre-Garde | 0.39570 | 51.75% | 52.18% | 97.01% | 27.36% | 9.56% | 93.25% |

Lecture:

- La variante fait bien ce qu'on voulait mesurer:
  - plus de cibles inconnues;
  - legerement plus de cartes 7+ vues;
  - plus de Princesses vues;
  - plus de cartes non-Garde donc potentiellement devinables ensuite.
- Mais elle n'ameliore pas le jeu:
  - composite `-0.00050` vs Step3;
  - Pretre->Garde hit baisse de `96.74%` a `93.25%`.

Conclusion: l'idee est strategiquement plausible, mais sous cette forme elle ne
marche pas encore. Le modele ne manque pas seulement de voir une grosse carte:
il faut que l'information soit encore exploitable au moment ou il rejoue. Pour
la V2, il faut donc scorer le Pretre par **valeur future realisee** plutot que
par soupcon statique de grosse carte.
