# Step5 - Conclusion Phase A Teacher/Audit

Date: 2026-04-25.

## Question

Step4 a montre que le Step3 rapide sait globalement jouer fort, mais que
certaines cartes semblent peu maitrisees dans leur execution fine:

- Roi: choix de la cible d'echange;
- Baron: choix de cible, surtout avec petite carte gardee;
- Pretre: choix de la cible a regarder;
- Chancelier: choix de la carte gardee et ordre des cartes remises.

La question de cette phase A etait:

> Quand Step3 decide deja naturellement de jouer ces cartes, existe-t-il des
> alternatives d'execution clairement meilleures selon un oracle rollout CRN ?

## Ce Qui A Ete Fait

Un teacher/audit a ete implemente:

```text
step5_execution_heads/collect_execution_teacher.py
```

Il ne force jamais le modele a jouer une carte. Il observe uniquement des etats
naturels du Step3 rapide, puis compare les executions legales par rollouts
apparies CRN.

Run principal:

```bash
python3 -m step5_execution_heads.collect_execution_teacher \
  --games 500 \
  --max-states-per-kind 40 \
  --rollouts-per-action 12 \
  --seed-start 510000 \
  --dataset execution_teacher_initial_40x12.json \
  --output execution_teacher_initial_40x12_report.json \
  --markdown execution_teacher_initial_40x12_report.md \
  --run-log step5_execution_heads/logs/2026-04-25_execution_teacher_initial_40x12.md
```

Verification technique:

- `python3 -m py_compile step5_execution_heads/collect_execution_teacher.py`
- smoke-test `30` games, `3` etats max/type, `2` rollouts/action;
- run initial complet `500` games, `40` etats/type, `12` rollouts/action.

## Resultats

Rapport brut:

```text
step5_execution_heads/reports/execution_teacher_initial_40x12_report.md
```

Synthese:

| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chancelier - choix carte/ordre | 40 | 67.50% | 37.50% | 2.50% | 0.00% | 0.1199 | 0.1104 |
| Baron avec carte faible - cible | 40 | 65.00% | 20.00% | 15.00% | 87.50% | 0.0715 | 0.0667 |
| Roi - cible | 40 | 30.00% | 17.50% | 47.50% | 37.50% | 0.0615 | 0.0583 |
| Baron - cible | 40 | 32.50% | 15.00% | 32.50% | 27.50% | 0.0562 | 0.0521 |
| Pretre - cible | 40 | 40.00% | 12.50% | 27.50% | 15.00% | 0.0382 | 0.0354 |

Definitions:

- `Best != modele`: l'oracle rollout prefere une autre execution en moyenne.
- `Regret clair`: l'alternative passe les seuils de stabilite du teacher.
- `Execution forcee`: une seule execution legale etait disponible.
- `Carte forcee`: le modele n'avait pas d'autre carte/action principale legale
  au moment ou il a joue cette carte.

## Lecture

La phase A est un succes, parce que le teacher trouve du regret exploitable.

Le signal le plus fort est le Chancelier. C'est exactement la carte que l'on
soupconnait: l'ablation random montrait peu de chute globale, mais l'oracle
montre qu'il existe localement beaucoup de choix meilleurs. Cela veut dire que
le Step3 actuel ne convertit pas assez bien le potentiel du Chancelier.

Le Baron faible est aussi interessant, mais il faut etre prudent: `87.50%` des
etats `baron_low_target` etaient des cas ou la carte jouee etait forcee. Donc le
probleme n'est pas toujours "il aurait du ne pas jouer Baron"; souvent il est
deja coince avec le Baron et il faut optimiser la cible ou minimiser les degats.

Le Roi a un signal reel malgre l'ablation globale neutre. L'explication la plus
probable est que le Roi est rare et souvent contraint: randomiser la cible ne
change pas assez d'evenements pour bouger le composite, mais dans les etats ou
plusieurs cibles comptent, l'oracle trouve parfois une vraie meilleure cible.

Le Pretre est le signal le plus faible. Il n'est pas nul, mais il passe apres
Chancelier, Baron faible et Roi.

## Decision

La partie Step5 phase A est terminee.

On ne doit pas entrainer une tete globale tout de suite sur tous les labels avec
la meme force. L'ordre rationnel est:

1. Chancelier: premiere tete ciblee, car regret clair le plus frequent et carte
   non-forcee.
2. Baron faible: tete de cible/risque, en separant les cas forcees des cas
   optionnels.
3. Roi: tete de cible prudente, parce que l'effet global est rare mais le
   regret local existe.
4. Pretre: audit plus large avant entrainement fort.

## Statut Des Criteres De Succes Step5

| Critere | Statut |
|---|---|
| Teacher trouve du regret exploitable | Valide |
| Dataset filtre disponible | Partiel: dataset initial cree |
| Tetes rapides entrainees | Pas encore |
| Gain composite `+0.010` vs Step3 rapide | Pas encore teste |
| Pas de regression `3H` | Pas encore teste |
| Inference sans rollout | Objectif maintenu |

Conclusion: Step5 n'est pas encore terminee comme etape de modele, mais sa
premiere partie est terminee proprement. On a maintenant une cible d'entrainement
justifiee par donnees, au lieu d'ameliorer "au feeling".

