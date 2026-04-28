# Step5 - Pretre

Date: 2026-04-26.

## Objectif

Entrainer une tete locale pour choisir la cible du Pretre. Le Pretre ne gagne
pas directement une manche: il gagne de l'information. Le bon critere n'est donc
pas seulement la carte vue, mais la valeur future de cette information,
notamment pour les Gardes, Barons, Princes et Rois suivants.

## Critere De Succes

- Ameliorer Step3 rapide au composite fair seat-rotated ou produire un signal
  tactique clair `Pretre -> Garde`.
- Faire mieux que le controle `priest_target_random`.
- Ne pas apprendre du bruit si le regret rollout reste trop faible.

## Premiere Version

`evaluate_priest_specialist.py` implemente une V1 volontairement locale:

- `baseline`: Step3 rapide;
- `priest_target_random`: controle qui randomise seulement la cible quand Step3
  joue Pretre;
- `Pretre V1`: retarget vers la cible dont l'information semble la plus utile.

Le score V1 favorise:

- une carte encore inconnue;
- une distribution incertaine;
- une probabilite elevee de carte 7+ ou Princesse;
- une information exploitable avec la carte conservee, surtout `Pretre + Garde`.

## Resultat Initial

Rapport:

```text
step5_execution_heads/cards/priest/reports/2026-04-26_priest_v1_initial_diagnostic.md
```

La V1 heuristique target-only n'est pas validee. Une tete apprise sur le dataset
teacher CRN existe:

```text
step5_execution_heads/cards/priest/checkpoints/priest_target_head_v1.pth
```

Elle bat legerement Step3 et le controle random sur 1000/config, mais le gain
est trop faible pour devenir une tete officielle:

```text
Step3 rapide:        0.38400 composite
Pretre random:       0.38080 composite
Step3 + tete Pretre: 0.38560 composite
```

## Statut

V1 candidate positive legere, non validee.
