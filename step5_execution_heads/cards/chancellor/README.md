# Step5 - Chancelier

Date: 2026-04-26.

## Statut

**Succes V1 valide.**

La tete Chancelier corrige uniquement l'effet Chancelier lorsque Step3 rapide a
deja decide de jouer Chancelier. Elle ne fait aucun rollout a l'inference.

Checkpoint local:

```text
checkpoints/chancellor_head_v1.pth
```

Dataset local:

```text
datasets/chancellor_teacher_v1_balanced_240x12.json
```

## Resultat Court

| Validation | Step3 rapide | Chancelier random | Tete Chancelier |
|---|---:|---:|---:|
| 1000/config seed 860000 | 0.39160 | 0.39200 | 0.40710 |
| 1000/config seed 870000 | 0.39340 | 0.39820 | 0.40830 |

Conclusion: le gain vient bien d'une competence apprise et pas seulement du
fait de remplacer des choix Chancelier au hasard.

## Limite

La tete apprend surtout a garder la meilleure carte. Elle ne semble pas encore
maitriser la planification fine de pioche, c'est-a-dire remettre volontairement
une bonne carte sous la pioche pour la recuperer plus tard ou manipuler le tour
d'un adversaire.

## Rapports

```text
reports/2026-04-26_chancellor_v1_conclusion.md
reports/chancellor_head_v1_eval_1000_seed860000.md
reports/chancellor_head_v1_eval_1000_seed870000.md
reports/chancellor_random_control_1000_seed860000.md
reports/chancellor_random_control_1000_seed870000.md
```
