# Step3 v2 DAgger Archive

Date: 25 avril 2026.

Ces checkpoints sont gardes de cote parce qu'ils contiennent le premier signal
positif de DAgger rapide, mais ils ne sont pas encore declares champion final.

## Checkpoints

```text
step3_v2_dagger_attempt1_iter1_candidate_fast.pth
```

Checkpoint rapide prioritaire a evaluer proprement. Il correspond a:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Inference prevue pour la validation propre:

```text
override_margin = 0.10
verify_rollouts = 0
```

```text
step3_v2_dagger_attempt1_iter2_overrides_warning.pth
```

Checkpoint garde comme diagnostic. Il apprend des donnees on-policy
supplementaires mais augmente trop le taux d'overrides, symptome probable
d'oubli catastrophique / interference d'objectifs.

## Interpretation

Le signal DAgger existe, mais les evaluations 1000/config etaient trop bruitees
pour tuner finement le seuil. La prochaine validation serieuse est donc une
evaluation 5000/config du candidat iter1 a marge 0.10.

La prochaine boucle DAgger doit ajouter une contrainte de type trust region ou
KL vers la politique precedente/Step2 pour empecher les sur-corrections.
