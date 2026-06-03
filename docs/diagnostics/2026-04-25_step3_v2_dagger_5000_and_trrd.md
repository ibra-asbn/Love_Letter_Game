# Diagnostic Step3 v2 DAgger 5000 + Preparation TRRD/KL

Date: 25 avril 2026.

## Contexte

Le joueur hybride `verify16` est valide et archive comme branche Step3 hybride:

```text
step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth
step3_action_value/hybrid_verified/step3_hybrid_verify16.json
```

Il ne s'agit pas d'une Step4 de pipeline. L'objectif rapide de Step3 v2 reste
un modele autonome, sans rollout a
l'inference. La tete rapide `attempt2_strict` avait un vrai signal mais souffrait
de decalage de distribution: elle etait entrainee sur trajectoires Step2, puis
visitait ses propres etats apres override.

## Ce Qui A Ete Fait

Ajout de la boucle DAgger:

```text
step3_action_value/train_advantage_dagger_v2.py
```

Principe:

1. jouer la tete rapide seule, sans verify;
2. collecter les etats qu'elle cree elle-meme;
3. relabeliser ces etats hors-ligne avec le meme oracle CRN strict;
4. agreger ces donnees avec une reference Step2;
5. reentrainer la tete advantage.

Checkpoints archives:

```text
step3_action_value/checkpoints/dagger_archive/step3_v2_dagger_attempt1_iter1_candidate_fast.pth
step3_action_value/checkpoints/dagger_archive/step3_v2_dagger_attempt1_iter2_overrides_warning.pth
```

Lecture:

- `iter1` est le candidat rapide propre;
- `iter2` est conserve comme diagnostic, car il apprend plus d'etats on-policy
  mais augmente trop le taux d'overrides.

## Evaluation 5000 Officielle

Candidat fixe:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Inference:

```text
override_margin = 0.10
verify_rollouts = 0
```

Commande:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_dagger_attempt1_iter1.pth \
  --games 5000 \
  --seed-start 140000 \
  --override-margin 0.10 \
  --compare-baseline \
  --example-limit 0 \
  --output step3_advantage_v2_dagger_attempt1_iter1_eval_5000_seed140000_m010.json \
  --run-log step3_action_value/logs/2026-04-25_step3_advantage_v2_dagger_attempt1_iter1_eval_5000_seed140000_m010.md
```

Resultats:

| Composition | DAgger iter1 rapide | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.96% | 52.08% | +1.88 pts |
| vs 1H+2R | 39.72% | 38.38% | +1.34 pts |
| vs 2H+1R | 26.42% | 25.02% | +1.40 pts |
| vs 3H | 14.90% | 15.12% | -0.22 pt |
| Composite | 0.27226 | 0.26438 | +0.00788 |

Le taux d'override est autour de `18%` a `20%` des decisions selon la
composition. Le gain composite est positif sur 20 000 parties, donc le signal
DAgger rapide est reel. En revanche, la legere regression vs `3H` dit que ce
n'est pas encore un champion rapide final contre les heuristiques.

## Pourquoi Iter2 N'Est Pas Retenu

L'iteration 2 a augmente le taux d'overrides et a donne un comportement plus
agressif. C'est compatible avec un oubli catastrophique / interference
d'objectifs: en apprenant les nouveaux labels on-policy, la tete oublie une
partie du comportement conservateur utile de Step2 et sur-corrige.

Conclusion: DAgger fonctionne, mais il doit etre contraint.

## Preparation TRRD / KL

La contrainte est maintenant preparee dans:

```text
step3_action_value/train_advantage_head_v2.py
step3_action_value/train_advantage_dagger_v2.py
```

Nouvelle regularisation:

```text
step2_trust_region_kl(scores, targets, valid, model_index, args)
```

Idee:

- transformer les scores de la tete en distribution sur les actions candidates;
- definir une distribution de reference Step2 qui met presque toute la masse
  sur l'action Step2;
- penaliser la KL quand la tete s'eloigne trop de cette reference;
- reduire cette penalite seulement quand le label CRN indique un avantage
  positif massif et fiable.

Arguments ajoutes:

```text
--trust-region-kl-weight
--trust-region-temperature
--trust-region-step2-epsilon
--trust-region-break-advantage
--trust-region-break-weight
```

Smoke test effectue:

```bash
python3 -m step3_action_value.train_advantage_dagger_v2 \
  --initial-checkpoint step3_advantage_v2_dagger_attempt1_iter1.pth \
  --output-prefix step3_advantage_v2_dagger_trrd_smoke \
  --iterations 1 \
  --include-step2-reference \
  --reference-collect-games 200 \
  --reference-states-per-category-config 1 \
  --onpolicy-collect-games 200 \
  --onpolicy-states-per-category-config 1 \
  --rollouts-per-action 4 \
  --epochs 1 \
  --trust-region-kl-weight 1.5
```

Le smoke passe. La perte logge `trust_region_kl` et le mini-entrainement reduit
fortement le taux d'override predit, ce qui est le comportement attendu.

## Verdict

Succes partiel solide:

- Step3 hybride `verify16` est sauvegarde comme reference hybride valide.
- Step3 v2 DAgger rapide a maintenant un gain positif mesure sur `5000`
  parties par composition: `+0.788` point composite vs Step2.
- Ce n'est pas encore le champion rapide final, car le score vs `3H` regresse
  legerement.

Prochaine boucle:

```bash
python3 -m step3_action_value.train_advantage_dagger_v2 \
  --initial-checkpoint step3_advantage_v2_dagger_attempt1_iter1.pth \
  --output-prefix step3_advantage_v2_dagger_trrd_attempt1 \
  --iterations 1 \
  --include-step2-reference \
  --reference-states-per-category-config 10 \
  --onpolicy-states-per-category-config 8 \
  --rollouts-per-action 24 \
  --epochs 10 \
  --trust-region-kl-weight 1.5 \
  --trust-region-temperature 0.25 \
  --trust-region-step2-epsilon 0.02 \
  --trust-region-break-advantage 0.20 \
  --trust-region-break-weight 0.15
```

Evaluation suivante: uniquement `5000` parties par composition pour le candidat
retenu, sans tuning de seuil sur des seeds de `1000`.
