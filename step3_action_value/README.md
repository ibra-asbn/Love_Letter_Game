# Step 3 - Action-Value

Date de rangement: 2026-04-25.

Objectif: partir du joueur Step2, puis apprendre quand une autre action a plus
de valeur que son action par defaut. La Step3 contient maintenant deux branches:

- **rapide autonome**: une tete `advantage(s, a)` joue sans rollout a
  l'inference;
- **hybride verifiee**: la tete propose une correction, puis un petit
  verificateur CRN a 16 rollouts accepte ou rejette cette correction.

Il n'y a pas de Step4 active dans cette pipeline. Ce qui avait ete appele
`Step4 v0 verify16` est requalifie ici comme **Step3 hybride verify16**.

## Organisation Active

```text
step3_action_value/
  common.py                         # utilitaires partages de la Step3 active
  mini_rollout_probe.py             # primitives de determinization/probe
  evaluate_rollout_guided.py        # oracle lent par rollouts
  train_advantage_head_v2.py        # collecte CRN + entrainement advantage
  train_advantage_dagger_v2.py      # DAgger on-policy pour tete rapide
  evaluate_advantage_head_v2.py     # evaluation rapide ou hybride verify
  hybrid_verified/
    step3_hybrid_verify16.json      # fiche du meilleur Step3 hybride valide
  checkpoints/
    step3_advantage_v2_attempt2_strict.pth
    step3_advantage_v2_dagger_attempt1_iter1.pth
    step3_advantage_v2_dagger_attempt1_iter2.pth
    dagger_archive/
    hybrid_verify16/
      step3_hybrid_verify16.pth
  legacy/
    obsolete_heads/                 # anciennes tentatives gardees pour trace
    checkpoints/                    # checkpoints des anciennes tentatives
    step3_hybrid_search_prototype/  # ancien dossier Step4, archive historique
```

## Branche Hybride Validee

Checkpoint:

```text
step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth
```

Fiche:

```text
step3_action_value/hybrid_verified/step3_hybrid_verify16.json
```

Inference:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_attempt2_strict.pth \
  --override-margin 0.10 \
  --verify-rollouts 16 \
  --verify-min-win-delta 0.125 \
  --verify-min-score-delta 0.05 \
  --verify-t-threshold 0.75 \
  --compare-baseline
```

Validation sur trois seeds independants, `1000` parties par composition:

| Seed | Step3 hybride verify16 | Step2 | Delta |
|---|---:|---:|---:|
| 134000 | 0.28230 | 0.26720 | +0.01510 |
| 135000 | 0.27890 | 0.26580 | +0.01310 |
| 136000 | 0.27890 | 0.26330 | +0.01560 |
| Moyenne | 0.28003 | 0.26543 | +0.01460 |

Moyenne par composition:

| Composition | Step3 hybride verify16 | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.30% | 51.57% | +1.73 pts |
| vs 1H+2R | 38.87% | 37.50% | +1.37 pts |
| vs 2H+1R | 26.47% | 24.73% | +1.73 pts |
| vs 3H | 17.40% | 16.17% | +1.23 pts |

Statut: succes Step3 hybride. C'est plus fort que Step2, mais pas instantane:
il utilise de la recherche locale a l'inference.

## Branche Rapide DAgger

Checkpoint principal:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Archive explicite:

```text
step3_action_value/checkpoints/dagger_archive/step3_v2_dagger_attempt1_iter1_candidate_fast.pth
step3_action_value/checkpoints/dagger_archive/step3_v2_dagger_attempt1_iter2_overrides_warning.pth
```

Evaluation officielle du candidat rapide `iter1`, sans verify, sur `5000`
parties par composition:

| Composition | DAgger iter1 | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.96% | 52.08% | +1.88 pts |
| vs 1H+2R | 39.72% | 38.38% | +1.34 pts |
| vs 2H+1R | 26.42% | 25.02% | +1.40 pts |
| vs 3H | 14.90% | 15.12% | -0.22 pt |
| Composite | 0.27226 | 0.26438 | +0.00788 |

Statut: signal positif mais pas encore champion rapide final. Le modele gagne
en moyenne contre Step2, mais regresse legerement contre `3H`.

## Prochaine Correction Rapide

L'iteration DAgger suivante doit utiliser la contrainte KL/trust-region deja
codee dans `train_advantage_head_v2.py` et exposee par
`train_advantage_dagger_v2.py`.

Arguments:

```text
--trust-region-kl-weight
--trust-region-temperature
--trust-region-step2-epsilon
--trust-region-break-advantage
--trust-region-break-weight
```

Hypothese: le modele rapide doit rester proche de Step2 dans les etats
incertains, et ne casser cette region de confiance que quand l'avantage CRN est
massif.

## Archives

Les anciennes tentatives ont ete deplacees dans `legacy/`:

- distillation directe des meilleurs coups rollout;
- heads de regret override;
- pairwise rankers;
- prototype historique actor + belief + search anciennement appele Step4.

Elles restent utiles pour comprendre les echecs, mais ne font plus partie de la
pipeline active.
