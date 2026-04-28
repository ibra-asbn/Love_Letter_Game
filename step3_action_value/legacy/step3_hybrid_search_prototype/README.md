# Step 4 - Joueurs Hybrides Avec Recherche A L'Inference

Date: 25 avril 2026.

Objectif: conserver les joueurs qui utilisent une recherche locale au moment de
jouer. Ils sont plus lents que Step3 rapide, mais servent de reference forte
pour jouer contre des humains et pour guider de futures distillations.

## Step4 v0 Valide - Step3 Advantage Verify16

Premier joueur hybride valide:

```text
step4_hybrid_champion/checkpoints/step4_v0_step3_advantage_verify16.pth
step4_hybrid_champion/step4_v0_verified_step3_advantage_verify16.json
```

Inference:

```text
Step2 + tete advantage Step3 v2 + verification CRN locale 16 rollouts
```

Validation sur trois seeds independants, `1000` parties par composition:

| Seed | Hybride verify16 | Step2 | Delta |
|---|---:|---:|---:|
| 134000 | 0.28230 | 0.26720 | +0.01510 |
| 135000 | 0.27890 | 0.26580 | +0.01310 |
| 136000 | 0.27890 | 0.26330 | +0.01560 |
| Moyenne | 0.28003 | 0.26543 | +0.01460 |

Ce joueur est archive comme **Step4 v0**, meme si son code d'evaluation vit
dans `step3_action_value/evaluate_advantage_head_v2.py`.

## Prototype Historique Belief-Search

Le joueur Step4 historique garde le modele Step2 comme intuition de base, puis
reflechit sur les coups tactiques:

1. l'actor Step2 choisit une action par defaut;
2. le belief du meme modele donne les probabilites de cartes adverses;
3. le belief sert a proposer de meilleurs candidats;
4. les rollouts determinises avec ce belief tranchent;
5. l'action de l'actor est remplacee seulement si la marge est claire.

Checkpoint utilise:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Script principal:

```bash
python3 -m step4_hybrid_champion.evaluate_hybrid_search
```

## Resultat Principal

Validation exploratoire sur `300` parties par configuration, seed `802000`,
avec `12` rollouts par action candidate. Ces chiffres sont conserves comme
trace historique, pas comme validation finale.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 brut | 52.00% | 32.00% | 26.33% | 16.00% | 0.25900 |
| Step3 rollout uniforme | 56.33% | 46.67% | 29.00% | 25.00% | 0.33667 |
| Step4 hybride belief-search | 54.67% | 43.00% | 32.67% | 26.00% | 0.34267 |

Lecture:

- Sur ce run court, le prototype bat nettement Step2: `+8.37` points de composite.
- Sur ce run court, il bat legerement Step3 uniforme: `+0.60` point de composite.
- Le gain du belief n'est pas enorme face au search uniforme, mais il existe
  sur cette validation.
- Le plus gros saut vient du fait d'assumer une vraie recherche tactique au
  moment de jouer.

## Commandes

Step4 vs Step2:

```bash
python3 -m step4_hybrid_champion.evaluate_hybrid_search \
  --checkpoint step2_retarget_distilled_attempt1.pth \
  --games 300 \
  --seed-start 802000 \
  --rollouts-per-action 12 \
  --max-actions 14 \
  --override-margin 0.12 \
  --compare-step2 \
  --output step4_hybrid_eval_300_seed802000.json \
  --run-log step4_hybrid_champion/logs/2026-04-25_step4_hybrid_eval_300_seed802000.md
```

Step3 uniforme de controle, memes seeds:

```bash
python3 -m step3_action_value.evaluate_rollout_guided \
  --checkpoint step2_retarget_distilled_attempt1.pth \
  --games 300 \
  --seed-start 802000 \
  --categories guard priest spy king prince chancellor_card chancellor_choice baron \
  --rollouts-per-action 12 \
  --max-actions 14 \
  --override-margin 0.12 \
  --player0-continuation heuristic \
  --output step3_uniform_broad_eval_300_seed802000_for_step4.json \
  --run-log step4_hybrid_champion/logs/2026-04-25_step3_uniform_broad_eval_300_seed802000_for_step4.md
```

## Logs Et Rapports

```text
step4_hybrid_champion/logs/2026-04-25_step4_hybrid_eval_300_seed802000.md
step4_hybrid_champion/reports/step4_hybrid_eval_300_seed802000.json
step4_hybrid_champion/logs/2026-04-25_step3_uniform_broad_eval_300_seed802000_for_step4.md
step3_action_value/reports/step3_uniform_broad_eval_300_seed802000_for_step4.json
```

## Verdict

Le Step4 v0 officiellement conserve est maintenant le joueur `verify16` documente
en haut de ce fichier.

Le prototype belief-search historique reste prometteur mais seulement valide sur
un run court de `300` parties par composition. Il ne doit pas etre utilise comme
base officielle tant qu'il n'a pas ete confirme sur une evaluation longue.

## Suite

- Brancher d'abord Step4 v0 `verify16` dans `love_letter_web` et/ou `play_vs_agent`.
- Valider tout nouveau Step4 sur au moins `5000` parties par configuration.
- Ajouter des diagnostics humains: Gardes justes, Barons gagnes/perdus,
  Princes sur Princesse, qualite du Chancelier.
- Tuner le budget temps: rollouts adaptatifs selon l'importance du coup.
- Ensuite seulement, distiller Step4 dans une tete rapide si necessaire.
