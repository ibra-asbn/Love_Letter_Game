# Love Letter RL

Etat historique detaille au **26 avril 2026**. Etat documentaire consolide le
**3 juin 2026**.

Pour lire le projet dans l'ordre, commencer par:

- `docs/project_journal_fr.md` pour le recap canonique etape par etape;
- `docs/github_handoff_fr.md` pour le nettoyage et la publication GitHub;
- `docs/linkedin_post_fr.md` pour le brouillon de communication;
- `love_letter_web/README.md` pour lancer l'app web.

Le point d'arret actuel est volontaire: conserver le champion `champion_cbp`,
mettre la documentation et le repo au propre, puis publier une synthese.

Objectif: construire un agent de reinforcement learning fort pour jouer a **Love Letter**, puis fournir une interface jouable contre le meilleur modele.

## Diagnostic Court

Le projet est maintenant nettoye autour d'une base plus saine.

Ce qu'on sait:

- Love Letter est exploitable par une politique meilleure que le hasard.
- Le meilleur checkpoint historique conserve reste `models/checkpoints/curriculum_phase1.pth`.
- Les meilleurs joueurs de la nouvelle pipeline sont maintenant les deux Step3:
  Step3 rapide DAgger autonome (`0.39750` composite seat-rotated post-fix
  regles) et Step3 hybride verify16 avec recherche locale (`0.39690`). L'ecart
  entre eux est trop faible pour declarer une hierarchie definitive.
- L'ancienne arena `player_0 only` etait biaisee par le ciblage absolu de
  `HeuristicBot`; elle est conservee comme hard mode. Une nouvelle fair arena
  avec `HeuristicBot(shuffle_targets=True)` confirme que Step2 et Step3 restent
  au-dessus de l'heuristique au composite.
- Le 25 avril 2026, le moteur a ete audite contre les regles: corrections sur
  Baron/Roi/Servante et validation par `scripts/debug/check_rules_conformance.py`.
- Le modele Step2 bat `HeuristicBot` sur nos arenas de reference; Step3 ajoute une correction action-value stabilisee par rollouts apparies.
- Step5 vient de produire une premiere tete d'execution validee: Chancelier V1
  ameliore Step3 rapide de `+0.0155` puis `+0.0149` composite sur deux blocs de
  `1000` parties par composition, sans rollout a l'inference.
- Les essais `belief-conditioned` ont montre que le belief est utile, surtout quand il pilote les cibles, les devinettes et les determinizations.
- L'ancien empilement de scripts PPO/Tianshou a ete supprime pour repartir sur un pipeline plus controlable.

## Arborescence

```text
love_letter/
  engine.py                  # moteur PettingZoo/Gym du jeu
  paths.py                   # chemins centraux du projet
  belief_actor.py            # architecture actor + belief conditionne
  belief_policy.py           # chargement/inference des checkpoints
  bots/heuristic.py          # bot heuristique de reference
  gameplay/play_vs_agent.py  # partie console humain vs modele

scripts/
  training/
    collect_heuristic_data.py     # genere le dataset d'imitation
    distill_belief_retarget.py    # apprend a l'actor a internaliser le retarget belief
    pretrain_belief_actor.py      # pretraining supervised actor + belief
    train_belief_ppo.py           # fine-tuning PPO custom sans Tianshou
  evaluation/
    evaluate_baselines.py         # random vs heuristic sur beaucoup de parties
    evaluate_belief_counterfactual.py # teste si l'actor exploite bien le belief
    evaluate_models.py            # matrice d'evaluation des checkpoints
    diagnose_model.py             # diagnostic tactique detaille d'un modele
    measure_belief.py             # accuracy du belief head
  debug/
    check_engine_invariants.py    # sanity checks du moteur
    check_rules_conformance.py    # tests executables des regles Love Letter

docs/
  love_letter_rules_fr.md         # reference locale des regles utilisees

models/checkpoints/
  curriculum_phase1.pth              # reference historique conservee
  belief_conditioned_bc.pth          # warm start BC pour le futur pipeline
  belief_conditioned_ppo_final.pth   # experience belief-conditioned non championne

data/
  heuristic_dataset.pkl              # dataset BC issu du bot heuristique

diagnostics/
  rapports tactiques et diagnostics modeles

logs/evaluations/
  resultats bruts d'evaluation

interlude_heuristic_comparison/
  evaluate_interlude_arena.py         # arena HeuristicBot vs Step2 vs Step3
  evaluate_fair_arena.py              # meme arena, mais heuristiques sans focus player_0
  evaluate_rotating_tactical_arena.py # rotation des sieges + diagnostics tactiques
  seat_bias_probe.md                  # diagnostic du biais de ciblage historique
  README.md                           # verdict: modeles actuels > heuristique

love_letter_web/
  streamlit_app.py                   # interface jouable
  backend/main.py                    # prototype FastAPI

step1_heuristic_mastery/
  collect_teacher_sequences.py        # collecte sequencee depuis HeuristicBot
  train_heuristic_student.py          # imitation recurrente avec split train/val/test
  compare_student_teacher.py          # diagnostic action par action
  evaluate_heuristic_mastery.py       # arena student vs heuristic/random

step2_rl_finetune/
  evaluate_step2.py                   # evaluations step2 vs random/heuristic
  train_step2_ppo.py                  # tentative PPO depuis le student heuristique
  checkpoints/                        # checkpoints locaux ignores par Git

step3_action_value/
  common.py                           # utilitaires partages Step3 active
  mini_rollout_probe.py               # primitives de probe/determinization
  evaluate_rollout_guided.py          # oracle lent action-value/search
  train_advantage_head_v2.py          # tete advantage avec labels CRN apparies
  train_advantage_dagger_v2.py        # DAgger on-policy pour tete rapide autonome
  evaluate_advantage_head_v2.py       # Step3 v2 verifie: tete + verification CRN
  hybrid_verified/                    # fiche du Step3 hybride verify16
  legacy/                             # anciennes tentatives archivees

step4_weakness_analysis/
  CARD_TAXONOMY.md                     # familles de cartes figees pour audit
  cluster_step3_card_archetypes.py     # clustering Step3 par cartes/phases
  reports/                             # resultats de l'audit des faiblesses

step5_execution_heads/
  collect_execution_teacher.py          # teacher/audit des executions fines
  README.md                             # objectif, protocole, criteres de succes
  reports/                              # audits Roi/Baron/Pretre/Chancelier
```

## Baseline Statistique

Test lance le 24 avril 2026 avec `20 000` parties par politique.

Commande:

```bash
python3 scripts/evaluation/evaluate_baselines.py \
  --games 20000 \
  --output logs/evaluations/2026-04-24_baselines_random_vs_heuristic_20000.json
```

Resultats:

| Politique player_0 | Adversaires | Winrate | IC 95% winrate | Reward moyen | IC 95% reward |
|---|---|---:|---:|---:|---:|
| Random | 3 randoms | 29.92% | +/- 0.63 pt | 0.397 | +/- 0.009 |
| HeuristicBot | 3 randoms | 45.69% | +/- 0.69 pt | 0.677 | +/- 0.011 |

Lift de l'heuristique:

- `+15.77` points de winrate vs random.
- `x1.53` en winrate.
- `+0.280` reward moyen.
- `x1.71` en reward moyen.

Conclusion: le jeu est bien exploitable. Une politique simple, lisible, non-neuronale obtient un edge massif et statistiquement stable. C'est donc un bon terrain pour du RL, mais il faut une vraie chaine d'entrainement et d'evaluation.

## Etape 1 - Maitrise De L'Heuristique

Sous-dossier dedie:

```text
step1_heuristic_mastery/
```

But: construire un modele qui absorbe proprement `HeuristicBot` avant de relancer du RL. Cette etape sert de base stable: si le reseau ne sait meme pas reproduire l'heuristique, il est trop tot pour attendre de PPO qu'il la batte.

Checkpoint obtenu le 24 avril 2026:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Correction moteur importante faite pendant cette etape: pendant l'effet du Chancelier, l'observation expose maintenant les cartes disponibles et leur ordre de choix. Avant cela, l'action exacte `900/902/904` etait partiellement impossible a imiter, car l'information utile n'etait pas dans l'observation.

Entrainement imitation:

| Split | Action accuracy | Action loss | Belief accuracy |
|---|---:|---:|---:|
| Train | 99.92% | 0.0048 | 32.93% |
| Validation | 98.93% | 0.0313 | 31.43% |
| Test | 98.94% | 0.0292 | 30.62% |

Gap train-validation: `0.99` point. Pas de signal d'overfitting bloquant.

Comparaison action par action sur `129 858` decisions:

| Mesure | Resultat |
|---|---:|
| Exact action accuracy | 99.72% |
| Meme carte jouee | 99.95% |
| Garde exact | 99.76% |
| Baron exact | 99.80% |
| Prince exact | 99.98% |
| Chancelier exact | 99.97% |
| Choix Chancelier exact | 99.78% |

Evaluation arena player_0 sur `5 000` parties par configuration:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student heuristique | 47.68% | 32.86% | 21.40% | 11.72% | 0.22448 |
| HeuristicBot | 47.86% | 32.78% | 21.62% | 11.82% | 0.22556 |

Conclusion etape 1: succes pour l'imitation. Le student n'est pas encore meilleur que l'heuristique, mais il est maintenant au meme niveau statistique et suffisamment proche pour servir de warm start RL. La prochaine vraie question n'est plus "sait-il copier l'heuristique ?", mais "comment le faire depasser l'heuristique sans detruire ce socle ?".

Rapport detaille:

```text
diagnostics/2026-04-24_step1_heuristic_mastery.md
```

## Etape 2 - Battre L'Heuristique

Sous-dossier dedie:

```text
step2_rl_finetune/
```

Objectif: partir du student heuristique de l'etape 1 et obtenir un modele DL qui bat vraiment `HeuristicBot`.

Checkpoint retenu:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Ce qu'on a teste:

- PPO depuis le student heuristique: succes court, mais pas confirme sur evaluation longue;
- contre-factuel actor/belief: gros signal positif si les cibles/devinettes suivent le belief;
- distillation du retarget belief dans l'actor: succes confirme.

Confirmation longue sur `5000` parties par configuration:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.54% | 36.80% | 26.36% | 16.04% | 0.26738 |
| HeuristicBot | 46.02% | 33.48% | 21.36% | 11.30% | 0.22226 |

Gain composite: `+4.51` points contre `HeuristicBot`.

Diagnostic post-distillation:

| Mode | Composite 1000 parties/config |
|---|---:|
| Actor brut distille | 0.2634 |
| Retarget encore force | 0.2695 |
| Tactical force | 0.2777 |

Conclusion: l'etape 2 est reussie. Le modele bat l'heuristique et l'actor utilise beaucoup mieux le belief, surtout pour les Gardes, Barons, Princes et Rois. Il reste encore un potentiel tactique a exploiter.

Rapport detaille:

```text
diagnostics/2026-04-24_step2_rl_finetune.md
```

Benchmark de clarification contre le champion historique:

```text
diagnostics/2026-04-24_step2_vs_curriculum_benchmark.md
```

Lecture courte: `step2_retarget_distilled_attempt1.pth` bat `HeuristicBot`, mais ne bat pas encore `curriculum_phase1.pth`. Il est meilleur sur plusieurs gestes tactiques locaux, notamment Garde, Chancelier et Roi, mais curriculum garde une politique globale plus forte.

Mini-test action-value pour l'etape 3:

```text
diagnostics/2026-04-24_step3_mini_action_value_probe.md
```

Lecture courte: des rollouts sur quelques etats critiques donnent deja des labels utiles. Le signal est particulierement interessant sur Baron/Prince/Garde: l'enjeu n'est pas d'ecrire une regle, mais d'apprendre une valeur par action dans le contexte.

## Etape 3 - Action-Value / Search

Sous-dossier dedie:

```text
step3_action_value/
```

Objectif: verifier si une estimation de valeur par action peut ameliorer le
Step2 sans ajouter de nouvelles regles manuelles.

Clarification de nomenclature: il n'y a pas de Step4 active. La variante
anciennement appelee `Step4 v0 verify16` est maintenant classee comme
**Step3 hybride verify16**.

Resultat important: les deux premieres tentatives de distillation offline ont
echoue, mais la policy `rollout-guided` a reussi. Elle garde Step2 comme action
par defaut, puis evalue quelques actions tactiques par rollouts sur
`baron`, `guard`, `prince` et override seulement si la marge est claire.

Confirmation `1000` parties par configuration, seed `783000`:

| Policy | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 brut | 51.00% | 35.10% | 22.80% | 15.50% | 0.25160 |
| Step3 rollout-guided | 50.90% | 36.80% | 26.00% | 18.00% | 0.27450 |

Gain composite: `+2.29` points. Le gain vient surtout des compositions avec
heuristiques, donc la piste action-value/search est valide.

Limite nuancee: cette policy est trop lente pour generer des millions de
parties d'entrainement, mais elle est acceptable pour jouer contre un humain.
Elle reste donc un oracle/teacher de search, pas encore un checkpoint autonome.

Redo du 25 avril 2026: on a tente de distiller ce teacher dans une tete
`pairwise action-value ranker`. Le meilleur gain confirme est positif mais trop
faible pour declarer un nouveau champion autonome.

| Candidat distille | Composite 1000/config | Delta vs Step2 |
|---|---:|---:|
| Step2 baseline | 0.26950 | - |
| Pairwise early-stop attempt2 | 0.27920 | +0.00970 |
| Pairwise stage-weighted attempt4 | 0.27760 | +0.00810 |
| Pairwise guard+baron attempt4 | 0.27370 | +0.00420 |

Step3 hybride verify16 du 25 avril 2026: on a repris cette piste avec une tete
`advantage(s, a)` relative a l'action Step2, des labels par rollouts apparies
CRN, et surtout une verification locale avant chaque override. La tete seule
reste instable, mais le mode verifie est maintenant un succes.

Reglage retenu:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_attempt2_strict.pth \
  --override-margin 0.10 \
  --verify-rollouts 16 \
  --verify-min-win-delta 0.125 \
  --verify-min-score-delta 0.05 \
  --verify-t-threshold 0.75
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

Le verificateur a accepte `1409` overrides sur `5603` propositions verifiees
sur ces validations, soit `25.15%`. Les propositions rejetees sont aussi
importantes que les acceptations: c'est ce filtre qui transforme une tete
instable en joueur plus fort.

Conclusion de cette sous-etape: Step3 est reussie sous forme hybride. A ce
moment-la, ce n'etait pas encore une tete autonome ultra-rapide; la distillation
devait encore apprendre la decision finale `proposition + verification`, pas
seulement copier les rollouts bruts.

DAgger rapide du 25 avril 2026: on a ensuite collecte les etats crees par la
tete rapide elle-meme, puis relabelise ces etats hors-ligne avec le meme oracle
CRN strict. Le premier candidat autonome utile est:

```text
step3_action_value/checkpoints/dagger_archive/step3_v2_dagger_attempt1_iter1_candidate_fast.pth
```

Evaluation officielle sans verify, `5000` parties par composition:

| Composition | DAgger iter1 rapide | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.96% | 52.08% | +1.88 pts |
| vs 1H+2R | 39.72% | 38.38% | +1.34 pts |
| vs 2H+1R | 26.42% | 25.02% | +1.40 pts |
| vs 3H | 14.90% | 15.12% | -0.22 pt |
| Composite | 0.27226 | 0.26438 | +0.00788 |

Lecture: le signal rapide DAgger est reel, mais ce n'est pas encore le champion
rapide final car il regresse legerement contre `3H`. L'iteration 2 a augmente
trop le taux d'overrides: symptome probable d'oubli catastrophique. Une
penalite KL/trust-region vers Step2 est maintenant codee pour la prochaine
boucle.

Rapport detaille:

```text
diagnostics/2026-04-24_step3_action_value_search.md
diagnostics/2026-04-25_step3_pairwise_ranker_redo.md
diagnostics/2026-04-25_step3_v2_advantage_verified.md
diagnostics/2026-04-25_step3_v2_dagger_5000_and_trrd.md
diagnostics/2026-04-25_step3_reorganisation_no_step4.md
```

## Interlude - Sommes-Nous Vraiment Sous L'Heuristique ?

Sous-dossier:

```text
interlude_heuristic_comparison/
```

Question: faire passer `HeuristicBot`, Step2, Step3 rapide et Step3 hybride
dans la meme arena progressive (`3R`, `1H+2R`, `2H+1R`, `3H`).

Premier resultat en hard mode `player_0`, sur `3` seeds de `1000` parties par
composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| HeuristicBot | 46.40% | 32.57% | 21.50% | 11.83% | 0.22337 | +0.00000 |
| Step2 retarget | 51.57% | 37.50% | 24.73% | 16.17% | 0.26543 | +0.04207 |
| Step3 rapide DAgger | 53.87% | 38.27% | 26.97% | 16.63% | 0.27783 | +0.05447 |
| Step3 hybride verify16 | 53.30% | 38.87% | 26.47% | 17.40% | 0.28003 | +0.05667 |

Diagnostic important: cette arena etait biaisee par le ciblage absolu de
`HeuristicBot`. Le moteur choisit bien le premier joueur aleatoirement, mais
les adversaires heuristiques resolvaient leurs egalites de cible dans l'ordre
`player_0, player_1, player_2, player_3`. En full heuristique, `player_0` etait
donc cible `16678` fois sur `5000` parties, contre seulement `4147` fois pour
`player_3`.

Correction: `HeuristicBot` accepte maintenant `shuffle_targets=True`. Ce mode
garde les memes regles, mais randomise les egalites de cible. Sonde full
heuristique fair sur `5000` parties:

| Siege | Winrate | Reward moyen | Ciblages recus |
|---|---:|---:|---:|
| player_0 | 29.24% | 0.4352 | 9903 |
| player_1 | 29.64% | 0.4436 | 10074 |
| player_2 | 30.02% | 0.4533 | 10070 |
| player_3 | 28.52% | 0.4301 | 9957 |

Fair arena du 25 avril 2026, `1000` parties par composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Fair Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 52.80% | 40.40% | 36.60% | 28.20% | 0.35620 | +0.00000 |
| Step2 retarget | 49.10% | 43.60% | 38.30% | 31.50% | 0.37720 | +0.02100 |
| Step3 rapide DAgger | 53.40% | 45.70% | 41.30% | 28.60% | 0.38310 | +0.02690 |
| Step3 hybride verify16 | 53.00% | 46.80% | 39.90% | 30.70% | 0.38910 | +0.03290 |

Benchmark seat-rotated post-correction des regles du 25 avril 2026, `1000`
parties par composition. Le joueur evalue tourne entre les quatre positions:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 51.80% | 39.20% | 31.50% | 29.20% | 0.34150 |
| Step2 retarget | 51.70% | 43.00% | 35.70% | 33.30% | 0.37800 |
| Step3 rapide DAgger | 51.90% | 46.20% | 38.40% | 34.50% | 0.39750 |
| Step3 hybride verify16 | 51.20% | 45.30% | 39.30% | 34.30% | 0.39690 |

Conclusion revisee: les modeles actuels ne sont pas moins bons que
`HeuristicBot`. L'ancien benchmark les mettait dans un hard mode tres cible; le
fair mode retire ce focus artificiel, et la rotation des sieges confirme que
Step2 et Step3 restent au-dessus de l'heuristique au composite.

Rapport:

```text
interlude_heuristic_comparison/README.md
interlude_heuristic_comparison/reports/interlude_arena_heuristic_step2_step3_3x1000.md
interlude_heuristic_comparison/reports/fair_arena_heuristic_step2_step3_1000.md
interlude_heuristic_comparison/reports/rotating_tactical_arena_1000.md
interlude_heuristic_comparison/reports/rotating_tactical_arena_post_rules_fix_1000.md
```

## Etape 4 - Identification Des Faiblesses

Sous-dossier:

```text
step4_weakness_analysis/
```

Objectif: ne pas encore entrainer un nouveau modele, mais comprendre ou le
dernier Step3 rapide gagne/perd selon les cartes qu'il rencontre et le moment
de la manche.

Taxonomie figee:

```text
step4_weakness_analysis/CARD_TAXONOMY.md
```

Clustering aligne sur le benchmark fair seat-rotated post-fix, `1000` parties
par composition, seed `260000`:

| Mesure | Resultat |
|---|---:|
| Parties analysees | 4000 |
| Composite Step3 rapide | 0.39750 |
| Winrate moyen | 42.75% |
| Reward moyen | 0.6447 |

Quand Step3 rapide perd, sa position de sortie est:

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 33.01% | 27.90% | 18.82% | 20.26% |

Premiers archetypes a surveiller:

| Archetype | Games | Winrate |
|---|---:|---:|
| Partie avec Roi | 464 | 44.61% |
| Baron avec petite carte | 1010 | 47.13% |
| Princesse tot | 445 | 48.76% |
| Grosse carte tot | 1194 | 49.75% |
| Partie avec Chancelier | 1142 | 51.14% |

Lecture courte: le modele est clairement meilleur que notre inquietude
initiale. Ses bons signaux sont `Pretre -> Garde`, les parties riches en
Gardes, et les fins de manche avec cartes de controle. Les axes faibles a
verifier sont le Roi, les Barons mal equipes, les grosses cartes tres tot, et
certains usages du Chancelier.

Une deuxieme passe Step4 a ajoute des ablations conditionnelles: le modele
choisit naturellement la carte, puis on randomise seulement l'execution fine
de cette carte.

| Ablation | Composite | Delta vs normal | Lecture |
|---|---:|---:|---|
| Normal | 0.39750 | +0.00000 | Reference |
| Garde cible random | 0.38330 | -0.01420 | Ciblage Garde utile |
| Garde guess random | 0.33320 | -0.06430 | Guess Garde tres maitrise |
| Prince cible random | 0.37470 | -0.02280 | Ciblage Prince net |
| Baron cible random | 0.39100 | -0.00650 | Ciblage Baron un peu utile |
| Roi cible random | 0.39750 | +0.00000 | Faiblesse Roi pas expliquee par cible seule |
| Chancelier choix random | 0.39550 | -0.00200 | Choix Chancelier peu converti en winrate |
| Pretre cible random | 0.39900 | +0.00150 | Pas d'effet clair |

Conclusion courte: Garde et Prince sont des competences a proteger. Roi,
Baron faible, Pretre et Chancelier demandent maintenant un audit decisionnel avec
`forcedness` et regret rollout avant tout entrainement cible.

Rapports:

```text
step4_weakness_analysis/reports/step3_fast_card_clusters_post_rules_fix_1000.md
step4_weakness_analysis/reports/step3_fast_card_clusters_1000.md
step4_weakness_analysis/reports/step3_fast_card_ablation_1000.md
step4_weakness_analysis/reports/2026-04-25_step4_card_ablation_analysis.md
step4_weakness_analysis/reports/2026-04-25_step4_to_step5_decision.md
```

## Etape 5 - Tetes D'Execution

Sous-dossier:

```text
step5_execution_heads/
```

Objectif: ameliorer le Step3 rapide sans changer sa decision principale de
carte. L'hypothese issue de Step4 est que le modele sait souvent **quand** jouer
une carte, mais pas toujours **comment** executer Roi, Baron, Pretre et
Chancelier.

La premiere partie Step5 est maintenant terminee: un teacher/audit collecte des
etats naturels du Step3 rapide et compare les executions legales par rollouts
apparies CRN.

Run initial, `500` games de collecte, `40` etats/type, `12` rollouts/action:

| Type | Etats | Best != modele | Regret clair | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|
| Chancelier - choix carte/ordre | 40 | 67.50% | 37.50% | 0.1199 | 0.1104 |
| Baron avec carte faible - cible | 40 | 65.00% | 20.00% | 0.0715 | 0.0667 |
| Roi - cible | 40 | 30.00% | 17.50% | 0.0615 | 0.0583 |
| Baron - cible | 40 | 32.50% | 15.00% | 0.0562 | 0.0521 |
| Pretre - cible | 40 | 40.00% | 12.50% | 0.0382 | 0.0354 |

Conclusion phase A: succes. Il existe bien du regret oracle exploitable, surtout
sur Chancelier.

Le 26 avril 2026, une premiere tete rapide Chancelier a ete entrainee et
validee:

```text
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
```

Cette tete ne choisit pas de jouer Chancelier. Elle intervient uniquement quand
Step3 rapide a deja joue Chancelier, puis corrige le choix des cartes a garder
et remettre sous la pioche. Elle ne fait aucun rollout a l'inference.

| Validation | Step3 rapide | Step3 + tete Chancelier | Delta | Chancelier random |
|---|---:|---:|---:|---:|
| 1000/config seed 860000 | 0.39160 | 0.40710 | +0.01550 | 0.39200 |
| 1000/config seed 870000 | 0.39340 | 0.40830 | +0.01490 | 0.39820 |

Conclusion Chancelier V1: succes. Le gain est superieur au controle random, ne
degrade pas `vs 3H`, et augmente fortement le taux de conservation de la
meilleure carte avec Chancelier. Limite actuelle: la tete apprend surtout le tri
local; la planification fine de la pioche reste a travailler plus tard.

Rapports:

```text
step5_execution_heads/README.md
step5_execution_heads/reports/execution_teacher_initial_40x12_report.md
step5_execution_heads/reports/2026-04-25_step5_phase_a_execution_teacher_conclusion.md
step5_execution_heads/reports/2026-04-26_step5_chancellor_execution_head_v1.md
```

## Etat Des Modeles

Meilleur checkpoint historique conserve:

```text
models/checkpoints/curriculum_phase1.pth
```

Joueur Step3 hybride avec recherche locale:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_attempt2_strict.pth \
  --override-margin 0.10 \
  --verify-rollouts 16 \
  --verify-min-win-delta 0.125 \
  --verify-min-score-delta 0.05 \
  --verify-t-threshold 0.75
```

Joueur Step5 rapide actuel:

```text
Step3 rapide DAgger + tete Chancelier V1
```

Checkpoint de la tete locale:

```text
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
```

Evaluation:

```bash
python3 -m step5_execution_heads.evaluate_chancellor_head \
  --head chancellor_head_attempt3_small_regularized.pth \
  --games 1000 \
  --chancellor-margin 0.12
```

Fondation imitation heuristique pour la prochaine phase RL:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Socle Step2 de la nouvelle pipeline contre `HeuristicBot`:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Evaluation 100 parties par configuration:

| Modele | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|
| `curriculum_phase1.pth` | 55% | 43% | 33% | 20% |
| `belief_conditioned_ppo_final.pth` | 49% | 35% | 27% | 18% |

Interpretation:

- `curriculum_phase1.pth` reste le meilleur checkpoint historique de reference.
- Les deux meilleurs joueurs issus de la pipeline nettoyee sont Step3 rapide
  DAgger et Step3 hybride verify16. Le premier est autonome et legerement devant
  sur le benchmark seat-rotated post-fix du 25 avril 2026; le second combine
  Step2, une tete advantage et une verification CRN locale.
- Le meilleur joueur rapide en cours devient Step5 Chancelier V1: ce n'est pas
  encore un checkpoint actor unique, mais un wrapper `Step3 rapide + tete locale
  Chancelier`. Il sert de base active pour ajouter ensuite Baron/Roi/Pretre.
- L'ancien prototype actor + belief + search a ete archive dans
  `step3_action_value/legacy/step3_hybrid_search_prototype/`: il ne fait plus
  partie de la pipeline active.
- `heuristic_student_attempt4_player0_chancellor_order.pth` n'est pas le champion final de jeu; c'est le meilleur warm start supervise pour apprendre ensuite a depasser l'heuristique.
- `step2_retarget_distilled_attempt1.pth` bat maintenant `HeuristicBot` et sert de base au joueur Step3.
- `belief_conditioned_ppo_final.pth` prouve que l'architecture actor + belief peut tourner, mais elle n'a pas encore appris a exploiter le belief de facon fiable.
- Le diagnostic tactique a montre des erreurs du type Baron joue contre une cible que le belief estimait tres probablement haute.

## Test Contre-Factuel Belief

Question posee le 24 avril 2026: est-ce que le dernier actor belief-conditioned aurait mieux joue si ses decisions tactiques avaient vraiment suivi ses propres probabilites de belief ?

Checkpoint teste:

```text
models/checkpoints/champion_belief_ppo_attempt2_tactical_best.pth
```

Commande:

```bash
python3 scripts/evaluation/evaluate_belief_counterfactual.py \
  --checkpoint champion_belief_ppo_attempt2_tactical_best.pth \
  --games 1000 \
  --seed-start 200000 \
  --modes raw retarget tactical \
  --output logs/evaluations/2026-04-24_attempt2_belief_counterfactual_1000.json \
  --run-log logs/runs/2026-04-24_belief_counterfactual_attempt2.md
```

Modes:

- `raw`: le modele tel quel.
- `retarget`: meme actor, meme carte choisie, mais cible/devinette corrigee selon le belief pour Garde/Baron/Prince/Roi, plus choix visible du Chancelier.
- `tactical`: autorise aussi quelques remplacements de carte si le belief donne une opportunite tactique nette.

Resultats sur `1000` parties par configuration:

| Mode | Score composite | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|---:|
| Actor brut | 0.2345 | 49.1% | 33.2% | 22.2% | 13.1% |
| Retarget belief | 0.2948 | 53.3% | 41.8% | 28.5% | 18.1% |
| Tactical belief | 0.2694 | 49.5% | 36.1% | 26.3% | 17.2% |
| Champion `curriculum_phase1.pth` | 0.2968 | 56.6% | 45.0% | 25.8% | 18.2% |

Lecture:

- Oui, les probas du belief contiennent bien un signal utile.
- Le probleme principal n'est pas seulement "belief mauvais"; c'est surtout "actor ne s'en sert pas proprement".
- Le simple retargeting corrige `4385` decisions sur `13116` et monte de `+6.03` points de score composite.
- Sur les Gardes, le taux de choix du top belief passe de `57.7%` a `100%`; le taux de hit passe de `23.1%` a `31.7%`.
- Le mode `tactical`, plus ambitieux, aide moins que `retarget`: changer la carte jouee avec des regles externes perturbe davantage la politique.

Accuracy du belief sur ce checkpoint:

| Adversaires | Accuracy globale | Debut de manche `14-17` cartes | Fin de manche `0-2` cartes |
|---|---:|---:|---:|
| Heuristiques | 31.1% | 24.6% | 57.5% |
| Randoms | 25.6% | 23.8% | 43.6% |

Conclusion: il faut garder le belief, mais entrainer explicitement l'actor a l'utiliser pour les choix de cible/devinette, plutot que lui ajouter le belief comme simple feature en esperant que PPO comprenne seul.

## Distillation Retarget Belief

Tentative lancee le 24 avril 2026:

```bash
python3 scripts/training/distill_belief_retarget.py \
  --start champion_belief_ppo_attempt2_tactical_best.pth \
  --output champion_belief_retarget_distilled_attempt1.pth \
  --games 5000 \
  --epochs 10
```

Principe:

- on gele `encoder` et `belief_head`;
- on collecte des decisions de l'actor brut;
- on calcule la cible `retarget belief`;
- on entraine seulement l'actor a produire cette action corrigee.

Dataset collecte:

- `5000` parties mixtes `0H/1H/2H/3H`;
- `15892` decisions player_0;
- `5579` decisions corrigees par retarget, soit `35.1%`;
- corrections principales: `3108` Garde, `980` Prince, `682` Chancelier, `659` Baron, `150` Roi.

Resultats arena sur `1000` parties par configuration:

| Modele / mode | Score composite | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|---:|
| Avant: actor brut attempt2 | 0.2345 | 49.1% | 33.2% | 22.2% | 13.1% |
| Avant: retarget oracle attempt2 | 0.2948 | 53.3% | 41.8% | 28.5% | 18.1% |
| Apres: actor brut distille | 0.2856 | 52.4% | 39.6% | 27.2% | 18.1% |
| Apres: retarget du distille | 0.2954 | 53.0% | 41.7% | 28.2% | 18.6% |
| Champion `curriculum_phase1.pth` | 0.2968 | 56.6% | 45.0% | 25.8% | 18.2% |

Conclusion:

- succes partiel important: l'actor a bien internalise l'essentiel du retarget;
- score composite brut: `+5.11` points;
- winrate vs 3 heuristiques: `+5.0` points;
- ecart actor brut vs retarget: `0.0603` avant, `0.0098` apres;
- le nouveau checkpoint ne depasse pas encore `curriculum_phase1.pth`, mais il valide clairement la direction actor-belief.

Logs:

- `logs/runs/2026-04-24_belief_retarget_distillation_attempt1.md`
- `logs/evaluations/2026-04-24_belief_retarget_distillation_attempt1_train.json`
- `logs/evaluations/2026-04-24_belief_retarget_distilled_attempt1_eval_1000.json`
- `logs/evaluations/2026-04-24_belief_retarget_distilled_attempt1_counterfactual_1000.json`

## Ce Qu'on Garde

- Le moteur `LoveLetterRLEnv`.
- Le bot `HeuristicBot`, comme professeur et benchmark.
- Le dataset `data/heuristic_dataset.pkl`.
- La reference historique `curriculum_phase1.pth`.
- Le warm start `belief_conditioned_bc.pth`.
- Les outils d'evaluation et de diagnostic.
- L'interface Streamlit et le mode console.

## Ce Qui A Ete Abandonne

- Les anciens scripts PPO/Tianshou.
- Les checkpoints historiques non champions.
- Le pool self-play obsolete.
- Les scripts personnels ou redondants non necessaires a un repo public.
- L'idee de choisir un champion sur un run court ou sur le dernier checkpoint sauvegarde.

## Nouvelle Direction

Deux branches Step3 sont maintenant separees clairement:

1. Step3 hybride `verify16` est conserve comme joueur fort avec recherche a
   l'inference.
2. Step3 rapide continue avec DAgger + trust region/KL, sans rollout a
   l'inference.

Objectif intermediaire Step3:

- stabiliser le gain rapide DAgger `+0.788` point composite mesure sur
  `5000` parties par composition;
- supprimer la regression vs `3H`;
- lancer la prochaine boucle DAgger avec `--trust-region-kl-weight` pour
  empecher les sur-corrections;
- ne plus tuner les seuils sur des evaluations `1000` parties.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Note GitHub

Les gros artefacts sont volontairement ignores par `.gitignore`:

- `data/*.pkl`
- `models/checkpoints/*.pth`
- `logs/`

Pour publier le projet, garder le code dans Git et distribuer les datasets/checkpoints via Git LFS, release GitHub, ou stockage externe.

## Jouer

Interface Streamlit:

```bash
streamlit run love_letter_web/streamlit_app.py
```

Mode console:

```bash
python3 -m love_letter.gameplay.play_vs_agent
```

Les interfaces chargent par defaut:

```text
models/checkpoints/curriculum_phase1.pth
```

Note: le joueur Step3 v2 verifie n'est pas encore branche dans ces interfaces.
Pour l'instant il se lance via:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_attempt2_strict.pth \
  --games 1000 \
  --mode composite \
  --compare-baseline \
  --override-margin 0.10 \
  --verify-rollouts 16 \
  --verify-min-win-delta 0.125 \
  --verify-min-score-delta 0.05 \
  --verify-t-threshold 0.75
```

## Entrainement

Collecte du dataset heuristique:

```bash
python3 -m scripts.training.collect_heuristic_data
```

Pretraining actor + belief:

```bash
python3 scripts/training/pretrain_belief_actor.py
```

PPO curriculum actor + belief:

```bash
python3 scripts/training/train_belief_ppo.py
```

Distillation du retarget belief:

```bash
python3 scripts/training/distill_belief_retarget.py
```

## Evaluation

Baselines random/heuristique:

```bash
python3 scripts/evaluation/evaluate_baselines.py --games 20000
```

Evaluation d'un checkpoint ou des checkpoints conserves:

```bash
python3 -m scripts.evaluation.evaluate_models
```

Diagnostic tactique:

```bash
python3 scripts/evaluation/diagnose_model.py \
  --checkpoint models/checkpoints/curriculum_phase1.pth \
  --games 30 \
  --output-json logs/evaluations/diagnostic.json \
  --output-md diagnostics/diagnostic.md
```

Mesure du belief:

```bash
python3 scripts/evaluation/measure_belief.py \
  --checkpoint models/checkpoints/curriculum_phase1.pth \
  --games 500
```

Test contre-factuel actor/belief:

```bash
python3 scripts/evaluation/evaluate_belief_counterfactual.py \
  --checkpoint champion_belief_ppo_attempt2_tactical_best.pth \
  --games 1000 \
  --output logs/evaluations/counterfactual.json
```

Sanity-check du moteur:

```bash
python3 scripts/debug/check_engine_invariants.py
```

Conformite aux regles:

```bash
python3 scripts/debug/check_rules_conformance.py
```

## Backend Et Web App

Le backend FastAPI et le frontend React/Vite sont maintenant la cible produit.
Streamlit reste utile comme prototype/debug.

```bash
uvicorn love_letter_web.backend.main:app --host 127.0.0.1 --port 8000
cd love_letter_web/frontend
npm run dev
```

Voir `love_letter_web/README.md`.

## Etat De Pause Technique

Le projet est arrete proprement sur `champion_cbp` et la web app jouable. Les
prochaines etapes ci-dessous restent des pistes de reprise, pas du travail en
cours.

Piste IA possible: relancer Step3 v2 DAgger avec trust region/KL:

- repartir de `step3_advantage_v2_dagger_attempt1_iter1.pth`;
- garder `verify_rollouts = 0` a l'inference;
- utiliser `--trust-region-kl-weight` pour empecher l'oubli catastrophique;
- evaluer le candidat retenu sur `5000` parties par composition;
- viser un gain composite positif sans regression vs `3H`;
- brancher ensuite le meilleur joueur rapide dans `play_vs_agent` et
  `love_letter_web`.

Piste produit possible: terminer le polish UI, publier les checkpoints via
GitHub Release/Git LFS/Hugging Face, puis tester l'app de bout en bout.
