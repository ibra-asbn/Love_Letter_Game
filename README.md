# Love Letter RL

Etat final documente le **3 juin 2026**.

Ce README est le recap complet et date du projet, du premier diagnostic au
point d'arret final. Quand une date exacte existe dans un rapport, elle est
reprise. Quand une date manque, elle est reconstruite a partir de l'ordre des
fichiers, des rapports et des commits.

Objectif initial: construire un agent de reinforcement learning capable de
jouer fort a **Love Letter**, puis fournir une interface jouable contre le
meilleur modele.

Objectif final du depot: garder le projet propre, lisible, documente et pret a
etre partage sur GitHub, sans relancer de nouvelle experimentation.

## Etat Final

Champion retenu:

```text
champion_cbp = Step3 rapide DAgger + Chancelier V1 + Baron V1 + Prince V1
```

Statut:

- le moteur de jeu est audite contre les regles locales;
- la pipeline IA est documentee de bout en bout;
- la web app FastAPI + React/Vite est la cible produit;
- le depot GitHub est organise autour du code, des rapports et de l'app web;
- le projet est arrete proprement au 3 juin 2026.

Documents importants:

| Fichier | Role |
|---|---|
| `README.md` | Recap complet date |
| `docs/project_journal_fr.md` | Journal synthetique etape par etape |
| `docs/github_handoff_fr.md` | Handoff GitHub et nettoyage |
| `docs/love_letter_rules_fr.md` | Regles locales de reference |
| `docs/diagnostics/` | Diagnostics dates et audits transverses |
| `love_letter_web/README.md` | Lancer l'app web |

## Chronologie Rapide

| Date | Etape | Decision |
|---|---|---|
| 2026-04-24 | Baseline random vs heuristique | Le jeu est exploitable |
| 2026-04-24 | Step1 - Heuristic Mastery | Le modele copie l'heuristique |
| 2026-04-24 | Step2 - RL/retarget | Le modele bat `HeuristicBot` |
| 2026-04-24 | Diagnostics belief/action-value | Le belief est utile mais mal exploite |
| 2026-04-25 | Audit regles | Corrections Baron/Roi/Servante |
| 2026-04-25 | Step3 - Action-value | Signal positif, deux branches conservees |
| 2026-04-25 | Interlude arena | Correction du biais `player_0 only` |
| 2026-04-25 | Step4 - Weakness analysis | Ne pas refaire un PPO global |
| 2026-04-25 | Step5 Phase A | Regret exploitable trouve |
| 2026-04-26 | Step5 Chancelier/Baron/Prince | `champion_cbp` devient la reference |
| 2026-04-26 | Step6 population | Champion sain, mais curriculum reste fort |
| 2026-04-26 | Step7 self-play league | Ligue OK, candidats rejetes |
| 2026-04-28 au 2026-05-01, reconstruit | Web app produit | FastAPI + React/Vite devient la cible produit |
| 2026-06-03 | Handoff final | Documentation et GitHub |

## Architecture Du Depot

```text
love_letter/
  engine.py                  # moteur PettingZoo/Gym du jeu
  belief_actor.py            # architecture actor + belief
  belief_policy.py           # chargement/inference des policies
  bots/heuristic.py          # bot heuristique de reference
  gameplay/play_vs_agent.py  # jeu console

scripts/
  debug/                     # sanity checks et debug local
  evaluation/                # evaluations, baselines, diagnostics
  training/                  # anciens et nouveaux scripts d'entrainement

docs/
  project_journal_fr.md      # journal narratif canonique
  github_handoff_fr.md       # checklist handoff GitHub
  love_letter_rules_fr.md    # regles locales
  diagnostics/               # diagnostics dates et audits transverses

step1_heuristic_mastery/     # imitation du bot heuristique
step2_rl_finetune/           # depasser l'heuristique
step3_action_value/          # action-value, verify16, DAgger
step4_weakness_analysis/     # analyse des faiblesses par cartes
step5_execution_heads/       # tetes locales d'execution
step6_self_play/             # population et matchups asymetriques
step7_self_play_league/      # ligue self-play + promotion gate
  state/                     # roster, resultats jsonl, historique promotion

love_letter_web/
  backend/main.py            # API FastAPI
  frontend/                  # app React/Vite
```

Convention de rangement:

- `docs/` contient la documentation globale et les diagnostics qui ne sont pas
  propres a une seule etape.
- `step*_*/reports/` contient les resultats experimentaux de chaque etape.
- `step*_*/checkpoints/` et `step*_*/datasets/` contiennent les artefacts
  techniques associes a l'etape.
- `step7_self_play_league/state/` contient l'etat mutable de la ligue
  self-play.
- `love_letter_web/` contient l'application produit FastAPI + React/Vite.

## Regles Du Jeu En Bref

Love Letter est un jeu de deduction a information cachee. Chaque joueur garde
une seule carte en main. A son tour, il pioche une carte, choisit une des deux
cartes a jouer, applique son effet, puis garde l'autre carte.

Une manche se termine quand il ne reste plus qu'un joueur en vie ou quand la
pioche est vide. Si la pioche est vide, les joueurs encore en vie comparent la
carte gardee en main: la valeur la plus haute gagne la manche. Dans l'app web
du projet, le premier joueur a `2` points gagne la partie.

Le paquet utilise les cartes suivantes:

| Valeur | Carte | Effet |
|---:|---|---|
| 0 | Espionne | Pas d'effet immediat. Bonus de 1 point si un seul joueur vivant en a joue le plus. |
| 1 | Garde | Devine la carte d'un adversaire, sauf Garde. Si la devinette est juste, la cible sort. |
| 2 | Qadi | Regarde secretement la carte d'un adversaire. |
| 3 | Emir | Compare sa carte gardee avec celle d'un adversaire; la plus faible sort. |
| 4 | Hajib | Protege le joueur des effets ciblants jusqu'a son prochain tour. |
| 5 | Wali | Force un joueur, soi inclus, a defausser sa carte et repiocher. Defausser l'Amira elimine. |
| 6 | Vizir | Pioche jusqu'a deux cartes, en garde une, puis remet les autres au fond de la pioche. |
| 7 | Sultan | Echange sa carte avec celle d'un adversaire. |
| 8 | Sultane | Doit etre jouee si elle est en main avec le Wali ou le Sultan. |
| 9 | Amira | Si elle est jouee ou defaussee, son proprietaire est elimine. |

La reference complete des regles et des cas particuliers est dans
`docs/love_letter_rules_fr.md`.

## Baseline - 24 Avril 2026

Question: est-ce qu'une politique simple fait mieux que le hasard ?

Ce qu'on voulait ajouter:

- un test statistique propre;
- un point de comparaison random vs heuristique;
- une preuve que Love Letter est un terrain exploitable pour du RL.

Commande de reference:

```bash
python3 scripts/evaluation/evaluate_baselines.py \
  --games 20000 \
  --output logs/evaluations/2026-04-24_baselines_random_vs_heuristic_20000.json
```

Resultats:

| Politique player_0 | Adversaires | Winrate | Reward moyen |
|---|---|---:|---:|
| Random | 3 randoms | 29.92% | 0.397 |
| HeuristicBot | 3 randoms | 45.69% | 0.677 |

Conclusion: le jeu est bien exploitable. Une politique lisible non neuronale
obtient un edge massif. La suite peut chercher a copier puis depasser cette
heuristique.

## Step1 - Heuristic Mastery - 24 Avril 2026

Dossier: `step1_heuristic_mastery/`

Objectif: obtenir un modele qui absorbe tout ce que `HeuristicBot` sait faire.
Si le reseau ne sait pas copier l'heuristique, il est trop tot pour attendre du
RL qu'il la depasse.

Ce qu'on voulait ajouter:

- une collecte de sequences depuis `HeuristicBot`;
- un modele recurrent imitant l'heuristique;
- un split train/validation/test;
- une evaluation action par action;
- une correction de l'observation du Chancelier.

Correction importante: pendant l'effet du Chancelier, l'observation expose les
cartes disponibles et leur ordre de choix. Avant cette correction, certaines
actions `900/902/904` etaient partiellement impossibles a imiter.

Checkpoint:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Resultats imitation:

| Split | Action accuracy | Action loss | Belief accuracy |
|---|---:|---:|---:|
| Train | 99.92% | 0.0048 | 32.93% |
| Validation | 98.93% | 0.0313 | 31.43% |
| Test | 98.94% | 0.0292 | 30.62% |

Evaluation arena:

| Politique player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Student heuristique | 47.68% | 32.86% | 21.40% | 11.72% | 0.22448 |
| HeuristicBot | 47.86% | 32.78% | 21.62% | 11.82% | 0.22556 |

Decision: succes. Le student n'est pas encore meilleur que l'heuristique, mais
il est statistiquement au meme niveau et devient un warm start valide.

## Step2 - Battre L'Heuristique - 24 Avril 2026

Dossier: `step2_rl_finetune/`

Objectif: partir du student Step1 et obtenir un modele qui bat vraiment
`HeuristicBot`.

Ce qu'on voulait ajouter:

- du RL ou une correction au-dessus du student;
- une exploitation plus forte du belief;
- une evaluation longue, pas seulement un run court;
- un modele plus fort que `HeuristicBot` dans toutes les compositions.

Ce qui a marche: le mode contre-factuel `retarget` utilisait mieux le belief
pour choisir les cibles et les devinettes. On a donc distille ce comportement
dans l'actor.

Checkpoint retenu:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Confirmation longue, `5000` parties par configuration:

| Modele player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.54% | 36.80% | 26.36% | 16.04% | 0.26738 |
| HeuristicBot | 46.02% | 33.48% | 21.36% | 11.30% | 0.22226 |

Gain confirme:

- `+4.51` points de composite vs `HeuristicBot`;
- `+4.52` points vs 3 randoms;
- `+4.74` points vs 3 heuristiques.

Decision: succes. Step2 devient le socle de la nouvelle pipeline.

## Diagnostics Belief Et Action-Value - 24 Avril 2026

Dossiers et rapports:

- `docs/diagnostics/2026-04-24_belief_conditioned_ppo_model_diagnostic.md`
- `docs/diagnostics/2026-04-24_step3_mini_action_value_probe.md`
- `docs/diagnostics/2026-04-24_attempt2_tactical_best_model_diagnostic.md`

Objectif: comprendre pourquoi les modeles belief-conditioned avaient du signal
mais ne devenaient pas automatiquement champions.

Ce qu'on voulait ajouter:

- des diagnostics action par action;
- des contre-factuels actor/belief;
- des mini probes de rollouts;
- une lecture tactique des erreurs de Garde, Baron, Prince, Roi et Chancelier.

Constat:

- le belief contient de l'information utile;
- l'actor ne l'utilise pas assez naturellement;
- le Baron est un probleme de timing et de cible;
- certaines erreurs viennent d'une execution locale, pas du choix global de
  jouer une carte.

Decision: garder le belief, mais ne pas esperer qu'un PPO global decouvre seul
la bonne exploitation. Il faut des corrections plus ciblees.

## Audit Des Regles - 25 Avril 2026

Rapports:

- `docs/love_letter_rules_fr.md`
- `docs/diagnostics/2026-04-25_rules_conformance_audit.md`

Objectif: verifier que le moteur respecte les regles locales de Love Letter
avant d'entrainer davantage.

Ce qu'on voulait ajouter:

- un texte de reference local;
- des tests executables de conformite;
- une correction des ecarts de regles;
- une base fiable pour les evaluations longues.

Corrections importantes:

- Baron;
- Roi;
- Servante;
- cas particuliers autour des actions forcees.

Commande:

```bash
python3 scripts/debug/check_rules_conformance.py
```

Decision: les evaluations post-25 avril deviennent la reference principale.

## Step3 - Action-Value / Search - 25 Avril 2026

Dossier: `step3_action_value/`

Objectif: apprendre quand une action alternative est meilleure que l'action par
defaut de Step2.

Ce qu'on voulait ajouter:

- une estimation action-value;
- des rollouts CRN apparies;
- une branche hybride verifiee;
- une branche rapide autonome sans rollouts a l'inference;
- une meilleure decision sans casser les acquis de Step2.

### Branche Hybride Verify16

Checkpoint:

```text
step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth
```

Validation sur trois seeds independants, `1000` parties par composition:

| Seed | Step3 hybride verify16 | Step2 | Delta |
|---|---:|---:|---:|
| 134000 | 0.28230 | 0.26720 | +0.01510 |
| 135000 | 0.27890 | 0.26580 | +0.01310 |
| 136000 | 0.27890 | 0.26330 | +0.01560 |
| Moyenne | 0.28003 | 0.26543 | +0.01460 |

Decision: succes technique. La branche hybride est meilleure que Step2, mais
elle est trop lente pour devenir l'agent produit principal.

### Branche Rapide DAgger

Checkpoint:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Evaluation officielle, `5000` parties par composition:

| Composition | DAgger iter1 | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.96% | 52.08% | +1.88 pts |
| vs 1H+2R | 39.72% | 38.38% | +1.34 pts |
| vs 2H+1R | 26.42% | 25.02% | +1.40 pts |
| vs 3H | 14.90% | 15.12% | -0.22 pt |
| Composite | 0.27226 | 0.26438 | +0.00788 |

Decision: signal positif, mais il faut proteger le modele contre les
regressions. Step3 rapide devient la base pratique pour la suite.

## Interlude - Verification Du Biais D'Arene - 25 Avril 2026

Dossier: `interlude_heuristic_comparison/`

Objectif: verifier si les modeles etaient vraiment sous l'heuristique ou si
l'arene historique favorisait certains comportements.

Ce qu'on voulait ajouter:

- une arena fair seat-rotated;
- un `HeuristicBot` sans focus artificiel sur `player_0`;
- une lecture plus propre des comparaisons Step2/Step3/heuristique.

Constat:

- l'ancienne arena `player_0 only` etait biaisee;
- elle reste utile comme hard mode;
- la nouvelle arena fair confirme que Step2 et Step3 sont au-dessus de
  l'heuristique au composite.

Decision: utiliser l'arene fair seat-rotated pour les decisions principales.

## Step4 - Weakness Analysis - 25 Avril 2026

Dossier: `step4_weakness_analysis/`

Objectif: ne pas relancer un entrainement global. Comprendre ou Step3 rapide
gagne, ou il perd, et quelles competences doivent etre protegees.

Ce qu'on voulait ajouter:

- une taxonomie des cartes et phases de partie;
- un clustering des archetypes de mains;
- des ablations conditionnelles;
- une lecture des faiblesses par carte;
- une decision claire avant Step5.

Modele analyse:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Resultats principaux:

| Mesure | Resultat |
|---|---:|
| Parties | 4000 |
| Composite | 0.39750 |
| Winrate moyen | 42.75% |
| Reward moyen | 0.6447 |

Ablations conditionnelles:

| Ablation | Composite | Delta vs normal | Lecture |
|---|---:|---:|---|
| Normal | 0.39750 | +0.00000 | Reference |
| Garde cible random | 0.38330 | -0.01420 | Ciblage Garde utile |
| Garde guess random | 0.33320 | -0.06430 | Guess Garde tres maitrise |
| Prince cible random | 0.37470 | -0.02280 | Ciblage Prince net |
| Baron cible random | 0.39100 | -0.00650 | Ciblage Baron un peu utile |
| Chancelier choix random | 0.39550 | -0.00200 | Effet global faible |

Lecture:

- Garde et Prince sont des competences fortes a proteger;
- Baron faible, Roi, Pretre et Chancelier demandent un audit local;
- le probleme n'est pas seulement "quelle carte jouer", mais souvent
  "comment executer la carte".

Decision: passer a Step5 avec des tetes d'execution locales.

## Step5 Phase A - Teacher/Audit D'Execution - 25 Avril 2026

Dossier: `step5_execution_heads/`

Objectif: observer les etats naturels ou Step3 rapide joue deja Roi, Baron,
Pretre ou Chancelier, puis mesurer si une meilleure execution existe.

Ce qu'on voulait ajouter:

- un teacher par rollouts CRN;
- une mesure du regret par execution;
- une separation entre decision de carte et execution fine;
- un dataset filtre pour entrainer de petites tetes specialisees.

Run principal:

```bash
python3 -m step5_execution_heads.collect_execution_teacher \
  --games 300 \
  --max-states-per-kind 40 \
  --rollouts-per-action 12
```

Synthese:

| Type | Etats | Best != modele | Regret clair | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|
| Chancelier | 40 | 67.50% | 37.50% | 0.1199 | 0.1104 |
| Baron avec carte faible | 40 | 65.00% | 20.00% | 0.0715 | 0.0667 |
| Roi | 40 | 30.00% | 17.50% | 0.0615 | 0.0583 |
| Baron | 40 | 32.50% | 15.00% | 0.0562 | 0.0521 |
| Pretre | 40 | 40.00% | 12.50% | 0.0382 | 0.0354 |

Decision: Step5 est justifie. Le regret exploitable existe, surtout sur
Chancelier et Baron faible.

## Step5 Chancelier V1 - 26 Avril 2026

Objectif: corriger uniquement l'execution du Chancelier quand Step3 rapide a
deja decide de jouer Chancelier.

Ce qu'on voulait ajouter:

- un dataset Chancelier equilibre;
- une tete rapide sans rollouts a l'inference;
- un seuil de confiance;
- une validation sur seeds independants.

Checkpoint retenu:

```text
step5_execution_heads/checkpoints/chancellor_head_attempt3_small_regularized.pth
```

Resultats:

| Validation | Step3 rapide | Step3 + Chancelier | Delta |
|---|---:|---:|---:|
| 500/config seed 850000 | 0.39920 | 0.40700 | +0.00780 |
| 1000/config seed 860000 | 0.39160 | 0.40710 | +0.01550 |
| 1000/config seed 870000 | 0.39340 | 0.40830 | +0.01490 |

Decision: succes. La tete Chancelier V1 devient un module de reference.

## Step5 Baron V1 - 26 Avril 2026

Objectif: corriger la faiblesse Baron, surtout les mauvaises decisions avec
`Baron + Prince` et `Baron + Chancelier`.

Ce qu'on voulait ajouter:

- une comparaison locale `jouer Baron` vs `jouer l'autre carte`;
- une cible plus sure;
- un module rapide, sans rollouts a l'inference;
- une evaluation longue.

Evaluation `5000` parties par composition:

| Politique | Composite | Baron en main | Baron joue | Duel gagne | Duel perdu |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.38564 | 46.51% | 48.89% | 71.65% | 24.44% |
| Baron target random | 0.38230 | 45.66% | 47.71% | 69.92% | 26.22% |
| Step5 Baron specialist | 0.39504 | 49.02% | 54.88% | 79.83% | 17.01% |

Decision: succes. Baron V1 devient un module de reference.

## Step5 Combine Chancelier + Baron - 26 Avril 2026

Objectif: verifier que les deux modules s'additionnent sans se neutraliser.

Evaluation `5000` parties par composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.68% | 44.16% | 38.76% | 34.00% | 0.39228 |
| Step3 + Chancelier V1 | 52.62% | 45.46% | 39.80% | 35.72% | 0.40582 |
| Step3 + Baron V1 | 52.48% | 45.74% | 39.00% | 35.16% | 0.40160 |
| Step3 + Chancelier + Baron | 53.36% | 46.86% | 39.96% | 37.00% | 0.41496 |

Decision: les modules s'additionnent bien. Le joueur Step5 de reference devient
`Step3 rapide + Chancelier V1 + Baron V1`.

## Step5 Prince V1 - 26 Avril 2026

Objectif: verifier si un module Prince peut ameliorer le ciblage sans forcer
trop souvent la carte.

Resultats:

| Politique | Composite global | Prince en main | Prince joue | Hit Princesse | Suicide soi |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.39162 | 48.31% | 52.02% | 7.64% | 0.51% |
| Step3 + Prince V1 | 0.39428 | 48.97% | 51.03% | 9.38% | 0.35% |

Decision: signal positif mais plus leger. Prince V1 reste candidat utile et
est inclus dans la composition `champion_cbp`, mais avec moins de certitude que
Chancelier et Baron.

## Step6 - Self-Play Et Population - 26 Avril 2026

Dossier: `step6_self_play/`

Objectif: verifier le champion contre son lignage direct avant de lancer une
ligue de self-play.

Ce qu'on voulait ajouter:

- une evaluation de population;
- des matchups asymetriques;
- un test contre l'ancien champion historique `curriculum_phase1`;
- une lecture plus riche que les arenas vs random/heuristique.

Champion teste:

```text
champion_cbp = Step3 + Chancelier + Baron + Prince
```

Evaluation de lignage, `5000` manches:

| Politique | Score >=1 | Victoire manche | Reward moyen |
|---|---:|---:|---:|
| Champion CBP | 32.28% | 28.36% | 0.5010 |
| Step3 seul | 28.96% | 25.08% | 0.4567 |
| Step2 | 28.50% | 26.02% | 0.4416 |
| Heuristique fair | 25.28% | 21.58% | 0.3811 |

Matchups asymetriques:

| Matchup | Singleton | Victoire manche | Opposants/copie | Victoire manche |
|---|---|---:|---|---:|
| champion_vs_3_step3 | Champion CBP | 27.64% | Step3 seul | 24.43% |
| champion_vs_3_step2 | Champion CBP | 26.42% | Step2 | 24.93% |
| champion_vs_3_heuristic | Champion CBP | 33.14% | Heuristique fair | 22.61% |
| step3_vs_3_champions | Step3 seul | 23.42% | Champion CBP | 25.97% |
| step2_vs_3_champions | Step2 | 22.38% | Champion CBP | 26.25% |

Point de vigilance:

| Matchup | Singleton | Victoire manche | Opposants/copie | Victoire manche |
|---|---|---:|---|---:|
| champion_vs_3_curriculum | Champion CBP | 25.10% | Curriculum phase1 | 25.31% |

Decision: le champion est sain et domine son lignage direct, mais
`curriculum_phase1.pth` reste un excellent sparring partner.

## Step7 - Ligue Self-Play - 26 Avril 2026

Dossier: `step7_self_play_league/`

Objectif: maintenir une ligue de self-play autour de `champion_cbp` et ne
promouvoir un candidat que s'il passe l'Elo et les garde-fous tactiques.

Ce qu'on voulait ajouter:

- un roster actif de 5 politiques;
- une evaluation Elo multi-joueurs;
- un gate de promotion/rejet;
- des checkpoints candidats sauvegardes a chaque iteration;
- une facon de continuer le projet sans oublier les anciens adversaires.

Roster initial:

- `champion_cbp`;
- `curriculum_phase1`;
- `step3_fast`;
- `step2_retarget`;
- `heuristic_fair`.

Bootstrap Elo initial, `10 000` manches:

| Policy | Elo bootstrap |
|---|---:|
| `champion_cbp` | 1546.7 |
| `step2_retarget` | 1520.5 |
| `curriculum_phase1` | 1506.8 |
| `heuristic_fair` | 1479.5 |
| `step3_fast` | 1446.5 |

Premier cycle self-play:

- `sp_iter_0001`: candidat conservateur, rejete;
- `sp_iter_0002`: candidat plus libre, rejete.

Decision `sp_iter_0002`:

| Check | Resultat |
|---|---|
| Elo | False |
| Main win guardrail | True |
| Baron loss guardrail | True |
| Chancellor guardrail | True |
| Guard hit guardrail | True |

Metrics:

| Mesure | Valeur |
|---|---:|
| Candidate Elo | 1508.659563 |
| Best Elo | 1519.655293 |
| Candidate main round win rate | 0.283724 |
| Best main round win rate | 0.271448 |

Decision: la ligue fonctionne, mais aucun candidat ne remplace le champion. Le
projet s'arrete avec `champion_cbp` comme champion courant.

## Web App Produit - 28 Avril Au 1 Mai 2026, Date Reconstruite

Dossier: `love_letter_web/`

Objectif: sortir de l'ancien prototype local et construire une experience jouable.

Ce qu'on voulait ajouter:

- backend FastAPI;
- frontend React/Vite;
- integration du champion `champion_cbp`;
- menu narratif du Qadi;
- rappel des regles;
- choix des adversaires IA;
- profils joueurs;
- stats locales;
- replay omniscient de fin de partie;
- assets visuels et audio.

Etat actuel:

- `love_letter_web/backend/main.py` expose les endpoints jeu, regles, profils,
  policies, replay, actions humaines, pas IA et manches suivantes;
- `love_letter_web/frontend/` contient l'app React/Vite;
- l'app propose menu, intro video, tutoriel, cartes, regles, parametres, table
  jouable et journal de partie;
- `tests/test_love_letter_web_backend.py` couvre les profils, stats, replay,
  actions speciales et logs structures.

Commandes:

```bash
uvicorn love_letter_web.backend.main:app --host 127.0.0.1 --port 8000
cd love_letter_web/frontend
npm run dev
```

Decision: l'app web devient la cible produit unique.

## Hebergement Web Et Recuperation Des Donnees - 3 Juin 2026

Statut public au 3 juin 2026: le depot ne contient pas encore de lien public
verifie vers le jeu complet. Le lien doit etre ajoute ici seulement apres
verification d'un frontend Vercel connecte au backend FastAPI deploye.

Lien public de jeu complet:

```text
A renseigner apres deploiement verifie.
```

Architecture cible:

- backend: FastAPI dans `love_letter_web/backend/main.py`, prevu pour etre
  heberge sur Hugging Face Spaces;
- frontend: React/Vite dans `love_letter_web/frontend/`, prevu pour etre
  heberge sur Vercel;
- en local, Vite redirige les appels `/api` vers
  `http://127.0.0.1:8000`;
- en production, le frontend doit recevoir `VITE_API_BASE` avec l'URL publique
  du Space Hugging Face, par exemple `https://<space>.hf.space`;
- le backend garde les parties actives en memoire; un redemarrage du service
  coupe donc les sessions en cours.

Endpoints backend principaux:

| Methode | Endpoint | Role |
|---|---|---|
| `GET` | `/api/health` | verifier que le backend repond |
| `GET` | `/api/policies` | lister les IA jouables |
| `GET` | `/api/rules` | recuperer les regles affichees dans l'app |
| `POST` | `/api/player-profile/dialogue` | creer le dialogue du Qadi selon le profil |
| `POST` | `/api/games` | creer une nouvelle partie |
| `GET` | `/api/games/{game_id}` | recuperer l'etat courant d'une partie |
| `GET` | `/api/games/{game_id}/replay` | recuperer le replay omniscient final |
| `POST` | `/api/games/{game_id}/play` | jouer l'action humaine |
| `POST` | `/api/games/{game_id}/ai-step` | avancer un tour IA |
| `POST` | `/api/games/{game_id}/next-round` | lancer la manche suivante |

Recuperation et stockage des donnees:

- `love_letter_web/frontend/src/api.js` centralise les appels `fetch`;
- les donnees de profil, de configuration IA et d'audio cote navigateur sont
  stockees en `localStorage`;
- les stats joueur et evenements de partie cote backend sont ecrits dans
  `love_letter_web/logs/player_stats.json` et
  `love_letter_web/logs/game_events.jsonl`;
- `love_letter_web/logs/` est ignore par Git pour ne pas publier les donnees
  locales de test ou de jeu.

## Handoff GitHub Et Documentation Finale - 3 Juin 2026

Objectif: tout mettre au propre et s'arreter.

Ce qu'on voulait ajouter:

- un README complet;
- un journal de projet;
- une checklist GitHub;
- un `.gitignore` adapte au depot;
- un commit final sur `main`.

Fichiers ajoutes ou mis a jour:

- `README.md`;
- `docs/project_journal_fr.md`;
- `docs/github_handoff_fr.md`;
- `love_letter_web/README.md`;
- `.gitignore`.

Commit de handoff:

```text
bfc0523 Document final Love Letter project handoff
```

Derniere mise a jour du README:

```text
3 juin 2026
```

Decision: le projet est proprement documente et la version GitHub s'arrete a ce
point.

## Comment Lancer

Installer les dependances Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lancer le backend:

```bash
uvicorn love_letter_web.backend.main:app --host 127.0.0.1 --port 8000
```

Lancer le frontend:

```bash
cd love_letter_web/frontend
npm install
npm run dev
```

Ouvrir:

```text
http://127.0.0.1:5173/
```

## Verification

Tests backend:

```bash
python3 -m pytest tests/test_love_letter_web_backend.py
```

Build frontend:

```bash
cd love_letter_web/frontend
npm run build
```

Sanity checks moteur:

```bash
python3 scripts/debug/check_engine_invariants.py
python3 scripts/debug/check_rules_conformance.py
```
