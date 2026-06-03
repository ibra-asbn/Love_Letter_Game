# Love Letter RL - Journal De Projet

Etat consolide le 2026-06-03.

Ce document est le point d'entree narratif du projet. Il resume, etape par
etape, ou nous en etions, ce que nous voulions ajouter, ce qui a ete mesure,
et la decision prise. Les rapports complets restent dans les dossiers
`diagnostics/`, `step*_*/reports/` et les `README.md` de chaque etape.

## Objectif Final

Construire un agent fort pour jouer a Love Letter, puis le rendre jouable dans
une experience web propre.

Le projet est maintenant en mode arret/documentation:

- conserver le champion courant `champion_cbp`;
- publier un repo comprehensible;
- garder les artefacts lourds hors Git;
- ne pas relancer de nouvelle phase d'entrainement avant une reprise explicite;
- preparer une communication LinkedIn claire.

## Carte De Lecture

| Sujet | Fichier |
|---|---|
| Etat global historique | `README.md` |
| Regles locales du jeu | `docs/love_letter_rules_fr.md` |
| Journal canonique | `docs/project_journal_fr.md` |
| Handoff GitHub | `docs/github_handoff_fr.md` |
| Brouillon LinkedIn | `docs/linkedin_post_fr.md` |
| Web app | `love_letter_web/README.md` |
| Etapes IA | `step1_.../README.md` a `step7_.../README.md` |

## Baseline - Le Jeu Est-Il Exploitable ?

Question: une politique simple peut-elle faire mieux que le hasard ?

Protocole: `20 000` parties par politique.

| Politique player_0 | Adversaires | Winrate | Reward moyen |
|---|---|---:|---:|
| Random | 3 randoms | 29.92% | 0.397 |
| HeuristicBot | 3 randoms | 45.69% | 0.677 |

Decision: oui, Love Letter est exploitable. Il existe un edge tactique net. Le
projet peut passer a l'imitation puis au RL.

Reference: `README.md`, section `Baseline Statistique`.

## Etape 1 - Maitrise De L'Heuristique

Objectif: obtenir un modele qui copie proprement `HeuristicBot` avant de tenter
de le depasser.

Ajout important: correction de l'observation du Chancelier, afin que le modele
voie les cartes disponibles et l'ordre des choix.

Checkpoint:

```text
step1_heuristic_mastery/checkpoints/heuristic_student_attempt4_player0_chancellor_order.pth
```

Resultats:

| Mesure | Resultat |
|---|---:|
| Test action accuracy | 98.94% |
| Exact action accuracy sur comparaison longue | 99.72% |
| Composite student | 0.22448 |
| Composite HeuristicBot | 0.22556 |

Decision: succes. Le modele ne bat pas encore l'heuristique, mais il la copie
assez bien pour servir de warm start.

Reference: `step1_heuristic_mastery/README.md`.

## Etape 2 - Battre L'Heuristique

Objectif: partir du student heuristique et obtenir un modele qui bat vraiment
`HeuristicBot`.

Idee qui a marche: le belief du modele contenait deja un signal utile. On a
distille un mode contre-factuel `retarget` dans l'actor.

Checkpoint:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Confirmation longue, `5000` parties par configuration:

| Modele | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.54% | 36.80% | 26.36% | 16.04% | 0.26738 |
| HeuristicBot | 46.02% | 33.48% | 21.36% | 11.30% | 0.22226 |

Decision: succes valide. Step2 devient le socle de la pipeline nettoyee.

Reference: `step2_rl_finetune/README.md`.

## Etape 3 - Action-Value Et Recherche Locale

Objectif: apprendre quand une action alternative a plus de valeur que l'action
naturelle du Step2.

Deux branches ont ete conservees:

- Step3 hybride verify16: plus fort que Step2 mais lent, car il utilise des
  rollouts/recherche locale a l'inference;
- Step3 rapide DAgger: autonome, sans rollouts a l'inference.

Checkpoint hybride:

```text
step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth
```

Checkpoint rapide:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Resultat hybride, moyenne de trois seeds:

| Modele | Composite |
|---|---:|
| Step3 hybride verify16 | 0.28003 |
| Step2 | 0.26543 |

Resultat rapide DAgger, `5000` parties par composition:

| Modele | Composite |
|---|---:|
| DAgger iter1 | 0.27226 |
| Step2 | 0.26438 |

Decision: Step3 est un vrai progres. La branche hybride prouve le potentiel de
l'action-value. La branche rapide devient la base pratique pour l'app, mais
elle doit etre protegee contre les regressions, surtout contre `3H`.

Reference: `step3_action_value/README.md`.

## Interlude - Verification Du Biais D'Arene

Question: les modeles sont-ils vraiment sous l'heuristique, ou l'arene est-elle
biaisee ?

Constat: l'ancienne arena `player_0 only` favorisait artificiellement certains
comportements du `HeuristicBot`. Une arena fair avec rotation de sieges et
`HeuristicBot(shuffle_targets=True)` donne une lecture plus saine.

Decision: garder l'ancienne arena comme hard mode, mais utiliser l'arene fair
seat-rotated comme reference principale.

References:

- `interlude_heuristic_comparison/README.md`
- `interlude_heuristic_comparison/seat_bias_probe.md`

## Etape 4 - Identifier Les Faiblesses

Objectif: ne pas entrainer tout de suite. Comprendre ou Step3 rapide gagne,
perd, et quelles competences sont deja a proteger.

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

Lecture:

- Garde et Prince sont deja des competences fortes a proteger.
- Baron faible, Roi, Pretre et Chancelier demandent des corrections locales.
- Un nouveau PPO global n'est pas le meilleur prochain pas.

Decision: passer a des tetes d'execution specialisees.

Reference: `step4_weakness_analysis/README.md`.

## Etape 5 - Tetes D'Execution

Objectif: ameliorer Step3 rapide sans changer sa decision principale de carte.
Quand le modele choisit deja une carte, une petite tete locale corrige
l'execution.

Modules testes:

- Chancelier V1: choix de carte et ordre de remise.
- Baron V1: correction locale du duel et de la decision `jouer Baron` vs
  `jouer l'autre carte`.
- Prince V1: signal positif, mais module encore candidat.

Evaluation combinee Chancelier + Baron, `5000` parties par composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 51.68% | 44.16% | 38.76% | 34.00% | 0.39228 |
| Step3 + Chancelier V1 | 52.62% | 45.46% | 39.80% | 35.72% | 0.40582 |
| Step3 + Baron V1 | 52.48% | 45.74% | 39.00% | 35.16% | 0.40160 |
| Step3 + Chancelier + Baron | 53.36% | 46.86% | 39.96% | 37.00% | 0.41496 |

Decision: succes. Le joueur de reference devient une composition, pas un seul
checkpoint: `champion_cbp`.

Reference: `step5_execution_heads/README.md`.

## Etape 6 - Self-Play Et Population

Objectif: verifier que le champion tient contre son lignage avant de lancer une
ligue de self-play.

Champion courant:

```text
champion_cbp = Step3 rapide + Chancelier V1 + Baron V1 + Prince V1
```

Evaluation de lignage, `5000` manches:

| Politique | Score >=1 | Victoire manche | Reward moyen |
|---|---:|---:|---:|
| Champion CBP | 32.28% | 28.36% | 0.5010 |
| Step3 seul | 28.96% | 25.08% | 0.4567 |
| Step2 | 28.50% | 26.02% | 0.4416 |
| Heuristique fair | 25.28% | 21.58% | 0.3811 |

Point de vigilance: contre trois copies de l'ancien `curriculum_phase1`, le
resultat reste quasi a egalite. Ce checkpoint historique doit rester dans la
population de reference.

Decision: lancer une ligue de self-play, mais garder `curriculum_phase1` comme
sparring partner fort.

Reference: `step6_self_play/README.md`.

## Etape 7 - Ligue Self-Play

Objectif: maintenir une population active de politiques et promouvoir un
candidat seulement s'il bat le champion sur Elo et garde-fous tactiques.

Roster initial:

- `champion_cbp`
- `curriculum_phase1`
- `step3_fast`
- `step2_retarget`
- `heuristic_fair`

Bootstrap Elo initial, `10 000` manches:

| Policy | Elo bootstrap |
|---|---:|
| `champion_cbp` | 1546.7 |
| `step2_retarget` | 1520.5 |
| `curriculum_phase1` | 1506.8 |
| `heuristic_fair` | 1479.5 |
| `step3_fast` | 1446.5 |

Deux candidats ont ete lances depuis `champion_cbp`:

- `sp_iter_0001`: rejet;
- `sp_iter_0002`: rejet.

Pour `sp_iter_0002`, les garde-fous tactiques passent, mais l'Elo reste sous
le meilleur actif:

| Mesure | Valeur |
|---|---:|
| Candidate Elo | 1508.66 |
| Best Elo | 1519.66 |
| Candidate main round win rate | 28.37% |
| Best main round win rate | 27.14% |

Decision: self-play operationnel, mais aucun candidat n'a remplace le champion.
Le projet s'arrete donc avec `champion_cbp` comme reference.

Reference: `step7_self_play_league/README.md`.

## Web App

Objectif: remplacer Streamlit par une experience jouable.

Etat actuel:

- backend FastAPI;
- frontend React/Vite;
- menu Qadi, intro video, musique, regles, cartes et tutoriel;
- profils joueurs, raison d'entree, dialogues personnalises;
- choix de politique IA par adversaire;
- logs structures, stats locales, replay omniscient en fin de partie;
- champion par defaut: `champion_cbp`.

Decision: l'app web devient la cible produit. Streamlit reste un prototype ou
outil de debug.

Reference: `love_letter_web/README.md`.

## Etat Final Du Projet

Ce qui est termine:

- moteur Love Letter audite et corrige contre les regles;
- pipeline IA documentee de l'heuristique au self-play;
- champion courant defini comme `champion_cbp`;
- app web jouable avec backend et frontend;
- tests backend autour des profils, logs, replay et actions speciales;
- rapports experimentaux conserves.

Ce qui est volontairement arrete:

- pas de nouveau PPO global;
- pas de nouveau cycle self-play sans objectif precis;
- pas de tentative de fusionner toutes les tetes dans un actor unique;
- pas d'ajout des checkpoints lourds dans Git classique.

Prochaine reprise possible:

- brancher une evaluation longue automatisee pour `champion_cbp`;
- convertir les meilleurs checkpoints en assets de release ou Git LFS;
- finir le polish UI/UX et tester l'app de bout en bout;
- relancer Step7 avec plus de diversite de population.

