# Step7 - Self-Play League

Date: 2026-04-26.

## Objectif

Construire une ligue de self-play autour du champion actuel `champion_cbp`.
La ligue maintient une composition active de 5 politiques. Quand un candidat
devient meilleur via Elo, il entre dans la composition active et la politique
active la plus faible sort de la composition, sans suppression physique des
checkpoints.

## Fichiers Principaux

- `league_roster.json`: roster actif, Elo, checkpoints et metadata.
- `league_results.jsonl`: lignes de manches evaluees.
- `promotion_history.jsonl`: decisions de promotion/rejet.
- `league_policy.py`: factory commune de politiques.
- `league_evaluate.py`: evaluation Elo multi-joueurs.
- `train_self_play_candidate.py`: PPO self-play du candidat compose.
- `league_promote.py`: gate de promotion/rejet depuis un rapport Elo.

Les checkpoints de self-play sont sauvegardes a chaque iteration, plus un
checkpoint final. Un manifeste `reports/<candidate>_checkpoint_manifest.json`
est aussi mis a jour a chaque iteration pour retrouver rapidement le dernier
checkpoint si une session plante.

## Etat Initial

Roster actif initial, taille fixe `5`:

- `champion_cbp`: Step3 + heads Chancelier/Baron/Prince.
- `curriculum_phase1`: ancien champion historique.
- `step3_fast`: Step3 seul.
- `step2_retarget`: Step2.
- `heuristic_fair`: heuristique fair, sans focus artificiel sur un siege.

Le candidat Step7 demarre par defaut depuis le meilleur parent choisi dans le
roster, actuellement `champion_cbp`. En v1, seuls l'acteur et le critic sont
entraines; l'encoder, le belief, la tete Step3 et les heads Chancelier/Baron/
Prince restent geles.

## Smoke Tests Du 2026-04-26

- Chargement et jeu de toutes les policies: `8` manches OK, rapport
  `reports/smoke_load_8.md`.
- Mini evaluation Elo: `40` manches OK, rapport `reports/smoke_eval_40.md`.
- Mini training self-play: `1` iteration, `256` decisions, checkpoint
  rechargeable `checkpoints/smoke_sp_final_candidate.pth`.
- Smoke promotion sur roster temporaire `/tmp`: mecanique OK, `smoke_sp` entre
  actif et le plus faible Elo evalue sort, en conservant `5` actifs. Cette
  promotion etait forcee pour tester la plomberie, pas une validation sportive.

## Bootstrap Elo Initial

Bootstrap officiel lance le 2026-04-26 sur `10 000` manches, `K=16`, avec mise
a jour du roster:

| Policy | Elo bootstrap |
|---|---:|
| `champion_cbp` | 1546.7 |
| `step2_retarget` | 1520.5 |
| `curriculum_phase1` | 1506.8 |
| `heuristic_fair` | 1479.5 |
| `step3_fast` | 1446.5 |

Rapports:

- `reports/bootstrap_elo_10000.md`
- `reports/bootstrap_elo_10000.json`

## Premier Cycle Self-Play

Deux candidats ont ete lances depuis `champion_cbp`:

- `sp_iter_0001`: training conservateur, KL forte vers le parent.
- `sp_iter_0002`: learning rate plus haut, KL plus faible, exploration un peu
  plus libre.

Les deux candidats ont ete rejetes par le gate de promotion: leurs garde-fous
tactiques passent, et ils ameliorent legerement la victoire principale, mais
leur Elo de ligue reste sous `champion_cbp`.

Rapports:

- `reports/sp_iter_0001_league_eval_10000.md`
- `reports/sp_iter_0001_promotion_decision.md`
- `reports/sp_iter_0002_league_eval_10000.md`
- `reports/sp_iter_0002_promotion_decision.md`

## Commandes Utiles

Smoke evaluation:

```text
python3 step7_self_play_league/league_evaluate.py --games 40 --k-factor 16 --output smoke_eval.json --markdown smoke_eval.md --no-append-results
```

Smoke training:

```text
python3 step7_self_play_league/train_self_play_candidate.py --iterations 1 --steps-per-iteration 256 --candidate-id smoke_sp --output-prefix smoke_sp
```

Bootstrap Elo initial recommande:

```text
python3 step7_self_play_league/league_evaluate.py --games 10000 --bootstrap --k-factor 16 --output bootstrap_elo_10000.json --markdown bootstrap_elo_10000.md --update-roster
```

Evaluation d'un candidat inscrit mais pas encore actif:

```text
python3 step7_self_play_league/league_evaluate.py --games 10000 --include-policy sp_iter_0001 --k-factor 8 --output sp_iter_0001_league_eval.json --markdown sp_iter_0001_league_eval.md
```

Gate de promotion:

```text
python3 step7_self_play_league/league_promote.py --candidate-id sp_iter_0001 --evaluation-report step7_self_play_league/reports/sp_iter_0001_league_eval.json
```
