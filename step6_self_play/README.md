# Step6 - Self-Play Et Population

Date: 2026-04-26.

## Point De Depart

Le champion courant est fige comme une composition, pas comme un unique
checkpoint:

- acteur rapide Step3: `step3_advantage_v2_dagger_attempt1_iter1.pth`;
- tete Chancelier V1:
  `step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth`;
- specialiste Baron V1;
- specialiste Prince V1.

Dans les rapports, ce champion est note `champion_cbp`.

## Intention

Les arenas contre randoms et heuristiques restent utiles comme tests de
regression, mais elles ne suffisent plus pour mesurer un bot qui commence a
avoir des comportements tactiques locaux. Avant de lancer un entrainement en
self-play, on veut verifier comment le champion se comporte dans une table plus
riche, composee de son lignage direct.

## Phase A - Evaluation De Lignage

Arena fixe:

- `champion_cbp`: Step3 + Chancelier + Baron + Prince;
- `step3_fast`: Step3 rapide seul;
- `step2_retarget`: modele Step2;
- `heuristic_fair`: heuristique avec ciblage aleatorise.

Cette phase est une evaluation uniquement. Elle doit nous dire si la population
est saine pour demarrer du self-play ensuite, et si un ancien modele exploite
encore une faiblesse du champion.

Script:

```text
step6_self_play/lineage_arena.py
```

Critere de lecture:

- le champion doit etre devant son lignage direct;
- aucun ancien modele ne doit l'ecraser dans une position precise;
- les sorties et les tactiques doivent expliquer le winrate, pas seulement le
  score final.

## Resultat Initial

Rapport: `reports/lineage_arena_5000_seed3400000.md`.

Sur 5000 manches, avec permutation des sieges:

| Politique | Score >=1 | Victoire manche | Reward moyen |
|---|---:|---:|---:|
| Champion CBP | 32.28% | 28.36% | 0.5010 |
| Step3 seul | 28.96% | 25.08% | 0.4567 |
| Step2 | 28.50% | 26.02% | 0.4416 |
| Heuristique fair | 25.28% | 21.58% | 0.3811 |

Conclusion: le champion domine son lignage direct dans cette table. Ce n'est
pas encore de l'entrainement, mais c'est un signal sain pour passer ensuite a
une ligue de self-play/population training. Les anciennes versions restent
utiles comme adversaires de reference, pas comme objectif final.

## Phase B - Matchups Asymetriques

Rapport: `reports/asymmetric_matchups_5000_seed3500000.md`.

Objectif: verifier si le champion tient quand il est seul contre trois copies
d'un meme adversaire, puis inverser la pression en mettant un ancien modele seul
contre trois champions.

| Matchup | Singleton | Victoire manche | Opposants/copie | Victoire manche |
|---|---|---:|---|---:|
| champion_vs_3_step3 | Champion CBP | 27.64% | Step3 seul | 24.43% |
| champion_vs_3_step2 | Champion CBP | 26.42% | Step2 | 24.93% |
| champion_vs_3_heuristic | Champion CBP | 33.14% | Heuristique fair | 22.61% |
| step3_vs_3_champions | Step3 seul | 23.42% | Champion CBP | 25.97% |
| step2_vs_3_champions | Step2 | 22.38% | Champion CBP | 26.25% |

Lecture: le champion est bien devant, mais l'ecart contre les anciens modeles
ne ressemble pas a une domination totale. Il exploite tres nettement
l'heuristique; contre Step3 et Step2, il gagne surtout par des ameliorations
tactiques locales et par une meilleure stabilite moyenne. C'est un bon signal
pour demarrer une ligue de self-play, mais cela confirme aussi qu'il faudra des
adversaires adaptatifs et pas seulement des heuristiques fixes.

## Phase C - Ancien Champion Curriculum

Rapport: `reports/champion_vs_3_curriculum_5000_seed3800000.md`.

Objectif: tester le champion courant contre trois copies de l'ancien champion
historique `models/checkpoints/curriculum_phase1.pth`.

| Matchup | Singleton | Victoire manche | Opposants/copie | Victoire manche |
|---|---|---:|---|---:|
| champion_vs_3_curriculum | Champion CBP | 25.10% | Curriculum phase1 | 25.31% |

Lecture: le resultat est quasiment a egalite, avec un tres leger avantage pour
`curriculum_phase1` par copie. Le champion actuel garde de meilleures tactiques
locales visibles, notamment Baron (`76.86%` de duels gagnes contre `70.73%`)
et Chancelier (`89.61%` de garde de la meilleure carte contre `68.63%`), mais
cela ne se transforme pas encore en domination globale contre l'ancien champion.

Conclusion: `curriculum_phase1.pth` reste un excellent sparring partner. Pour la
suite, il faut l'inclure dans la population de self-play/evaluation, meme s'il
ne conditionne pas le projet final.
