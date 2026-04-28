# Interlude - HeuristicBot vs Modeles Actuels

Date: 2026-04-25.

Objectif initial: clarifier une question precise avant de continuer Step3:

> Est-ce que nos modeles actuels sont vraiment moins bons que `HeuristicBot`,
> ou est-ce une impression due aux comparaisons contre le champion historique et
> aux regressions locales de Step3 ?

## Modeles Compares

| Nom court | Role | Fichier |
|---|---|---|
| `HeuristicBot` | professeur / benchmark lisible | `love_letter/bots/heuristic.py` |
| `Step2 retarget` | meilleur socle neural actuel | `step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth` |
| `Step3 rapide DAgger` | tete advantage autonome, sans rollout a l'inference | `step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth` |
| `Step3 hybride verify16` | tete advantage + verification CRN locale | `step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth` |

## Protocole

Arena progressive habituelle, mais attention: cette evaluation mesure toujours
`player_0`.

```text
vs_0H_3R = player_0 vs 3 randoms
vs_1H_2R = player_0 vs 1 heuristic + 2 randoms
vs_2H_1R = player_0 vs 2 heuristics + 1 random
vs_3H    = player_0 vs 3 heuristics
```

Chaque politique passe sur les memes `3` seeds:

```text
134000, 135000, 136000
```

Soit `1000` parties par composition et par seed, donc `3000` parties par
composition et par politique.

Commande:

```bash
python3 -m interlude_heuristic_comparison.evaluate_interlude_arena \
  --games 1000 \
  --seeds 134000 135000 136000 \
  --output interlude_arena_heuristic_step2_step3_3x1000.json \
  --markdown interlude_arena_heuristic_step2_step3_3x1000.md \
  --run-log interlude_heuristic_comparison/logs/2026-04-25_interlude_arena_3x1000.md
```

Note: `Step3 hybride verify16` reutilise les rapports officiels deja calcules
sur ces memes seeds, car relancer les rollouts serait couteux et redondant.

## Alerte Biais De Siege

Cette arena ne fait pas tourner le joueur evalue sur toutes les positions.
Elle evalue toujours `player_0`.

Le moteur tire bien le premier joueur aleatoirement au reset, donc `player_0`
ne commence pas toujours. En revanche, `HeuristicBot` casse ses egalites de
ciblage en parcourant `env.possible_agents` dans l'ordre absolu
`player_0, player_1, player_2, player_3`.

Consequence: quand plusieurs cibles sont equivalentes, les bots adverses
choisissent tres souvent `player_0`. Une sonde full heuristique sur `5000`
parties donne:

| Siege | Winrate | Reward moyen | Nombre de ciblages recus |
|---|---:|---:|---:|
| player_0 | 12.18% | 0.2053 | 16678 |
| player_1 | 21.96% | 0.3395 | 11318 |
| player_2 | 36.98% | 0.5445 | 6663 |
| player_3 | 46.30% | 0.6734 | 4147 |

Donc le `11-12%` de `HeuristicBot` vs `3H` dans nos tableaux ne signifie pas
que l'heuristique vaut 12% dans une partie symetrique. Cela signifie:

> `HeuristicBot` assis en `player_0`, face a trois `HeuristicBot` qui ont un
> biais de ciblage vers `player_0`, gagne environ 12%.

Cette arena reste utile comme **hard mode player_0**, mais elle ne doit pas etre
interpretee comme une estimation neutre de force globale.

## Correction Fair Mode

On a ajoute un mode de benchmark ou `HeuristicBot` casse ses egalites de cible
aleatoirement:

```python
HeuristicBot(shuffle_targets=True)
```

Cela ne change pas la logique tactique de l'heuristique. Cela change seulement
l'ordre des adversaires quand plusieurs cibles sont equivalentes. L'ancien mode
reste disponible par defaut et devient notre **hard mode player_0**.

Sonde full heuristique sur `5000` parties avec `shuffle_targets=True`:

| Siege | Winrate | Reward moyen | Ciblages recus |
|---|---:|---:|---:|
| player_0 | 29.24% | 0.4352 | 9903 |
| player_1 | 29.64% | 0.4436 | 10074 |
| player_2 | 30.02% | 0.4533 | 10070 |
| player_3 | 28.52% | 0.4301 | 9957 |

Les sieges ne sont pas parfaitement identiques sur seulement `5000` parties,
mais le biais massif de ciblage vers `player_0` a disparu.

Commande fair arena:

```bash
python3 -m interlude_heuristic_comparison.evaluate_fair_arena \
  --games 1000 \
  --seed-start 150000 \
  --output fair_arena_heuristic_step2_step3_1000.json \
  --markdown fair_arena_heuristic_step2_step3_1000.md \
  --run-log interlude_heuristic_comparison/logs/2026-04-25_fair_arena_1000.md
```

Resultats fair mode, `1000` parties par composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Fair Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 52.80% | 40.40% | 36.60% | 28.20% | 0.35620 | +0.00000 |
| Step2 retarget | 49.10% | 43.60% | 38.30% | 31.50% | 0.37720 | +0.02100 |
| Step3 rapide DAgger | 53.40% | 45.70% | 41.30% | 28.60% | 0.38310 | +0.02690 |
| Step3 hybride verify16 | 53.00% | 46.80% | 39.90% | 30.70% | 0.38910 | +0.03290 |

Lecture: en fair mode, l'heuristique remonte logiquement tres fort contre
`3H`, de `~12%` a `28.2%`. Les modeles restent au-dessus au composite, mais
l'ecart est moins spectaculaire que dans le hard mode. Cette version devient
notre arena par defaut pour comparer des modeles contre des heuristiques qui ne
focus pas artificiellement `player_0`.

## Seat-Rotated Tactical Arena

On a ensuite ajoute le vrai benchmark propre demande: le joueur evalue tourne
entre les quatre sieges, tout en gardant les heuristiques adverses en fair mode.
Le script collecte aussi les sorties et des metriques tactiques.

Commande:

```bash
python3 -m interlude_heuristic_comparison.evaluate_rotating_tactical_arena \
  --games 1000 \
  --seed-start 260000 \
  --output rotating_tactical_arena_post_rules_fix_1000.json \
  --markdown rotating_tactical_arena_post_rules_fix_1000.md \
  --run-log interlude_heuristic_comparison/logs/2026-04-25_rotating_tactical_arena_post_rules_fix_1000.md
```

Resultats seat-rotated post-correction des regles, `1000` parties par
composition:

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 51.80% | 39.20% | 31.50% | 29.20% | 0.34150 |
| Step2 retarget | 51.70% | 43.00% | 35.70% | 33.30% | 0.37800 |
| Step3 rapide DAgger | 51.90% | 46.20% | 38.40% | 34.50% | 0.39750 |
| Step3 hybride verify16 | 51.20% | 45.30% | 39.30% | 34.30% | 0.39690 |

Sorties moyennes sur les quatre compositions:

| Politique | Gagnant | 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 37.92% | 20.60% | 16.80% | 10.90% | 13.78% |
| Step2 retarget | 40.92% | 19.62% | 15.25% | 12.53% | 11.68% |
| Step3 rapide DAgger | 42.75% | 18.90% | 15.97% | 10.78% | 11.60% |
| Step3 hybride verify16 | 42.52% | 19.20% | 15.60% | 11.25% | 11.43% |

Metriques tactiques globales:

| Politique | Garde juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier pioche connue gagne | Espionne bonus / Espionne |
|---|---:|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 26.43% | 91.35% | 69.97% | 27.61% | 50.48% | 59.67% |
| Step2 retarget | 35.57% | 93.49% | 75.56% | 22.66% | 63.08% | 55.05% |
| Step3 rapide DAgger | 30.50% | 92.12% | 73.56% | 24.12% | 65.03% | 55.37% |
| Step3 hybride verify16 | 34.14% | 94.26% | 74.86% | 23.23% | 61.74% | 57.01% |

Lecture: une fois les sieges tournes, Step2 reste au-dessus de l'heuristique,
et les deux Step3 restent au-dessus de Step2. Le Step3 rapide finit tres
legerement devant le Step3 hybride au composite sur ce run, mais l'ecart entre
les deux est trop faible pour en faire une hierarchie definitive. Le signal
important est que les deux Step3 sont nettement devant l'heuristique fair.

## Resultats

### Hard Mode Player0

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite | Delta vs Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| HeuristicBot | 46.40% | 32.57% | 21.50% | 11.83% | 0.22337 | +0.00000 |
| Step2 retarget | 51.57% | 37.50% | 24.73% | 16.17% | 0.26543 | +0.04207 |
| Step3 rapide DAgger | 53.87% | 38.27% | 26.97% | 16.63% | 0.27783 | +0.05447 |
| Step3 hybride verify16 | 53.30% | 38.87% | 26.47% | 17.40% | 0.28003 | +0.05667 |

Les deltas composites sont superieurs aux incertitudes approximees:

| Politique | Delta composite vs HeuristicBot | IC95 approx du delta |
|---|---:|---:|
| Step2 retarget | +0.04207 | +/- 0.01094 |
| Step3 rapide DAgger | +0.05447 | +/- 0.01103 |
| Step3 hybride verify16 | +0.05667 | +/- 0.01107 |

## Conclusion Revisee

Sur cette arena progressive `player_0 only`, nos modeles ne sont pas moins bons
que `HeuristicBot`: ils sont nettement au-dessus du `HeuristicBot` place dans la
meme position defavorable.

Mais cette conclusion ne suffit pas a dire que les modeles sont globalement
superieurs dans une partie symetrique. La confusion venait du fait qu'on
melangeait plusieurs questions:

- battre `HeuristicBot`;
- battre Step2;
- battre le champion historique `curriculum_phase1.pth`;
- avoir une Step3 rapide qui progresse sans regression contre `3H`.
- mesurer une vraie force moyenne en faisant tourner les sieges ou en retirant
  le biais de ciblage absolu de l'heuristique.

La bonne lecture est donc:

- Step2 a bien depasse l'heuristique.
- Les deux Step3 depassent encore Step2 sur la comparaison seat-rotated.
- Step3 rapide DAgger finit legerement devant Step3 hybride sur ce run, mais
  l'ecart est trop faible pour declarer une hierarchie definitive.
- Step3 hybride verify16 garde une lecture tactique interessante: meilleur taux
  de Barons favorables et meilleur lien Pretre -> Garde, mais il utilise de la
  recherche a l'inference.

La prochaine question n'est pas seulement "est-ce qu'on bat l'heuristique en
`player_0 hard mode` ?", mais:

> Comment evaluer proprement la force moyenne par siege, puis transformer le
> gain de Step3 hybride en modele rapide autonome robuste ?

Apres correction fair mode, la reponse est plus precise:

- l'ancien hard mode etait tres defavorable a `player_0`;
- ce hard mode reste utile pour mesurer la resistance au focus adverse;
- la fair arena retire le gros biais de ciblage et montre que Step2/Step3 sont
  bien competitifs avec `HeuristicBot`;
- l'arena seat-rotated confirme ensuite que Step2/Step3 restent au-dessus de
  l'heuristique quand le joueur evalue tourne sur les quatre positions.
