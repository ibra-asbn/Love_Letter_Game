# Sonde - Biais De Siege Dans L'Arena Player0

Date: 2026-04-25.

Question: pourquoi `HeuristicBot` ne gagne qu'environ `12%` contre trois autres
`HeuristicBot`, alors qu'une partie symetrique devrait plutot tourner autour de
`25%` par joueur ?

## Reponse Courte

L'arena actuelle evalue toujours `player_0`. Elle ne fait pas tourner le modele
sur les quatre positions.

Le premier joueur de la manche est bien tire aleatoirement par le moteur, donc
`player_0` ne commence pas toujours. Le probleme vient surtout du ciblage de
`HeuristicBot`: quand plusieurs cibles sont equivalentes, il parcourt les
joueurs dans l'ordre absolu `player_0, player_1, player_2, player_3`.

Donc les trois heuristiques adverses ciblent tres souvent `player_0`.

## Verification Moteur

Dans `love_letter/engine.py`, le premier joueur est tire aleatoirement au reset:

```python
starting_agent = np.random.choice(self.possible_agents)
```

Dans `step2_rl_finetune/evaluate_step2.py`, les winrates reportes sont ceux de
`player_0` uniquement:

```python
if agent == "player_0":
    reward0 += float(reward)
```

## Sonde Full Heuristique

Sonde sur `5000` parties avec quatre `HeuristicBot`, `starting_agent` aleatoire.

| Siege | Winrate | Reward moyen | Ciblages recus |
|---|---:|---:|---:|
| player_0 | 12.18% | 0.2053 | 16678 |
| player_1 | 21.96% | 0.3395 | 11318 |
| player_2 | 36.98% | 0.5445 | 6663 |
| player_3 | 46.30% | 0.6734 | 4147 |

Le demarrage aleatoire est bien equilibre:

```text
player_0: 1254
player_1: 1246
player_2: 1280
player_3: 1220
```

Donc le probleme n'est pas que `player_0` commencerait toujours. Le probleme est
le biais de ciblage absolu.

## Interpretation

Le score `HeuristicBot vs 3H = 11-12%` dans nos arenas signifie:

> performance de `HeuristicBot` assis en `player_0`, dans un environnement ou
> les heuristiques adverses ciblent preferentiellement `player_0`.

Ce n'est pas une estimation neutre de la force moyenne de `HeuristicBot`.

## Consequence Pour Nos Benchmarks

Les comparaisons `player_0 only` restent utiles comme stress test, parce que
Step2 et Step3 subissent le meme siege defavorable. Mais elles ne suffisent pas
pour mesurer la force globale d'un joueur de Love Letter.

## Correction Testee

On a ajoute un parametre optionnel a `HeuristicBot`:

```python
HeuristicBot(shuffle_targets=True)
```

Ce mode conserve les memes regles, mais melange les adversaires valides avant
de resoudre les egalites de cible. L'ancien comportement reste le defaut:

```python
HeuristicBot(shuffle_targets=False)
```

Sonde full heuristique sur `5000` parties avec le mode fair:

| Siege | Winrate | Reward moyen | Ciblages recus |
|---|---:|---:|---:|
| player_0 | 29.24% | 0.4352 | 9903 |
| player_1 | 29.64% | 0.4436 | 10074 |
| player_2 | 30.02% | 0.4533 | 10070 |
| player_3 | 28.52% | 0.4301 | 9957 |

Le nombre de ciblages recus est redevenu quasi symetrique. Le `player_0` n'est
donc plus artificiellement focus par les trois adversaires heuristiques.

Prochaine correction recommandee:

1. soit creer une arena qui fait tourner la politique evaluee sur les quatre
   sieges;
2. soit modifier `HeuristicBot` pour casser les egalites de cible de maniere
   relative ou aleatoire;
3. idealement faire les deux: benchmark hard-mode `player_0` + benchmark
   symetrique.

Etat au 25 avril 2026: le point 2 est fait via `shuffle_targets=True`. Le point
1 reste a faire pour une evaluation definitive par rotation des sieges.
