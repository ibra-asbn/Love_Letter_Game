# Diagnostic Step 3 Redo - Pairwise Action-Value Ranker

Date: 2026-04-25

## Etat de depart

Base testee:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Objectif de ce redo: produire un vrai Step3 rapide/autonome qui ameliore
significativement Step2. Le `curriculum_phase1.pth` reste une reference privee
d'evaluation, mais ne conditionne pas les choix du projet.

Le signal initial reste bon: la policy runtime `rollout-guided` bat Step2 de
`+2.29` points composite sur `1000` parties/config, seed `783000`.

```text
Step2 brut              0.25160
Step3 rollout-guided    0.27450
```

Le probleme n'est donc pas "les rollouts ne trouvent rien". Le probleme est:
comment distiller ce signal dans une tete rapide sans casser la politique.

## Hypothese utilisateur integree

Tu as pointe un point important: les rollouts n'ont pas la meme fiabilite selon
l'avancement de la partie.

- Debut de partie: horizon long, beaucoup de cartes cachees, labels tres bruites.
- Fin de partie: moins de cartes, plus de cartes sorties, plus de risque
  d'elimination, donc rollouts plus coherents.

J'ai donc ajoute deux choses:

- une ponderation d'entrainement par avancement de partie;
- un filtre d'inference optionnel pour ne laisser la tete intervenir que plus
  tard dans la partie.

## Ce qui a ete implemente

Scripts aujourd'hui archives:

```text
step3_action_value/legacy/obsolete_heads/train_pairwise_ranker.py
step3_action_value/legacy/obsolete_heads/evaluate_pairwise_ranker.py
```

Principe:

- Step2 propose son action.
- On genere des actions candidates.
- On evalue ces candidates par rollouts determinises.
- On entraine un ranker `Q(s,a)` qui apprend a ordonner les actions candidates.
- A l'inference, Step2 reste l'action par defaut; le ranker override seulement
  si son meilleur coup depasse l'action Step2 d'une marge.

Ajouts importants:

- collecte equilibree par categorie et composition adverse;
- features: observation, hidden Step2, belief, action candidate, action Step2,
  action heuristique;
- loss mixte MSE centree + listwise + pairwise ranking;
- poids de confiance lie a l'avancement de partie;
- filtres d'inference: `--min-played-cards`, `--min-deck-progress`,
  `--max-active-players`.

## Resultats principaux

Baseline seed `124000`, `1000` parties/config:

```text
Step2 = 0.26950 composite
```

| Tentative | Reglage | Composite | Delta vs Step2 | Lecture |
|---|---:|---:|---:|---|
| Attempt2 | early-stop, non balance, margin 0.12 | 0.27920 | +0.00970 | meilleur gain confirme mais faible |
| Attempt4 | balance + stage-weighted, all cards, margin 0.16 | 0.27760 | +0.00810 | gain faible, regression vs full random/full heuristic |
| Attempt4 | guard+baron, margin 0.16 | 0.27370 | +0.00420 | 300 games prometteur, 1000 games decevant |
| Attempt4 | guard only, margin 0.16 | 0.27210 | +0.00260 | signal trop petit |

Sur seed `123000`, `300` parties/config, certains reglages semblaient forts:

| Tentative | Composite | Delta vs Step2 |
|---|---:|---:|
| Attempt4 all cards | 0.27367 | +0.02133 |
| Attempt4 guard+baron | 0.27767 | +0.02533 |

Mais ces gains ne se confirment pas a `1000` parties/config. Ce sont donc des
signaux reels mais trop instables pour declarer un nouveau champion.

## Test specifique "fin de partie"

J'ai teste le meilleur ranker stage-weighted avec des filtres d'inference.

Seed `123000`, `300` parties/config:

| Filtre | Composite | Delta vs Step2 |
|---|---:|---:|
| Aucun filtre | 0.27367 | +0.02133 |
| Au moins 5 cartes sorties | 0.26300 | +0.01067 |
| Au moins 7 cartes sorties | 0.26367 | +0.01133 |
| Deck progress >= 0.45 | 0.26267 | +0.01033 |
| Au moins une elimination | 0.25300 | +0.00067 |

Conclusion: ton intuition est juste pour la fiabilite des rollouts, mais le
filtre seul reduit trop la couverture. Il rend le ranker plus prudent, sans
transformer le gain en champion stable.

## Tentative high-confidence

J'ai aussi lance une tentative plus stricte:

```text
step3_pairwise_ranker_attempt5_guard_baron_confident.pth
```

Reglages:

- seulement `guard` et `baron`;
- `32` rollouts/action;
- pairwise gap minimal `0.15`;
- loss plus orientee ranking que MSE/listwise.

Resultat `300` parties/config, seed `123000`:

```text
Composite = 0.26067
Delta vs Step2 = +0.00833
```

Ce n'est pas un succes. Le modele devient encore mal calibre: il repere des
actions differentes, mais ses marges internes ne correspondent pas assez bien a
un vrai gain arena.

## Ce qui bloque

1. Les labels de rollout sont encore trop bruites.

Avec 16 ou meme 32 rollouts/action, beaucoup d'ecarts sont proches du bruit
statistique. Le modele apprend alors des preferences locales qui ne tiennent pas
en arena.

2. Le ranker apprend une valeur absolue mal calibree.

Il peut classer des actions correctement dans certains etats, mais sa marge
predite n'est pas fiable. C'est pour ca que les seuils d'override marchent sur
300 parties puis retombent sur 1000.

3. Le offline distillation est en decalage avec l'usage reel.

On entraine sur des etats produits par Step2. Une fois que le ranker override,
il visite legerement d'autres trajectoires. Sans aggregation on-policy
type DAgger, les erreurs de distribution s'accumulent.

4. Les cartes n'ont pas le meme niveau de signal.

Guard est le signal le plus clair. Baron est utile mais plus instable. Prince
ajoute souvent du bruit dans la version actuelle. Les gains viennent surtout des
configs mixtes, pas assez de `vs_3H`.

5. Love Letter est un jeu a information imparfaite.

Les methodes qui marchent le mieux sur ce type de probleme apprennent souvent
du regret / une strategie moyenne par self-play, pas seulement une value one-step
locale. Les pistes serieuses a garder en tete sont NFSP et Deep CFR:

- Heinrich & Silver, "Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games" / NFSP:
  https://discovery.ucl.ac.uk/id/eprint/1523603/
- Brown et al., "Deep Counterfactual Regret Minimization":
  https://papers.cool/arxiv/1811.00164

## Verdict

Echec partiel, pas succes.

On a bien identifie et exploite un signal action-value, mais on n'a pas encore
un nouveau checkpoint Step3 autonome qui ameliore Step2 de maniere assez nette
et stable.

Meilleur distille confirme:

```text
step3_action_value/checkpoints/step3_pairwise_ranker_attempt2_early4.pth
```

Mais son gain confirme n'est que d'environ `+0.97` point composite. Ce n'est
pas assez pour le declarer nouveau champion.

Meilleur joueur decisionnel connu:

```text
Step2 + rollout-guided search
```

Mais ce n'est pas un checkpoint autonome.

## Prochaine bifurcation recommandee

Ne pas continuer a bricoler des thresholds.

La prochaine vraie tentative Step3 devrait etre:

1. garder `rollout-guided` comme oracle/teacher;
2. collecter un dataset plus propre avec incertitude:
   - plusieurs batches de rollouts independants par action;
   - intervalle de confiance ou test de separation;
   - labels uniquement quand l'action candidate bat Step2 hors bruit;
3. apprendre une tete de regret/advantage, pas une value brute:
   - prediction `A(s,a) = Q(s,a) - Q(s, action_step2)`;
   - calibration explicite de la marge;
   - loss pairwise uniquement sur paires separees statistiquement;
4. faire du DAgger:
   - entrainer une premiere tete;
   - la laisser jouer;
   - recollecter les etats qu'elle cree;
   - redemander a l'oracle rollout sur ces etats;
5. si cela reste instable, passer a une approche self-play/regret plus proche
   de NFSP/Deep CFR, parce que le jeu est imparfait-information et multi-agent.

Conclusion courte: on a progresse en comprehension, pas encore en champion
autonome. Le blocage n'est pas une absence de signal strategique; c'est la
distillation fiable de ce signal.
