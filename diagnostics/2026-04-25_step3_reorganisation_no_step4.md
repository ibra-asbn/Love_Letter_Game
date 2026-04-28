# Diagnostic - Reorganisation Step3 Sans Step4 Active

Date: 25 avril 2026.

## Constat

La branche `verify16` avait ete nommee `Step4 v0`, mais ce nom etait mauvais
pour notre pipeline. Elle ne represente pas une nouvelle etape conceptuelle:
c'est une variante Step3 qui garde la tete advantage et ajoute une verification
CRN locale a l'inference.

## Correction D'Organisation

- `verify16` devient officiellement `Step3 hybride verify16`.
- Le checkpoint valide est conserve ici:

```text
step3_action_value/checkpoints/hybrid_verify16/step3_hybrid_verify16.pth
```

- Sa fiche de reference est ici:

```text
step3_action_value/hybrid_verified/step3_hybrid_verify16.json
```

- L'ancien dossier `step4_hybrid_champion/` est archive ici:

```text
step3_action_value/legacy/step3_hybrid_search_prototype/
```

- Les anciennes tentatives Step3 qui ne sont plus la ligne active ont ete
  deplacees dans:

```text
step3_action_value/legacy/obsolete_heads/
step3_action_value/legacy/checkpoints/
```

## Pipeline Active Apres Rangement

```text
Step1: imitation propre de HeuristicBot
Step2: retarget/belief distille, meilleur socle actuel
Step3 rapide: advantage head + DAgger, sans rollout a l'inference
Step3 hybride: advantage head + verification CRN, avec rollouts a l'inference
```

Il n'y a pas de Step4 active.

## Pourquoi Les Modeles Ne Semblent Pas Aussi Forts Que L'Heuristique

Le point important: avoir appris de l'heuristique ne signifie pas disposer de la
meme robustesse que l'heuristique dans tous les etats.

1. L'imitation exacte a 99% peut encore perdre beaucoup en partie complete.
   Love Letter amplifie les petites erreurs: un mauvais Baron, un Garde mal
   cible ou une Princesse jetee au mauvais moment peut terminer la manche.

2. Le student apprend une distribution moyenne de coups, pas les raisons
   causales. L'heuristique applique des regles dures et coherentes; le reseau
   approxime ces regles avec une politique probabiliste.

3. Des qu'une politique neuronale devie un peu, elle visite des etats que le
   professeur heuristique ne visitait presque jamais. C'est le decalage de
   distribution qui a motive DAgger.

4. Les adversaires heuristiques punissent mieux les erreurs que les randoms.
   Un gain visible contre randoms ne suffit pas; il faut verifier `1H+2R`,
   `2H+1R` et surtout `3H`.

5. Step3 corrige des coups locaux. Elle n'a pas encore appris une value globale
   de manche qui arbitre parfaitement le risque, le timing et la gestion de fin
   de deck.

En clair: Step2 et Step3 ne sont pas faibles parce qu'ils ont mal imite
l'heuristique. Ils sont limites parce que depasser une heuristique stable demande
plus qu'une imitation: il faut une politique robuste sur ses propres etats et
une estimation de valeur fiable, surtout contre trois adversaires non random.

## Etat Actuel

- Step2 bat `HeuristicBot` sur l'arene longue de reference.
- Step3 hybride verify16 bat Step2 de `+1.46` point composite en moyenne sur
  trois seeds de `1000` parties par composition.
- Step3 rapide DAgger iter1 bat Step2 de `+0.788` point composite sur `5000`
  parties par composition, mais perd encore `0.22` point vs `3H`.

Conclusion: la direction est bonne, mais la prochaine victoire importante doit
venir d'une Step3 rapide stabilisee par DAgger + trust-region/KL, pas d'une
nouvelle etape ajoutee par-dessus.
