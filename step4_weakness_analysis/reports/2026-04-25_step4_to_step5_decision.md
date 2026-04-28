# Step4 - Decision De Passage Vers Step5

Date: 2026-04-25.

## Diagnostic Fige

Les analyses Step4 ont separe deux questions:

1. Est-ce que le modele gagne/perd parce que la dynamique du jeu lui donne de
   bonnes ou mauvaises cartes ?
2. Est-ce que le modele sait executer finement une carte une fois qu'il a deja
   decide de la jouer ?

Le clustering par cartes/phases a montre que le Step3 rapide est solide, mais
que certains archetypes restent faibles: Roi, Baron avec petite carte,
Chancelier, grosses cartes tot.

L'arene d'ablation conditionnelle a ensuite donne le signal cle:

| Carte / execution | Delta si randomisee | Interpretation |
|---|---:|---|
| Garde - guess | -0.06430 | Competence forte acquise. |
| Garde - cible | -0.01420 | Competence de ciblage utile. |
| Prince - cible | -0.02280 | Competence nette acquise. |
| Baron - cible | -0.00650 | Competence faible/moderee; le probleme n'est pas seulement la cible. |
| Roi - cible | +0.00000 | Faiblesse Roi non expliquee par cible seule. |
| Pretre - cible | +0.00150 | Pas de preuve d'expertise de cible. |
| Chancelier - choix | -0.00200 | Le potentiel du choix Chancelier est peu converti en winrate. |

## Decision

On ne relance pas un entrainement global.

On protege ce qui marche:

- Garde, surtout le guess;
- Prince, surtout le choix de cible;
- lignes information -> Garde.

On ouvre Step5 sur les executions ambiguës:

- Roi;
- Baron, surtout Baron avec carte faible;
- Pretre;
- Chancelier.

## Hypothese Step5

Le Step3 rapide sait souvent **quand** jouer une carte, mais il n'a pas encore
une expertise stable sur **comment** executer certaines cartes. On veut donc
apprendre des corrections locales d'execution sans changer la decision
principale de jouer la carte.

## Regle De Prudence

Une ablation qui ne change pas le winrate ne suffit pas a entrainer. Elle dit
seulement que l'execution actuelle n'est pas visiblement meilleure que le
hasard au niveau global.

Avant d'entrainer, Step5 doit mesurer le regret oracle:

- quelles alternatives etaient legales ?
- le modele etait-il force ?
- une alternative etait-elle clairement meilleure selon rollouts CRN ?
- le gain est-il suffisamment stable pour apprendre sans bruit ?

Seuls les etats non-forces avec regret oracle clair pourront devenir des labels
d'entrainement.

