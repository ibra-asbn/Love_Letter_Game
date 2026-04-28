# Step5 Prince V1 - Premiere Evaluation

Date: 2026-04-26.

## Question

On veut savoir si un correctif local Prince ajoute quelque chose au Step3 rapide.
Pour ne pas diluer le signal, l'analyse principale est conditionnelle: on regarde
les parties ou le joueur evalue a eu un Prince en main au moins une fois.

Politiques comparees:

- `Step3 rapide`;
- `Step3 + Prince V1`.

Aucune autre tete Step5 n'est branchee dans cette evaluation.

## Methode

Script:

```bash
python3 step5_execution_heads/cards/prince/evaluate_prince_specialist.py \
  --games 5000 \
  --seed-start 2500000 \
  --output prince_specialist_v1_eval_5000_seed2500000.json \
  --markdown prince_specialist_v1_eval_5000_seed2500000.md
```

Le benchmark reste l'arene fair seat-rotated:

- vs 3 randoms;
- vs 1 heuristique + 2 randoms;
- vs 2 heuristiques + 1 random;
- vs 3 heuristiques.

## Resultats

| Politique | Composite global | Winrate avec Prince en main | Parties avec Prince | Winrate si Prince joue | Parties Prince joue |
|---|---:|---:|---:|---:|---:|
| Step3 rapide | 0.39162 | 48.31% | 8176 | 52.02% | 6449 |
| Step3 + Prince V1 | 0.39428 | 48.97% | 8176 | 51.03% | 7080 |

Delta principal:

- composite global: `+0.00266`;
- condition Prince en main: `+0.66 point`;
- hit Princesse: `7.64%` -> `9.38%`;
- suicide sur soi: `0.51%` -> `0.35%`.

## Lecture Tactique

Prince V1 augmente le nombre de Princes joues:

| Politique | Pct joue Prince | Cible soi | Cible adversaire |
|---|---:|---:|---:|
| Step3 rapide | 54.99% | 42.43% | 57.57% |
| Step3 + Prince V1 | 69.46% | 54.20% | 45.80% |

Le gain est donc reel mais pas massif. Le module comprend mieux deux choses:

- recycler ses petites cartes avec Prince sur soi;
- viser un peu plus souvent la Princesse adverse.

Mais il force probablement trop le Prince, car le winrate conditionnel `si Prince
joue` baisse de `52.02%` a `51.03%`. Cela veut dire que le module transforme
certains etats ou Step3 gardait le Prince en action immediate, et que toutes ces
actions forcees ne sont pas rentables.

## Par Carte Accompagnante

| Carte avec Prince | Step3 winrate | Prince V1 winrate | Lecture |
|---|---:|---:|---|
| Espionne | 53.57% | 55.92% | Bon: recycler une carte faible aide. |
| Garde | 49.09% | 48.37% | Neutre/faible: jouer Garde reste souvent excellent. |
| Pretre | 42.76% | 47.32% | Bon signal, mais Prince V1 force beaucoup plus. |
| Baron | 37.34% | 39.21% | Bon: evite une partie des couples Baron+Prince toxiques. |
| Servante | 54.61% | 53.57% | Legere regression, peu de volume joue. |
| Prince | 47.35% | 49.40% | Bon leger. |
| Chancelier | 47.72% | 49.04% | Bon leger, mais attention a la perte de valeur Chancelier. |
| Roi | 50.05% | 49.26% | Regression: il ne faut pas traiter Roi+Prince trop simplement. |
| Comtesse | 51.10% | 51.63% | Neutre, Prince non jouable avec Comtesse forcee. |
| Princesse | 69.08% | 70.66% | Bon: cible adversaire plutot que suicide sur soi. |

## Decision

Prince V1 est un **succes leger**, pas encore une tete definitive.

On garde le module comme candidat, mais on ne le branche pas encore au joueur
Step5 de reference tant qu'il n'a pas ete raffine. La prochaine version doit
etre moins agressive:

- ne pas forcer Prince quand l'autre carte est `Garde` ou `Roi` sans signal fort;
- garder l'override fort seulement pour `Prince + petite carte`, `Prince +
Princesse`, ou Princesse adverse probable/connue;
- comparer explicitement `jouer Prince maintenant` contre `jouer l'autre carte`
sur les couples sensibles, comme on l'a fait pour Baron.

Conclusion courte: le signal Prince existe, mais la version actuelle gagne peu
car elle corrige des bons coups tout en introduisant des Princes trop impatients.
