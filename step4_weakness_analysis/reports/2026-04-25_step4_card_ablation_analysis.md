# Step4 - Analyse Des Ablations Conditionnelles Par Carte

Date: 2026-04-25.

Modele analyse:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Rapport brut:

```text
step4_weakness_analysis/reports/step3_fast_card_ablation_1000.md
```

## 1. Pourquoi Cette Ablation ?

Apres le clustering par types de parties, on avait deux hypotheses melangees:

1. certaines faiblesses viennent simplement de la dynamique du jeu ou de cartes
   difficiles;
2. certaines faiblesses viennent du style propre du modele, c'est-a-dire de sa
   maniere d'executer une carte une fois qu'il a choisi de la jouer.

Cette arena isole la deuxieme question.

Protocole:

- le modele choisit normalement quelle carte jouer;
- on ne le force jamais a jouer une carte;
- si la carte jouee correspond a l'ablation, on randomise uniquement
  l'execution fine: cible, guess, ou choix Chancelier;
- arena fair seat-rotated, `1000` parties par composition, memes seeds que le
  benchmark Step4 principal.

Ce test repond donc a:

> Quand le modele decide deja de jouer cette carte, est-ce qu'il l'execute
> mieux que le hasard ?

## 2. Resultats Globaux

Baseline Step3 rapide:

| vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---:|---:|---:|---:|---:|
| 51.90% | 46.20% | 38.40% | 34.50% | 0.39750 |

Synthese des ablations:

| Ablation | Composite | Delta | Changed games | Changed events | Lecture courte |
|---|---:|---:|---:|---:|---|
| Garde cible random, guess conserve | 0.38330 | -0.01420 | 1529 | 1808 | Competence de ciblage moderee. |
| Garde guess random, cible conservee | 0.33320 | -0.06430 | 2571 | 3893 | Tres forte competence de guess. |
| Garde cible+guess random | 0.33220 | -0.06530 | 2685 | 4144 | Le Garde est vraiment maitrise. |
| Pretre cible random | 0.39900 | +0.00150 | 651 | 678 | Effet nul, pas de correction directe. |
| Baron cible random | 0.39100 | -0.00650 | 528 | 533 | Petite competence de ciblage, pas toute l'histoire. |
| Prince cible random | 0.37470 | -0.02280 | 814 | 853 | Competence nette de ciblage Prince. |
| Roi cible random | 0.39750 | +0.00000 | 121 | 121 | Cible Roi pas expliquee par ce test. |
| Chancelier choix random | 0.39550 | -0.00200 | 919 | 979 | Choix Chancelier peu exploite ou peu sensible. |

Aucune ablation ne produit une amelioration nette. Donc on n'a pas trouve une
carte dont l'execution actuelle serait clairement pire que le hasard.

## 3. Garde

Resultats:

| Ablation Garde | Delta composite | Garde juste | Garde connu juste |
|---|---:|---:|---:|
| Normal | - | 30.50% | 87.02% |
| Cible random, guess conserve | -0.01420 | 25.19% | 81.52% |
| Guess random, cible conservee | -0.06430 | 10.39% | 12.76% |
| Cible+guess random | -0.06530 | 9.70% | 7.88% |

Lecture:

- Le modele sait tres clairement choisir le **guess**.
- Le ciblage compte aussi, mais beaucoup moins que la carte devinee.
- La ligne `Pretre -> Garde` reste haute dans les ablations, mais le nombre
  d'occasions change: le signal principal est surtout dans le hit-rate global.

Decision pour l'entrainement:

- Ne pas toucher lourdement au Garde.
- Proteger cette competence avec KL/trust-region dans tout futur entrainement.
- Si on travaille le Garde, ce sera pour des cas rares de cible, pas pour
  re-apprendre le guess global.

## 4. Pretre

Resultats:

| Ablation | Composite | Delta | Changed events |
|---|---:|---:|---:|
| Pretre cible random | 0.39900 | +0.00150 | 678 |

Metriques tactiques:

| Mode | Pretre->Garde juste | Garde juste |
|---|---:|---:|
| Normal | 92.12% | 30.50% |
| Pretre cible random | 91.63% | 30.39% |

Lecture:

Randomiser la cible du Pretre ne degrade pas le modele. Cela peut vouloir dire:

- beaucoup de cibles de Pretre sont equivalentes;
- l'information obtenue est souvent utile quelle que soit la cible;
- ou le modele ne choisit pas specialement bien qui regarder.

Decision pour l'entrainement:

- Ne pas entrainer Pretre directement sur cette base.
- Avant toute correction, faire un audit regret oracle: pour les etats Pretre,
  comparer les cibles par rollouts CRN.
- Si les rollouts ne trouvent pas de gros regret entre cibles, on laisse Pretre
  tranquille.

## 5. Baron

Resultats:

| Ablation | Composite | Delta | Baron gagne | Baron perdu |
|---|---:|---:|---:|---:|
| Normal | 0.39750 | - | 73.56% | 24.12% |
| Baron cible random | 0.39100 | -0.00650 | 71.04% | 26.33% |

Lecture:

Le modele a une petite competence de ciblage Baron: randomiser la cible degrade
un peu le composite et augmente les Barons perdus. Mais le delta est modere.

Cela veut dire que notre faiblesse `Baron avec petite carte` ne vient
probablement pas seulement de la cible. Elle peut venir de:

- mains ou le Baron est presque force;
- timing de jeu du Baron;
- mauvaise evaluation du risque quand la carte gardee est faible;
- manque d'usage du belief pour refuser certains duels.

Decision pour l'entrainement:

- Ne pas juste entrainer une tete `target Baron`.
- Faire un audit `forcedness`: quand il joue Baron avec carte <= 4, avait-il
  une alternative legale raisonnable ?
- Puis seulement relabeliser par oracle les Barons faibles non-forces.

## 6. Prince

Resultats:

| Ablation | Composite | Delta | Changed events |
|---|---:|---:|---:|
| Prince cible random | 0.37470 | -0.02280 | 853 |

Lecture:

Le choix de cible du Prince est une vraie competence. Randomiser soi/adversaire
degrade nettement. C'est un resultat important parce que Prince semblait faible
early dans le clustering, mais cette ablation dit:

> Quand le modele decide de jouer Prince, son choix de cible vaut mieux que le hasard.

La faiblesse Prince early vient donc probablement plus du timing ou du contexte
de la carte que du ciblage pur.

Decision pour l'entrainement:

- Proteger la competence de ciblage Prince.
- Investiguer surtout `quand jouer Prince`, pas seulement `sur qui`.
- Pour les Princes early faibles, auditer les alternatives: jouer l'autre carte,
  se cibler, cibler adversaire, ou attendre une meilleure fenetre.

## 7. Roi

Resultats:

| Ablation | Composite | Delta | Eligible games | Changed games |
|---|---:|---:|---:|---:|
| Roi cible random | 0.39750 | +0.00000 | 464 | 121 |

Lecture:

Le Roi etait un archetype faible dans le clustering, mais randomiser la cible
ne change rien au global. En plus, il n'y a que `121` parties ou l'action a
vraiment change, car Roi est rare et souvent contraint par peu de cibles.

Donc la faiblesse Roi n'est pas expliquee par ce simple test. Les causes plus
probables:

- le modele joue Roi dans de mauvais timings;
- l'echange donne trop souvent une bonne carte;
- l'information reciproque creee par Roi est sous-estimee;
- certaines situations de Roi sont simplement de mauvaises mains subies.

Decision pour l'entrainement:

- Pas d'entrainement direct sur cible Roi pour l'instant.
- Faire un audit Roi decisionnel: carte donnee, carte recue, cible, phase,
  alternatives legales, belief, et regret rollout.
- Si le regret montre des cibles meilleures, entrainer une correction cible.
- Sinon, travailler le timing du Roi ou les alternatives a jouer Roi.

## 8. Chancelier

Resultats:

| Ablation | Composite | Delta | Changed games | Changed events |
|---|---:|---:|---:|---:|
| Chancelier choix random | 0.39550 | -0.00200 | 919 | 979 |

Metriques locales:

| Mode | Chancelier garde plus haute | Chancelier pioche connue gagne |
|---|---:|---:|
| Normal | 897 / 1250 = 71.76% | 65.03% |
| Chancelier random | 485 / 1252 = 38.74% | 58.60% |

Lecture:

C'est le resultat le plus subtil. Randomiser le choix Chancelier change presque
1000 decisions et detruit clairement une heuristique locale simple
(`garder la plus haute`). Pourtant le composite ne baisse que de `-0.00200`.

Deux interpretations possibles:

1. Beaucoup de choix Chancelier sont reellement proches en valeur.
2. Le modele ne convertit pas encore assez bien le potentiel du Chancelier en
   victoire, donc randomiser ne coute pas beaucoup.

Vu la finesse strategique du Chancelier, la deuxieme hypothese est tres
plausible. Le Chancelier demande de raisonner sur:

- carte gardee maintenant;
- cartes remises au fond;
- ordre exact de remise;
- qui peut repiocher ces cartes;
- risque Princesse;
- valeur de garder une carte moyenne pour eviter d'etre lisible.

Decision pour l'entrainement:

- Chancelier devient un candidat prioritaire pour un teacher specialise.
- Mais on ne doit pas entrainer seulement "garder la plus haute", car ce serait
  trop simpliste.
- Il faut d'abord lancer un audit oracle Chancelier: pour chaque pool, comparer
  toutes les actions `900-905` par rollouts CRN, avec phase et ordre de pioche.

## 9. Conclusion Operationnelle

Ce que le modele maitrise deja:

- le guess du Garde;
- une partie du ciblage Garde;
- le ciblage Prince;
- les conversions information -> Garde.

Ce qui est ambigu et demande regret oracle:

- Pretre cible;
- Roi cible/timing;
- Chancelier choix;
- Baron avec carte faible.

Ce qu'il ne faut pas faire:

- ne pas re-entrainer globalement toutes les cartes;
- ne pas corriger le Garde lourdement;
- ne pas conclure que Chancelier est inutile simplement parce que l'ablation ne
  baisse pas le composite.

Prochaine action recommandee:

1. Construire un audit decisionnel `Roi / Baron faible / Chancelier`.
2. Pour chaque decision, mesurer `forcedness` et regret rollout CRN.
3. Entraîner seulement sur les etats non-forces ou le regret oracle est clair.
4. Utiliser une contrainte trust-region vers Step3 rapide pour proteger Garde
   et Prince, qui sont deja des competences acquises.

