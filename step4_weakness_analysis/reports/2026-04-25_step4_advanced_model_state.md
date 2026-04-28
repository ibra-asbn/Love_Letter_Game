# Step4 - Etat Des Lieux Avance Du Step3 Rapide

Date: 2026-04-25.

Modele analyse:

```text
step3_action_value/checkpoints/step3_advantage_v2_dagger_attempt1_iter1.pth
```

Ce document resume l'audit Step4 du modele actuel: types de parties, cartes
vues par le modele, moment de la manche, winrate, positions de sortie et
lecture strategique. Il complete le rapport brut:

```text
step4_weakness_analysis/reports/step3_fast_card_clusters_post_rules_fix_1000.md
```

## 1. Les Corrections De Regles Ont-Elles Abime Les Resultats ?

Lecture courte: non. Elles n'ont pas casse les modeles et la hierarchie reste
saine. Les scores absolus bougent un peu, ce qui est normal car le run post-fix
utilise aussi un autre bloc de seeds. Le point important est que Step2 reste
au-dessus de l'heuristique, et Step3 reste au-dessus de Step2.

| Politique | Composite avant | Composite post-fix | Delta |
|---|---:|---:|---:|
| Fair HeuristicBot | 0.36240 | 0.34150 | -0.02090 |
| Step2 retarget | 0.38570 | 0.37800 | -0.00770 |
| Step3 rapide DAgger | 0.40210 | 0.39750 | -0.00460 |
| Step3 hybride verify16 | 0.40020 | 0.39690 | -0.00330 |

Ce tableau ne doit pas etre lu comme un test A/B parfait: les seeds changent et
le moteur a ete corrige. Mais il montre que les corrections n'ont pas detruit
les politiques apprises. Le Step3 rapide reste le meilleur modele autonome du
moment dans cette arena propre.

Resultats post-fix, `1000` parties par composition:

| Politique | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Fair HeuristicBot | 51.80% | 39.20% | 31.50% | 29.20% | 0.34150 |
| Step2 retarget | 51.70% | 43.00% | 35.70% | 33.30% | 0.37800 |
| Step3 rapide DAgger | 51.90% | 46.20% | 38.40% | 34.50% | 0.39750 |
| Step3 hybride verify16 | 51.20% | 45.30% | 39.30% | 34.30% | 0.39690 |

Conclusion: le vrai changement de lecture vient surtout de la correction du
biais d'evaluation, pas d'un effondrement du modele.

## 2. Resultat Global Du Step3 Rapide

Clustering principal, aligne sur le benchmark interlude post-fix:

| Mesure | Resultat |
|---|---:|
| Parties analysees | 4000 |
| Winrate moyen | 42.75% |
| Reward moyen | 0.6447 |
| Composite arena | 0.39750 |

Quand le modele perd, sa position de sortie est:

| 1er sorti | 2e sorti | 3e sorti | Finaliste perdant |
|---:|---:|---:|---:|
| 33.01% | 27.90% | 18.82% | 20.26% |

Lecture: environ un tiers des defaites sont des sorties precoces. Ce n'est pas
automatiquement mauvais dans Love Letter, car beaucoup de sorties precoces
viennent de Gardes/Barons/Princesse et contiennent une grosse part de variance.
Mais si un archetype augmente trop les sorties precoces, il devient suspect.

## 3. Robustesse Sur Seed Independant

Un run secondaire a ete lance sur seed independant `310000`.

| Run | Composite | Winrate moyen |
|---|---:|---:|
| Principal seed 260000 | 0.39750 | 42.75% |
| Secondaire seed 310000 | 0.37240 | 40.62% |

La difference rappelle qu'un seul bloc de `4000` parties reste bruite pour une
analyse fine par sous-type de partie. Les signaux les plus credibles sont donc
ceux qui reviennent dans les deux runs.

## 4. Archetypes De Parties

Les clusters sont multi-label: une meme partie peut etre a la fois `Princesse
tot`, `Controle en fin de manche` et `Beaucoup de cartes pieges`.

| Archetype | Games | Winrate | Seed independant |
|---|---:|---:|---:|
| Comtesse volontaire | 158 | 72.78% | 67.06% |
| Pretre puis Garde | 504 | 66.47% | 64.50% |
| Beaucoup de cartes pieges | 2036 | 60.85% | 56.38% |
| Beaucoup de Gardes | 1546 | 60.54% | 56.09% |
| Partie avec Espionne | 1103 | 58.11% | 54.81% |
| Controle en fin de manche | 1717 | 57.37% | 55.37% |
| Main riche en controle | 1595 | 56.87% | 54.12% |
| Partie avec Servante | 1482 | 56.21% | 53.09% |
| Main riche en information | 2562 | 55.78% | 51.94% |
| Comtesse tot | 405 | 55.56% | 53.68% |
| Partie avec Prince | 1249 | 52.84% | 51.40% |
| Pression Baron | 1297 | 52.58% | 48.99% |
| Partie avec Chancelier | 1142 | 51.14% | 48.32% |
| Grosse carte tot | 1194 | 49.75% | 48.08% |
| Princesse tot | 445 | 48.76% | 49.45% |
| Baron avec petite carte | 1010 | 47.13% | 41.73% |
| Partie avec Roi | 464 | 44.61% | 42.74% |

### Forces Qui Semblent Solides

`Pretre puis Garde` est le signal tactique le plus propre. Il est fort sur les
deux seeds: `66.47%` puis `64.50%`. Cela indique que le modele sait vraiment
convertir une information privee en elimination quand la ligne est disponible.

`Beaucoup de Gardes` est aussi robuste: `60.54%` puis `56.09%`. Le Garde n'est
pas juste une carte de hasard pour lui; dans les parties ou il en voit beaucoup,
il transforme assez bien les probabilites et les infos accumulees.

`Controle en fin de manche` est bon: `57.37%` puis `55.37%`. Le modele semble
mieux comprendre la valeur des cartes de controle quand la pioche est basse et
que les consequences d'un Prince/Roi/Chancelier sont plus lisibles.

`Comtesse volontaire` est tres haut, mais rare: `158` parties. C'est
interessant pour le bluff ou la gestion de contrainte, mais il ne faut pas le
surinterpreter avant d'avoir plus d'exemples.

### Faiblesses Qui Reviennent

`Partie avec Roi` est le plus mauvais archetype stable: `44.61%`, puis
`42.74%` sur seed independant. C'est un vrai signal. Le Roi donne de
l'information aux deux joueurs, echange une carte qu'on connait contre une carte
adverse, et peut offrir une bonne carte au mauvais moment. Le modele semble ne
pas encore assez bien evaluer ce risque global.

`Baron avec petite carte` est le deuxieme signal faible solide: `47.13%`, puis
`41.73%`. Le probleme n'est pas seulement "jouer Baron"; c'est jouer ou subir
des etats de Baron avec une carte gardee trop faible. C'est exactement le genre
de situation ou belief + action-value doivent aider.

`Grosse carte tot` et `Princesse tot` sont faibles mais pas catastrophiques. Le
modele survit correctement, mais ces mains semblent le rendre rigide: il doit
proteger une valeur, eviter Prince/Baron, et ne pas trop reveler son etat.

`Chancelier` est ambigu. Le rapport tactique global dit que les pioches connues
via Chancelier peuvent etre tres rentables, mais l'archetype global
`Partie avec Chancelier` est moyen/faible. Cela suggere que le modele sait
parfois exploiter le Chancelier, mais pas encore choisir regulierement les bons
plans de fond de pioche.

## 5. Cartes Par Moment De Partie

Definitions:

- `early`: pioche restante >= 11.
- `mid`: pioche restante entre 6 et 10.
- `late`: pioche restante <= 5.

Attention: les winrates late sont naturellement plus hauts, car pour avoir une
carte en late il faut deja avoir survecu. Ce n'est donc pas une preuve causale
que la carte fait gagner; c'est une indication de la valeur de l'etat atteint.

Chaque case indique: `presence dans les parties / winrate de ces parties / coups joues`.

| Carte | Early | Mid | Late |
|---|---:|---:|---:|
| Espionne | 19.78% / 44.25% / 348 | 18.72% / 52.34% / 391 | 19.15% / 62.66% / 453 |
| Garde | 52.02% / 45.84% / 1549 | 43.73% / 52.37% / 1476 | 37.10% / 61.59% / 1366 |
| Pretre | 20.75% / 44.70% / 683 | 16.70% / 52.54% / 511 | 14.92% / 56.62% / 405 |
| Baron | 21.85% / 44.51% / 382 | 23.20% / 50.22% / 497 | 19.85% / 53.65% / 474 |
| Servante | 20.42% / 47.25% / 623 | 17.60% / 54.97% / 566 | 14.27% / 64.10% / 469 |
| Prince | 21.93% / 42.87% / 499 | 19.85% / 49.75% / 477 | 16.15% / 59.91% / 426 |
| Chancelier | 20.10% / 43.66% / 411 | 21.10% / 50.12% / 415 | 17.52% / 53.92% / 424 |
| Roi | 10.85% / 44.47% / 67 | 13.78% / 52.09% / 126 | 12.95% / 53.86% / 271 |
| Comtesse | 10.12% / 55.56% / 100 | 13.45% / 63.75% / 170 | 12.15% / 70.37% / 183 |
| Princesse | 11.12% / 48.76% / 0 | 15.27% / 58.92% / 0 | 19.98% / 73.09% / 0 |

### Lecture Carte Par Carte

**Espionne**: faible/moyenne tot, forte tard. Le bonus Espionne reste une vraie
source de points, mais surtout quand le modele survit assez longtemps.

**Garde**: tres present, et beaucoup plus fort tard. C'est coherent: moins il
reste de cartes, plus les guesses deviennent informees. Le modele est bon dans
les parties riches en Gardes, mais son taux de Garde juste global reste plus
bas que Step2/Step3 hybride dans certains runs.

**Pretre**: utile comme investissement d'information. Seul, il ne gagne pas la
partie; combine au Garde, il devient une des meilleures lignes du modele.

**Baron**: c'est une zone fragile. Le winrate augmente avec le temps, mais reste
moins impressionnant que d'autres cartes late. Le sous-type `Baron avec petite
carte` confirme une faiblesse tactique.

**Servante**: bon signal de survie, surtout tard. Elle aide le modele a passer
les zones dangereuses, mais elle ne cree pas seule de victoire.

**Prince**: tres faible early (`42.87%`) puis bon late (`59.91%`). C'est logique:
tot, le Prince est bruyant et peut offrir une bonne pioche; tard, il peut
forcer une Princesse ou casser une main finale.

**Chancelier**: early/mid assez moyen. La carte devrait etre une source de
controle tres forte, donc c'est un axe clair d'amelioration. Le modele doit
mieux apprendre quand garder une carte moyenne, quand enterrer une grosse carte,
et quand preparer une pioche future.

**Roi**: le plus inquietant. Meme late, le winrate avec Roi (`53.86%`) est
inferieur aux autres cartes de controle late. Il faut auditer ses Rois: cible,
timing, carte donnee, carte recue, et information reciproque creee.

**Comtesse**: bon signal, surtout mid/late. Le modele ne semble pas panique par
la Comtesse; il gere plutot bien la contrainte.

**Princesse**: tres forte tard, fragile tot. Le modele gagne beaucoup quand il
arrive a porter Princesse jusqu'en fin de manche, mais l'avoir tot ajoute une
contrainte defensive.

## 6. Familles De Cartes Par Phase

Chaque case indique: `presence dans les parties / winrate de ces parties`.

| Famille | Early | Mid | Late |
|---|---:|---:|---:|
| Information active | 84.05% / 44.71% | 73.58% / 50.73% | 57.23% / 58.10% |
| Hypothese / ciblage | 79.93% / 44.67% | 71.67% / 50.51% | 56.43% / 58.22% |
| Pression elimination | 79.63% / 45.02% | 71.13% / 51.56% | 57.75% / 59.09% |
| Controle main/pioche | 51.40% / 44.94% | 52.72% / 52.30% | 42.93% / 57.37% |
| Tempo sur | 44.70% / 47.20% | 41.95% / 55.01% | 36.28% / 63.40% |
| Valeur passive / contrainte | 37.23% / 47.62% | 40.07% / 57.02% | 39.47% / 66.81% |
| Risque fort | 61.30% / 46.04% | 62.90% / 52.98% | 52.92% / 60.27% |
| Revelation publique | 79.63% / 45.02% | 71.13% / 51.56% | 57.75% / 59.09% |

Lecture importante:

- Les cartes de valeur passive deviennent excellentes late, surtout
  Princesse/Comtesse.
- Les cartes d'hypothese deviennent meilleures late, ce qui est normal: les
  distributions adverses sont plus contraintes.
- Les cartes de controle early restent moyennes. C'est probablement une zone
  ou l'oracle rollout ou le belief peut encore beaucoup aider.

## 7. Diagnostic Sincere

Le modele actuel est meilleur que l'impression qu'on avait avant la correction
du benchmark. Il n'est pas juste "un clone fragile de l'heuristique"; il montre
des lignes tactiques propres:

- il convertit bien Pretre en Garde;
- il exploite correctement les Gardes quand il en voit beaucoup;
- il valorise mieux les fins de manche;
- il gere plutot bien Comtesse/Princesse quand il survit jusque tard;
- il bat nettement l'heuristique fair au composite post-fix.

Mais il n'est pas encore un bot humainement terrifiant. Ses faiblesses sont
assez nettes:

- Roi: mauvaise carte/archetype le plus stablement faible;
- Baron faible: mauvais risk management quand la carte gardee est petite;
- Chancelier early/mid: controle de pioche pas assez strategique;
- grosses cartes tot: manque probable de plan defensif ou de camouflage;
- Prince early: usage encore trop bruyant.

## 8. Suite Logique

Avant d'entrainer, il faut transformer ces constats en audits decisionnels:

1. **Audit Roi**: pour chaque Roi joue, noter carte donnee, carte recue, cible,
   phase, known/belief, et resultat final.
2. **Audit Baron faible**: isoler les Barons avec carte gardee <= 4 et verifier
   si le belief signalait deja un danger.
3. **Audit Chancelier**: comparer carte gardee, cartes enterrees, position
   future de pioche et resultat.
4. **Audit grosse carte early**: regarder comment le modele protege Princesse,
   Comtesse, Roi et Chancelier en debut de manche.

Si ces audits confirment les memes faiblesses, la prochaine amelioration doit
etre ciblee: collecter des etats Roi/Baron/Chancelier/Princess-early,
les relabeliser par oracle rollout CRN, puis faire une distillation
trust-region pour eviter de casser les bonnes lignes deja acquises.

