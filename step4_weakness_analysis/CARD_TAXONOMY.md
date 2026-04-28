# Step 4 - Taxonomie Des Cartes

Date: 2026-04-25.

But: figer une grille commune pour analyser les faiblesses du dernier modele
Step3. Une carte peut appartenir a plusieurs familles: Love Letter est un jeu
d'information partielle, donc une meme action peut a la fois produire de
l'information, forcer une hypothese, exposer un risque et changer la valeur
future d'une main.

## Cartes

| Valeur | Carte | Effet resume |
|---:|---|---|
| 0 | Espionne | Pas d'effet immediat; bonus si seul survivant avec Espionne jouee. |
| 1 | Garde | Devine une carte adverse sauf Garde; elimine si la devinette est juste. |
| 2 | Pretre | Regarde la carte d'un adversaire, information privee. |
| 3 | Baron | Compare la carte gardee avec celle d'un adversaire; la plus faible sort. |
| 4 | Servante | Immunite jusqu'au prochain tour. |
| 5 | Prince | Force une defausse/repioche sur soi ou un adversaire. |
| 6 | Chancelier | Pioche jusqu'a deux cartes, en garde une, remet les autres au fond. |
| 7 | Roi | Echange sa carte avec celle d'un adversaire. |
| 8 | Comtesse | Doit etre jouee avec Prince ou Roi; peut aussi etre jouee en bluff. |
| 9 | Princesse | Elimine son proprietaire si elle est jouee ou defaussee. |

## Familles Fixees Pour Step4

| Famille | Cartes | Pourquoi c'est utile pour l'analyse |
|---|---|---|
| `information_active` | Pretre, Chancelier, Roi, Baron, Garde | Actions qui creent ou exploitent de l'information privee ou publique. Le Garde rate donne aussi une information negative. |
| `hypothesis_pressure` | Garde, Baron, Prince, Roi | Actions dont la qualite depend fortement d'une hypothese sur la main adverse. |
| `elimination_pressure` | Garde, Baron, Prince, Princesse | Actions qui peuvent eliminer directement un joueur, soi compris. |
| `hand_deck_control` | Prince, Chancelier, Roi, Comtesse | Actions qui modifient la main, la pioche, ou les contraintes futures. |
| `safe_tempo` | Servante, Espionne, Comtesse | Cartes qui peuvent temporiser ou reduire le risque immediat sans forcement tuer. |
| `passive_value_or_constraint` | Espionne, Comtesse, Princesse | Cartes dont la valeur vient surtout de leur presence, de leur contrainte, ou du score final. |
| `high_risk_trap` | Baron, Prince, Roi, Comtesse, Princesse | Cartes qui punissent tres fort une mauvaise lecture du contexte. |
| `public_reveal` | Garde, Baron, Prince, Princesse | Actions qui peuvent rendre une carte visible a toute la table via elimination ou defausse. |

## Phases De Partie

Les analyses Step4 utilisent une definition simple et stable, calculee au
moment ou le modele doit agir, donc apres sa pioche normale:

| Phase | Taille de pioche restante | Lecture |
|---|---:|---|
| `early` | 11 cartes ou plus | Debut de manche, information encore tres diffuse. |
| `mid` | 6 a 10 cartes | Milieu, assez d'historique pour commencer a exploiter. |
| `late` | 5 cartes ou moins | Fin de manche, chaque action a un impact plus lisible. |

Cette separation n'est pas une regle strategique; c'est une grille d'audit.
Elle sert a detecter les situations ou le modele gagne/perd selon le type de
main qu'il rencontre au debut, au milieu ou en fin de manche.

