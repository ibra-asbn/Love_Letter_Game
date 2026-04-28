# Regles De Love Letter

Source: texte de reference fourni dans la conversation du 25 avril 2026.

Ce fichier sert de reference locale pour verifier le moteur `LoveLetterRLEnv`.
Il decrit les regles utiles a la version du projet avec les cartes `0` a `9`.

## Mise En Place

1. Melanger le paquet.
2. Donner une carte secrete a chaque joueur.
3. Mettre une carte de cote face cachee.
4. Les cartes restantes constituent la pioche face cachee.
5. A son tour, un joueur pioche une carte, a donc deux cartes en main, puis en
   joue une face visible.
6. Si un joueur est elimine pendant une manche, il ne rejoue plus avant la
   manche suivante.
7. La manche se termine quand il ne reste plus qu'un joueur vivant ou quand il
   n'y a plus de cartes a piocher.

## Effets Des Cartes

### 0 - Espionne

L'effet immediat ne fait rien. A la fin de la manche, si un joueur vivant est
le seul a avoir joue le plus d'Espionnes devant lui, il marque un point bonus.
L'Espionne doit etre jouee devant soi, pas seulement gardee en main.

### 1 - Garde

Le joueur designe un adversaire et devine sa carte. Il peut citer n'importe
quelle carte sauf Garde. Si la devinette est correcte, la cible est eliminee.
Si la devinette est fausse, la cible dit seulement que ce n'est pas cette carte
et ne revele pas sa main.

### 2 - Pretre

Le joueur regarde secretement la carte d'un adversaire. Il ne peut pas reveler
officiellement cette information aux autres joueurs.

### 3 - Baron

Le joueur compare sa carte gardee avec la carte d'un adversaire.

- Si la carte adverse est plus petite, l'adversaire est elimine.
- Si la carte adverse est plus haute, le joueur qui a joue Baron est elimine.
- En cas d'egalite, rien ne se passe.

La carte du joueur elimine est revelee parce qu'elle est defaussee devant lui.
La carte du survivant n'est pas revelee publiquement. Les observateurs peuvent
seulement inferer que la carte du survivant etait strictement superieure a la
carte du mort.

### 4 - Servante

Jusqu'au prochain tour du joueur, personne ne peut le cibler avec Garde, Pretre,
Baron, Prince ou Roi.

### 5 - Prince

Le joueur choisit un joueur, y compris lui-meme. La cible defausse sa carte
face visible, sans appliquer l'effet de cette carte, puis pioche une nouvelle
carte. Si la carte defaussee est la Princesse, la cible est eliminee.

Si la pioche est vide, la cible prend la carte mise de cote au debut.

Si tous les adversaires sont proteges par Servante et que le Prince doit etre
joue, le joueur doit se choisir lui-meme.

### 6 - Chancelier

Le joueur pioche jusqu'a deux cartes, choisit une carte a garder, puis remet les
autres cartes au fond de la pioche dans l'ordre de son choix.

S'il ne reste qu'une carte dans la pioche, il pioche seulement cette carte et
remet une des deux cartes au fond. Une Princesse remise au fond par Chancelier
n'est ni jouee ni defaussee: elle n'elimine pas le joueur.

### 7 - Roi

Le joueur doit echanger sa carte gardee avec la carte gardee d'un adversaire.
Les deux joueurs impliques connaissent alors les cartes echangees. Les autres
joueurs savent seulement qu'un echange a eu lieu, sauf s'ils avaient deja une
information permettant une deduction.

### 8 - Comtesse

Si un joueur a Comtesse avec Prince ou Roi, il est oblige de jouer la Comtesse.
Il peut aussi jouer la Comtesse volontairement avec une autre carte: c'est un
bluff possible.

### 9 - Princesse

Si un joueur joue ou defausse la Princesse, il est elimine. En revanche, la
Princesse peut etre remise dans la pioche par Chancelier sans elimination.

## Fin De Manche

Une manche peut etre gagnee de deux manieres:

1. Tous les autres joueurs sont elimines.
2. La pioche est vide: les joueurs vivants comparent leur carte en main, et la
   carte la plus haute gagne. En cas d'egalite, les joueurs a egalite marquent.

Le bonus d'Espionne est ensuite attribue au joueur vivant qui a seul le plus
d'Espionnes jouees devant lui. En cas d'egalite pour l'Espionne, personne ne
marque ce bonus.

Un joueur peut donc marquer jusqu'a deux points sur une manche: un point de
victoire de manche et un point bonus d'Espionne.

## Fin De Partie Complete

Seuils officiels de victoire:

| Nombre de joueurs | Points requis |
|---:|---:|
| 2 | 6 |
| 3 | 5 |
| 4 | 4 |
| 5-6 | 3 |

Le moteur RL actuel modelise surtout des manches independantes. Les tokens sont
exposes dans l'observation, mais la boucle complete multi-manches n'est pas le
coeur des evaluations actuelles.
