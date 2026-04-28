# Audit Regles Love Letter

Date: 2026-04-25.

Reference locale:

```text
docs/love_letter_rules_fr.md
```

## Corrections Faites

### Baron

Avant correction, quand un Baron gagnait, le moteur exposait dans
`_min_kept_card` la vraie carte du survivant. C'etait trop fort:

```text
Baron avec 6 contre 2 -> min_kept_card = 6
```

Regle correcte:

```text
les observateurs voient la carte morte 2 et savent seulement que le survivant
avait une carte strictement superieure, donc min_kept_card = 3.
```

Correction appliquee:

- Baron gagne: borne publique `loser_card + 1`;
- Baron perd: borne publique `loser_card + 1` pour le survivant;
- Baron egalite: pas de borne publique, mais les deux joueurs impliques
  connaissent maintenant exactement la carte de l'autre via `_known_cards`.

### Roi

Le Roi donnait deja une information privee bilaterale correcte aux deux joueurs
impliques:

- le joueur qui joue Roi connait la nouvelle carte de la cible;
- la cible connait la nouvelle carte du joueur.

Correction ajoutee:

- si un observateur connaissait deja une carte avant le Roi, cette connaissance
  suit maintenant la carte apres l'echange;
- les bornes publiques `_min_kept_card` liees aux cartes echangees suivent aussi
  l'echange quand elles n'ont pas ete invalidees par le coup joue.

### Servante

Le masque d'action interdisait deja de cibler un joueur protege. Le moteur est
maintenant aussi protege contre une action invalide injectee directement:

```text
si une cible adverse est protegee, l'effet cible est ignore.
```

## Tests Ajoutes

Script:

```bash
python3 scripts/debug/check_rules_conformance.py
```

Cas couverts:

- Garde faux: ne revele pas la carte;
- Pretre: information exacte privee;
- Baron gagne: carte morte publique, borne publique non exacte;
- Baron perd: carte morte publique, borne publique non exacte;
- Baron egalite: information exacte privee entre les deux joueurs;
- Servante: empeche le ciblage meme si une action invalide est forcee;
- Prince: peut cibler soi-meme quand tous les adversaires sont proteges;
- Chancelier: peut remettre Princesse dans la pioche sans elimination;
- Roi: information bilaterale et transfert des connaissances deja acquises;
- Comtesse: forcee avec Prince ou Roi;
- Espionne: bonus seulement pour l'unique joueur vivant avec le plus
  d'Espionnes jouees.

Resultat du 25 avril 2026:

```text
11 tests OK.
```

## Points Conformes Verifies Par Lecture

- Mise en place: une carte par joueur, une carte face cachee mise de cote,
  premier joueur qui pioche avant de jouer.
- Cartes jouees visibles via `_played_cards`.
- Garde ne peut pas deviner Garde dans le masque legal.
- Prince defausse sans appliquer l'effet de la carte defaussee.
- Prince sur Princesse elimine la cible.
- Prince avec pioche vide utilise la carte mise de cote.
- Chancelier pioche jusqu'a deux cartes, garde une carte, remet les autres au
  fond dans l'ordre choisi.
- Princesse jouee elimine le joueur.
- Fin de manche: dernier vivant ou plus haute carte en main si pioche vide.
- Egalite a la plus haute carte: plusieurs gagnants possibles.
- Bonus Espionne: seulement si un joueur vivant est unique meilleur en
  Espionnes jouees.

## Limites Volontaires Du Moteur RL

Ces points ne sont pas des bugs bloquants pour les manches RL actuelles, mais
ils ne representent pas toute la partie de plateau:

- le moteur evalue surtout des manches independantes, pas une partie complete
  jusqu'a 4 points a 4 joueurs;
- le premier joueur est controle par seed/options ou tire aleatoirement, au lieu
  d'etre le gagnant de la manche precedente;
- les rewards incluent du shaping (`+0.2` pour certaines eliminations, `-0.2`
  pour certains risques) en plus du score officiel de manche;
- la communication verbale, le bluff oral et le mensonge hors effets de cartes
  ne sont pas modelises;
- l'observation ne contient pas encore tout l'historique public detaille des
  actions, par exemple les cibles et devinettes exactes de tous les anciens
  Gardes;
- le signal humain "a-t-il joue la carte qu'il vient de piocher ?" n'est pas
  encore modelise.

## Consequence Pour Les Benchmarks

Les benchmarks produits avant cette correction ont ete faits avec une borne
Baron trop informative. Les conclusions globales restent utiles pour guider le
projet, mais les prochaines evaluations Step4 doivent etre relancees avec ce
moteur corrige.
