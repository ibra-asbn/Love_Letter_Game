# Step5 Baron V2 - Plan Action-Value

Date: 2026-04-26.

## Constat De Depart

L'audit `baron_companion_audit_step3_5000_seed1800000.md` montre que le
Step3 rapide ne joue pas le Baron au hasard.

Il evite quasiment toujours Baron avec les petites cartes:

- Baron + Espionne: Baron joue dans 0.36% des cas.
- Baron + Garde: 0.11%.
- Baron + Pretre: 2.36%.
- Baron + Servante: 1.53%.

Il joue presque toujours Baron avec les grosses cartes:

- Baron + Prince: 94.41%.
- Baron + Chancelier: 95.92%.
- Baron + Roi: 99.33%.
- Baron + Comtesse: 100%.
- Baron + Princesse: 100%.

Le probleme n'est donc pas "le modele ne sait pas quand Baron est une carte
forte". Le probleme est plus precis: avec les cartes moyennes-fortes, surtout
Prince et Chancelier, il joue souvent Baron mais perd encore trop de duels.

Exemples globaux:

- Baron + Prince: 55.01% duels gagnes, 39.83% duels perdus.
- Baron + Chancelier: 65.87% duels gagnes, 29.12% duels perdus.
- Baron + Roi: 78.27% duels gagnes, 17.94% duels perdus.
- Baron + Comtesse: 84.48% duels gagnes, 11.88% duels perdus.
- Baron + Princesse: 97.79% duels gagnes, 0.00% duels perdus.

## Pourquoi La Tete De Cible A Peu Marche

La premiere tentative ne modifiait que la cible apres que Step3 avait deja
choisi Baron. C'est trop tardif pour les mains `Baron + Prince` et
`Baron + Chancelier`.

Dans ces etats, la vraie question est:

> Dois-je jouer Baron maintenant, ou jouer l'autre carte?

Puis seulement si Baron est meilleur:

> Quelle cible choisir?

## Nouvelle Strategie

On remplace la logique "target-only" par un specialiste local:

1. Il ne s'active que si Baron est dans la main.
2. Il estime la valeur de jouer Baron avec la meilleure cible disponible.
3. Il compare cette valeur a l'action non-Baron naturelle.
4. Il refuse les overrides si l'information est trop faible.

Cette version V2 commence par une regle action-value explicable, afin de valider
la forme du probleme avant de distiller dans une tete neuronale:

- Petite carte gardee: ne jouer Baron que si l'information est quasi certaine.
- Carte moyenne (`Prince`, `Chancelier`): jouer Baron seulement si le risque de
  mourir est suffisamment bas.
- Grosse carte (`Roi`, `Comtesse`, `Princesse`): garder l'agressivite, mais
  corriger la cible avec une estimation EV.

## Critere De Succes

Le succes ne sera pas juge uniquement au composite global, car Baron est une
carte rare. On demande:

- ameliorer Step3 sur les parties ou Baron apparait en main;
- reduire le taux de duel perdu sur `Baron + Prince` et `Baron + Chancelier`;
- ne pas degrader `Baron + Roi`, `Baron + Comtesse`, `Baron + Princesse`;
- rester au-dessus du controle `Baron target random`;
- ne pas modifier les autres decisions hors etats avec Baron en main.

La version sera consideree valide seulement si elle confirme un gain sur au
moins deux blocs de seeds ou un bloc large statistiquement lisible.
