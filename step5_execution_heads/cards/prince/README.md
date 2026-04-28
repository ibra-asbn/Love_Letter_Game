# Step5 Prince

Date: 2026-04-26.

Le Prince est traite comme une carte de decision locale: quand Step3 a un
Prince en main, on veut verifier s'il sait bien choisir entre:

- recycler sa propre carte quand l'autre carte est faible;
- viser un adversaire susceptible d'avoir la Princesse, la Comtesse ou une
  grosse carte;
- eviter les Princes sans valeur quand l'autre carte en main est deja forte.

## Evaluation En Cours

Pour cette premiere passe, on ne compare que:

- `Step3 rapide`;
- `Step3 + Prince V1`.

La lecture principale est conditionnelle: on regarde les parties ou le joueur
evalue a eu un Prince en main au moins une fois. Cela evite de diluer le signal
Prince dans des parties ou la carte n'apparait jamais.

## Critere De Succes

Prince V1 est un succes local si, sur les parties avec Prince en main:

- le winrate conditionnel augmente nettement contre Step3 seul;
- le gain ne vient pas d'une degradation globale evidente;
- les stats tactiques sont coherentes: plus de bons Princes sur Princesse
  adverse, moins de Princes sur soi avec une carte forte.
