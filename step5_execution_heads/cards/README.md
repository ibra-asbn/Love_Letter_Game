# Step5 - Organisation Par Carte

Date: 2026-04-26.

Step5 est organise par carte parce que chaque correctif a un role local. Selon
la carte, il peut corriger uniquement **comment executer** une carte deja choisie
ou, si c'est indispensable, comparer localement cette carte avec l'autre carte en
main.

## Cartes Actives

| Carte | Dossier | Statut | Objectif local |
|---|---|---|---|
| Chancelier | `chancellor/` | V1 validee | Choisir carte gardee + ordre de remise |
| Baron | `baron/` | V1 validee | Comparer Baron vs autre carte, puis choisir une cible sure |
| Prince | `prince/` | V1 positive legere | Choisir cible/recyclage sans diluer le signal hors Prince |
| Roi | `king/` | En cours | Choisir une cible d'echange non toxique |
| Pretre | `priest/` | V1 candidate non validee | Choisir la cible dont l'information a de la valeur |

## Regle De Validation

Chaque carte doit avoir:

- un dataset teacher/audit CRN;
- un checkpoint de tete locale ou une regle action-value locale si le signal est exploitable;
- une evaluation fair seat-rotated contre Step3 rapide;
- un controle random equivalent;
- une conclusion claire: succes, echec, ou signal a retravailler.

Les scripts restent au niveau `step5_execution_heads/` pour eviter les copies de
code. Les artefacts importants sont ranges dans les sous-dossiers par carte.
