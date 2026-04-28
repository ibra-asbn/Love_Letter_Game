# Step5 - Roi

Date: 2026-04-26.

## Objectif

Entrainer une tete locale pour choisir la cible du Roi. Le but n'est pas
simplement d'echanger contre une carte plus haute: l'echange revele aussi une
information privee aux deux joueurs et peut devenir toxique si l'adversaire
exploite ensuite la carte recue.

## Critere De Succes

- Ameliorer Step3 rapide au composite fair seat-rotated.
- Faire mieux que le controle `king_target_random`.
- Eviter les overrides massifs sur les situations incertaines.

## Statut

En cours.

## Benchmark Diagnostic

Script:

```text
step5_execution_heads/cards/king/analyze_king_usage.py
```

Mesure:

- winrate quand Roi est en main;
- winrate quand Roi est joue;
- taux d'utilisation early/mid/late;
- qualite brute de l'echange: carte recue plus haute, plus basse ou egale;
- breakdown par carte accompagnante.

## Resultats Du 2026-04-26

Deux lectures ont ete lancees, 1000 parties par composition d'arene avec siege
tournant:

- Step3 rapide seul:
  `reports/king_usage_step3_1000_seed3200000.md`
- Step3 + tetes Chancelier + Baron + Prince:
  `reports/king_usage_step5_cbp_1000_seed3300000.md`

Lecture principale:

- Le Roi n'est pas maitrise comme Garde ou Prince, mais il n'est pas totalement
  aleatoire non plus.
- Sur Step3 seul, jouer Roi est faible: `39.76%` de winrate conditionnel.
- Avec les trois tetes deja branchees, jouer Roi remonte a `44.84%`.
- Le Roi est peu joue early/mid (`~18%` des occurrences), puis beaucoup plus
  late (`~38%`), ce qui est coherent.
- La cible est connue avant echange dans seulement `~2-3%` des Roi joues:
  le modele joue donc surtout le Roi a l'aveugle.
- Le point fragile le plus clair est Roi + Princesse: le modele joue Roi dans
  `100%` de ces cas, donne souvent la Princesse, et reste a seulement `~41%`
  de winrate conditionnel.
- Roi + Chancelier etait catastrophique dans Step3 seul (`32.26%` si Roi joue)
  mais devient beaucoup plus sain avec les tetes actuelles (`48.68%`).

Conclusion: le Roi doit probablement etre traite plus tard comme une tete
d'echange prudente, avec une attention speciale aux situations Roi+Princesse,
Roi+Garde et Roi late. Ce n'est pas la prochaine tete la plus simple: la valeur
de l'echange depend autant de l'information offerte que de la valeur brute de
la carte recue.
