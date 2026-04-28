# Step4 - Ablations Conditionnelles Par Carte

Date: 2026-04-25 17:46:05 CEST.

Objectif: comprendre ce qui vient du style du modele et ce qui vient de la dynamique du jeu.

On laisse Step3 rapide choisir naturellement quelle carte jouer. Si la carte jouee correspond a l'ablation, on randomise seulement son execution: cible, guess, ou choix Chancelier.

## Synthese Globale

| Ablation | Composite | Delta vs normal | Changed games | Changed events | Lecture |
|---|---:|---:|---:|---:|---|
| Step3 rapide normal | 0.39750 | +0.00000 | 0 | 0 | Reference normale. |
| Garde cible random, guess conserve | 0.38330 | -0.01420 | 1529 | 1808 | Competence moderee ou effet rare. |
| Garde guess random, cible conservee | 0.33320 | -0.06430 | 2571 | 3893 | Competence d'execution probable: le hasard degrade nettement. |
| Garde cible+guess random | 0.33220 | -0.06530 | 2685 | 4144 | Competence d'execution probable: le hasard degrade nettement. |
| Pretre cible random | 0.39900 | +0.00150 | 651 | 678 | Effet global faible: verifier le regret oracle avant d'entrainer. |
| Baron cible random | 0.39100 | -0.00650 | 528 | 533 | Competence moderee ou effet rare. |
| Prince cible random | 0.37470 | -0.02280 | 814 | 853 | Competence d'execution probable: le hasard degrade nettement. |
| Roi cible random | 0.39750 | +0.00000 | 121 | 121 | Effet global faible: verifier le regret oracle avant d'entrainer. |
| Chancelier choix random | 0.39550 | -0.00200 | 919 | 979 | Effet global faible: verifier le regret oracle avant d'entrainer. |

## Winrates Par Composition

| Ablation | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|
| Step3 rapide normal | 51.90% | 46.20% | 38.40% | 34.50% |
| Garde cible random, guess conserve | 51.70% | 44.90% | 38.20% | 31.80% |
| Garde guess random, cible conservee | 47.90% | 40.90% | 32.50% | 26.50% |
| Garde cible+guess random | 48.80% | 40.20% | 32.60% | 26.30% |
| Pretre cible random | 53.10% | 46.20% | 37.70% | 35.10% |
| Baron cible random | 52.30% | 44.40% | 37.30% | 34.50% |
| Prince cible random | 49.40% | 42.40% | 37.10% | 32.30% |
| Roi cible random | 52.00% | 46.10% | 38.70% | 34.30% |
| Chancelier choix random | 51.90% | 45.40% | 38.00% | 34.70% |

## Interventions

`Eligible` signifie que le modele a choisi la carte concernee. `Changed` signifie que le tirage random a effectivement remplace son action.

| Ablation | Eligible games | Changed games | Eligible events | Changed events | Changed / eligible |
|---|---:|---:|---:|---:|---:|
| Step3 rapide normal | 0 | 0 | 0 | 0 | 0.00% |
| Garde cible random, guess conserve | 2799 | 1529 | 4378 | 1808 | 41.30% |
| Garde guess random, cible conservee | 2799 | 2571 | 4495 | 3893 | 86.61% |
| Garde cible+guess random | 2799 | 2685 | 4515 | 4144 | 91.78% |
| Pretre cible random | 1441 | 651 | 1598 | 678 | 42.43% |
| Baron cible random | 1269 | 528 | 1357 | 533 | 39.28% |
| Prince cible random | 1249 | 814 | 1385 | 853 | 61.59% |
| Roi cible random | 464 | 121 | 464 | 121 | 26.08% |
| Chancelier choix random | 1142 | 919 | 1252 | 979 | 78.19% |

## Metriques Tactiques Globales

| Ablation | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier connu gagne |
|---|---:|---:|---:|---:|---:|---:|
| Step3 rapide normal | 30.50% | 87.02% | 92.12% | 73.56% | 24.12% | 65.03% |
| Garde cible random, guess conserve | 25.19% | 81.52% | 91.97% | 73.15% | 24.28% | 62.73% |
| Garde guess random, cible conservee | 10.39% | 12.76% | 94.74% | 73.10% | 23.92% | 52.60% |
| Garde cible+guess random | 9.70% | 7.88% | 90.00% | 73.34% | 24.05% | 53.61% |
| Pretre cible random | 30.39% | 86.90% | 91.63% | 73.74% | 23.70% | 64.88% |
| Baron cible random | 30.44% | 87.21% | 92.12% | 71.04% | 26.33% | 68.12% |
| Prince cible random | 30.53% | 86.68% | 90.05% | 73.60% | 24.13% | 62.35% |
| Roi cible random | 30.52% | 87.31% | 92.68% | 73.33% | 24.27% | 64.42% |
| Chancelier choix random | 29.83% | 85.80% | 92.72% | 73.54% | 24.08% | 58.60% |

## Comment Lire Ces Resultats

- Si une ablation degrade fort, le modele a une competence d'execution a proteger.
- Si une ablation ne change presque rien, on doit verifier le regret oracle: soit la carte est peu sensible, soit le modele ne sait pas l'exploiter.
- Si une ablation ameliore, c'est une alerte: l'execution du modele est probablement toxique sur cette carte.
- Le delta global doit toujours etre lu avec le nombre de `changed events`; une competence rare peut etre importante sans bouger beaucoup le composite.

## Fichiers

- JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step4_weakness_analysis/reports/step3_fast_card_ablation_1000.json`
- Log: `step4_weakness_analysis/logs/2026-04-25_step3_fast_card_ablation_1000.md`
