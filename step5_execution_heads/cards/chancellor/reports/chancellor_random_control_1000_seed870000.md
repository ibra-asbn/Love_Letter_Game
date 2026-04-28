# Step4 - Ablations Conditionnelles Par Carte

Date: 2026-04-26 13:25:37 CEST.

Objectif: comprendre ce qui vient du style du modele et ce qui vient de la dynamique du jeu.

On laisse Step3 rapide choisir naturellement quelle carte jouer. Si la carte jouee correspond a l'ablation, on randomise seulement son execution: cible, guess, ou choix Chancelier.

## Synthese Globale

| Ablation | Composite | Delta vs normal | Changed games | Changed events | Lecture |
|---|---:|---:|---:|---:|---|
| Step3 rapide normal | 0.39340 | +0.00000 | 0 | 0 | Reference normale. |
| Chancelier choix random | 0.39820 | +0.00480 | 881 | 920 | Effet global faible: verifier le regret oracle avant d'entrainer. |

## Winrates Par Composition

| Ablation | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|
| Step3 rapide normal | 54.30% | 45.90% | 38.70% | 32.80% |
| Chancelier choix random | 53.50% | 46.50% | 39.90% | 33.00% |

## Interventions

`Eligible` signifie que le modele a choisi la carte concernee. `Changed` signifie que le tirage random a effectivement remplace son action.

| Ablation | Eligible games | Changed games | Eligible events | Changed events | Changed / eligible |
|---|---:|---:|---:|---:|---:|
| Step3 rapide normal | 0 | 0 | 0 | 0 | 0.00% |
| Chancelier choix random | 1078 | 881 | 1156 | 920 | 79.58% |

## Metriques Tactiques Globales

| Ablation | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier connu gagne |
|---|---:|---:|---:|---:|---:|---:|
| Step3 rapide normal | 29.44% | 84.50% | 88.50% | 75.00% | 23.15% | 60.22% |
| Chancelier choix random | 29.74% | 84.85% | 88.21% | 75.11% | 23.00% | 54.97% |

## Comment Lire Ces Resultats

- Si une ablation degrade fort, le modele a une competence d'execution a proteger.
- Si une ablation ne change presque rien, on doit verifier le regret oracle: soit la carte est peu sensible, soit le modele ne sait pas l'exploiter.
- Si une ablation ameliore, c'est une alerte: l'execution du modele est probablement toxique sur cette carte.
- Le delta global doit toujours etre lu avec le nombre de `changed events`; une competence rare peut etre importante sans bouger beaucoup le composite.

## Fichiers

- JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step4_weakness_analysis/reports/step5_compare_chancellor_random_1000_seed870000.json`
- Log: `step5_execution_heads/logs/2026-04-26_step5_compare_chancellor_random_1000_seed870000.md`
