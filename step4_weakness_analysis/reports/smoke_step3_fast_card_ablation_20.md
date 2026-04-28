# Step4 - Ablations Conditionnelles Par Carte

Date: 2026-04-25 17:41:02 CEST.

Objectif: comprendre ce qui vient du style du modele et ce qui vient de la dynamique du jeu.

On laisse Step3 rapide choisir naturellement quelle carte jouer. Si la carte jouee correspond a l'ablation, on randomise seulement son execution: cible, guess, ou choix Chancelier.

## Synthese Globale

| Ablation | Composite | Delta vs normal | Changed games | Changed events | Lecture |
|---|---:|---:|---:|---:|---|
| Step3 rapide normal | 0.34000 | +0.00000 | 0 | 0 | Reference normale. |
| Garde cible+guess random | 0.31500 | -0.02500 | 55 | 85 | Competence d'execution probable: le hasard degrade nettement. |
| Chancelier choix random | 0.26500 | -0.07500 | 22 | 23 | Signal rare: ne pas conclure sans plus de parties. |
| Roi cible random | 0.34000 | +0.00000 | 1 | 1 | Signal rare: ne pas conclure sans plus de parties. |

## Winrates Par Composition

| Ablation | vs 3R | vs 1H+2R | vs 2H+1R | vs 3H |
|---|---:|---:|---:|---:|
| Step3 rapide normal | 55.00% | 35.00% | 45.00% | 20.00% |
| Garde cible+guess random | 60.00% | 35.00% | 35.00% | 20.00% |
| Chancelier choix random | 55.00% | 20.00% | 30.00% | 20.00% |
| Roi cible random | 55.00% | 35.00% | 45.00% | 20.00% |

## Interventions

`Eligible` signifie que le modele a choisi la carte concernee. `Changed` signifie que le tirage random a effectivement remplace son action.

| Ablation | Eligible games | Changed games | Eligible events | Changed events | Changed / eligible |
|---|---:|---:|---:|---:|---:|
| Step3 rapide normal | 0 | 0 | 0 | 0 | 0.00% |
| Garde cible+guess random | 57 | 55 | 95 | 85 | 89.47% |
| Chancelier choix random | 24 | 22 | 25 | 23 | 92.00% |
| Roi cible random | 11 | 1 | 11 | 1 | 9.09% |

## Metriques Tactiques Globales

| Ablation | Garde juste | Garde connu juste | Pretre->Garde juste | Baron gagne | Baron perdu | Chancelier connu gagne |
|---|---:|---:|---:|---:|---:|---:|
| Step3 rapide normal | 23.66% | 100.00% | 100.00% | 64.71% | 32.35% | 83.33% |
| Garde cible+guess random | 11.83% | 12.50% | 0.00% | 69.44% | 27.78% | 100.00% |
| Chancelier choix random | 24.73% | 100.00% | 100.00% | 67.65% | 29.41% | 33.33% |
| Roi cible random | 22.83% | 100.00% | 100.00% | 64.71% | 32.35% | 83.33% |

## Comment Lire Ces Resultats

- Si une ablation degrade fort, le modele a une competence d'execution a proteger.
- Si une ablation ne change presque rien, on doit verifier le regret oracle: soit la carte est peu sensible, soit le modele ne sait pas l'exploiter.
- Si une ablation ameliore, c'est une alerte: l'execution du modele est probablement toxique sur cette carte.
- Le delta global doit toujours etre lu avec le nombre de `changed events`; une competence rare peut etre importante sans bouger beaucoup le composite.

## Fichiers

- JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step4_weakness_analysis/reports/smoke_step3_fast_card_ablation_20.json`
- Log: `step4_weakness_analysis/logs/2026-04-25_smoke_step3_fast_card_ablation_20.md`
