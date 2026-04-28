# Step5 - Execution Teacher Initial

Date: 2026-04-26 13:36:57 CEST.

Objectif: mesurer le regret oracle des executions Roi/Baron/Pretre/Chancelier dans les etats naturels du Step3 rapide.

## Parametres

- Games par composition: `2500`
- Max states par type: `160`
- Rollouts CRN par action: `12`
- Continuation rollout: `heuristic`
- Seuil regret clair: win_delta >= `0.1`, score_delta >= `0.05`, t >= `0.75`

## Synthese

| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baron avec carte faible - choix de cible | 160 | 76 (47.50%) | 41 (25.62%) | 31 (19.38%) | 143 (89.38%) | 0.0822 | 0.0750 |
| Roi - choix de cible | 160 | 41 (25.62%) | 18 (11.25%) | 82 (51.25%) | 58 (36.25%) | 0.0362 | 0.0339 |
| Pretre - choix de cible | 160 | 67 (41.88%) | 17 (10.62%) | 39 (24.38%) | 28 (17.50%) | 0.0359 | 0.0333 |

## Lecture

- `Best != modele` signifie que l'oracle rollout prefere une autre execution en moyenne.
- `Regret clair` applique les seuils statistiques stricts ci-dessus.
- `Execution forcee` signifie qu'il n'y avait qu'une seule execution legale de cette carte.
- `Carte forcee` signifie qu'au moment ou la carte a ete jouee, le modele n'avait pas d'autre carte/action principale legale.

## Prochaine Etape

Les types avec assez de `regret clair` deviennent candidats pour un dataset d'entrainement. Les autres doivent rester en audit ou etre collectes avec plus de rollouts/etats.

## Fichiers

- Dataset JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/datasets/target_teacher_baronlow_king_priest_balanced_160x12.json`
- Rapport JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/reports/target_teacher_baronlow_king_priest_balanced_160x12_report.json`
- Log: `step5_execution_heads/logs/2026-04-26_target_teacher_baronlow_king_priest_balanced_160x12.md`
