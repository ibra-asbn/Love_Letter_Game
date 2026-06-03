# Step5 - Execution Teacher Initial

Date: 2026-04-26 13:39:42 CEST.

Objectif: mesurer le regret oracle des executions Roi/Baron/Pretre/Chancelier dans les etats naturels du Step3 rapide.

## Parametres

- Games par composition: `1800`
- Max states par type: `240`
- Rollouts CRN par action: `12`
- Continuation rollout: `heuristic`
- Seuil regret clair: win_delta >= `0.1`, score_delta >= `0.05`, t >= `0.75`

## Synthese

| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baron - choix de cible | 240 | 100 (41.67%) | 55 (22.92%) | 67 (27.92%) | 45 (18.75%) | 0.0738 | 0.0688 |

## Lecture

- `Best != modele` signifie que l'oracle rollout prefere une autre execution en moyenne.
- `Regret clair` applique les seuils statistiques stricts ci-dessus.
- `Execution forcee` signifie qu'il n'y avait qu'une seule execution legale de cette carte.
- `Carte forcee` signifie qu'au moment ou la carte a ete jouee, le modele n'avait pas d'autre carte/action principale legale.

## Prochaine Etape

Les types avec assez de `regret clair` deviennent candidats pour un dataset d'entrainement. Les autres doivent rester en audit ou etre collectes avec plus de rollouts/etats.

## Fichiers

- Dataset JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/cards/baron/datasets/baron_target_teacher_240x12.json`
- Rapport JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/cards/baron/reports/baron_target_teacher_240x12_report.json`
- Log: `step5_execution_heads/cards/baron/logs/2026-04-26_baron_target_teacher_240x12.md`
