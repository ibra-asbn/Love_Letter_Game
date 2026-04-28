# Step5 - Execution Teacher Initial

Date: 2026-04-25 17:58:46 CEST.

Objectif: mesurer le regret oracle des executions Roi/Baron/Pretre/Chancelier dans les etats naturels du Step3 rapide.

## Parametres

- Games par composition: `30`
- Max states par type: `3`
- Rollouts CRN par action: `2`
- Continuation rollout: `heuristic`
- Seuil regret clair: win_delta >= `0.1`, score_delta >= `0.05`, t >= `0.75`

## Synthese

| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baron avec carte faible - choix de cible | 2 | 1 (50.00%) | 0 (0.00%) | 0 (0.00%) | 2 (100.00%) | 0.0100 | 0.0000 |
| Baron - choix de cible | 3 | 1 (33.33%) | 0 (0.00%) | 0 (0.00%) | 1 (33.33%) | 0.0000 | 0.0000 |
| Chancelier - choix de carte/ordre | 3 | 3 (100.00%) | 2 (66.67%) | 0 (0.00%) | 0 (0.00%) | 0.5517 | 0.5000 |
| Roi - choix de cible | 3 | 0 (0.00%) | 0 (0.00%) | 2 (66.67%) | 0 (0.00%) | 0.0000 | 0.0000 |
| Pretre - choix de cible | 3 | 0 (0.00%) | 0 (0.00%) | 3 (100.00%) | 0 (0.00%) | 0.0000 | 0.0000 |

## Lecture

- `Best != modele` signifie que l'oracle rollout prefere une autre execution en moyenne.
- `Regret clair` applique les seuils statistiques stricts ci-dessus.
- `Execution forcee` signifie qu'il n'y avait qu'une seule execution legale de cette carte.
- `Carte forcee` signifie qu'au moment ou la carte a ete jouee, le modele n'avait pas d'autre carte/action principale legale.

## Prochaine Etape

Les types avec assez de `regret clair` deviennent candidats pour un dataset d'entrainement. Les autres doivent rester en audit ou etre collectes avec plus de rollouts/etats.

## Fichiers

- Dataset JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/datasets/smoke_execution_teacher.json`
- Rapport JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/reports/smoke_execution_teacher_report.json`
- Log: `step5_execution_heads/logs/2026-04-25_smoke_execution_teacher.md`
