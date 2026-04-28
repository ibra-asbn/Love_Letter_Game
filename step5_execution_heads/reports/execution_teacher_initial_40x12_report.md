# Step5 - Execution Teacher Initial

Date: 2026-04-25 17:59:18 CEST.

Objectif: mesurer le regret oracle des executions Roi/Baron/Pretre/Chancelier dans les etats naturels du Step3 rapide.

## Parametres

- Games par composition: `500`
- Max states par type: `40`
- Rollouts CRN par action: `12`
- Continuation rollout: `heuristic`
- Seuil regret clair: win_delta >= `0.1`, score_delta >= `0.05`, t >= `0.75`

## Synthese

| Type | Etats | Best != modele | Regret clair | Execution forcee | Carte forcee | Mean score regret | Mean win regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baron avec carte faible - choix de cible | 40 | 26 (65.00%) | 8 (20.00%) | 6 (15.00%) | 35 (87.50%) | 0.0715 | 0.0667 |
| Baron - choix de cible | 40 | 13 (32.50%) | 6 (15.00%) | 13 (32.50%) | 11 (27.50%) | 0.0562 | 0.0521 |
| Chancelier - choix de carte/ordre | 40 | 27 (67.50%) | 15 (37.50%) | 1 (2.50%) | 0 (0.00%) | 0.1199 | 0.1104 |
| Roi - choix de cible | 40 | 12 (30.00%) | 7 (17.50%) | 19 (47.50%) | 15 (37.50%) | 0.0615 | 0.0583 |
| Pretre - choix de cible | 40 | 16 (40.00%) | 5 (12.50%) | 11 (27.50%) | 6 (15.00%) | 0.0382 | 0.0354 |

## Lecture

- `Best != modele` signifie que l'oracle rollout prefere une autre execution en moyenne.
- `Regret clair` applique les seuils statistiques stricts ci-dessus.
- `Execution forcee` signifie qu'il n'y avait qu'une seule execution legale de cette carte.
- `Carte forcee` signifie qu'au moment ou la carte a ete jouee, le modele n'avait pas d'autre carte/action principale legale.

## Prochaine Etape

Les types avec assez de `regret clair` deviennent candidats pour un dataset d'entrainement. Les autres doivent rester en audit ou etre collectes avec plus de rollouts/etats.

## Fichiers

- Dataset JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/datasets/execution_teacher_initial_40x12.json`
- Rapport JSON: `/Users/assebbi/Library/CloudStorage/OneDrive-UniversalMusicGroup/Love Letter Test/step5_execution_heads/reports/execution_teacher_initial_40x12_report.json`
- Log: `step5_execution_heads/logs/2026-04-25_execution_teacher_initial_40x12.md`
