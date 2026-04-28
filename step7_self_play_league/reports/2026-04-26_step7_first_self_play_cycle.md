# Step7 - Premier Cycle Self-Play

Date: 2026-04-26.

## Constat

La ligue Step7 est operationnelle: roster, Elo multi-joueurs, training
self-play, checkpoints iteratifs et gate de promotion.

Le bootstrap initial sur `10 000` manches donne `champion_cbp` comme meilleur
actif:

| Policy | Elo bootstrap |
|---|---:|
| `champion_cbp` | 1546.7 |
| `step2_retarget` | 1520.5 |
| `curriculum_phase1` | 1506.8 |
| `heuristic_fair` | 1479.5 |
| `step3_fast` | 1446.5 |

## Candidats Testes

### `sp_iter_0001`

- Parent: `champion_cbp`.
- Training: `10` iterations, `8192` decisions par iteration.
- Reglage: conservateur, KL forte vers le parent.
- Decision: rejete.

Evaluation ligue `10 000` manches:

| Metric | Champion | Candidat |
|---|---:|---:|
| Elo | 1519.78 | 1508.81 |
| Main win | 27.14% | 28.33% |
| Guard hit | 28.76% | 29.17% |
| Baron loss | 20.64% | 22.22% |
| Chancelier keep-highest | 89.11% | 89.42% |

Lecture: le candidat ne casse pas les acquis tactiques, gagne un peu plus de
manches principales, mais perd au classement Elo global.

### `sp_iter_0002`

- Parent: `champion_cbp`.
- Training: `10` iterations, `8192` decisions par iteration.
- Reglage: plus agressif, learning rate plus haut et KL reduite.
- Decision: rejete.

Evaluation ligue `10 000` manches:

| Metric | Champion | Candidat |
|---|---:|---:|
| Elo | 1519.66 | 1508.66 |
| Main win | 27.14% | 28.37% |
| Guard hit | 28.69% | 29.21% |
| Baron loss | 20.72% | 22.26% |
| Chancelier keep-highest | 89.19% | 89.49% |

Lecture: meme profil que `sp_iter_0001`. Le candidat apprend quelque chose de
positif sur la victoire principale, mais pas assez pour convertir cela en force
Elo globale.

## Conclusion

Ce premier cycle est un echec de promotion, mais pas un echec de pipeline.

Ce qu'il montre:

- La boucle self-play fonctionne techniquement.
- Les checkpoints reguliers permettent de recuperer chaque iteration.
- Les garde-fous tactiques filtrent correctement les candidats.
- L'acteur PPO pur ne suffit pas encore a depasser `champion_cbp`.

Hypothese principale: l'objectif PPO optimise le reward final de maniere trop
bruitee pour produire un saut Elo rapide. Les candidats ameliorent des signaux
locaux visibles, comme la victoire principale, mais ne dominent pas assez les
comparaisons pairwise de reward sur la population active.

Suite conseillee:

1. Ajouter une selection automatique du meilleur checkpoint intermediaire avant
   evaluation longue.
2. Faire une evaluation plus diagnostique par matchups fixes:
   candidat vs `champion_cbp`, vs `curriculum_phase1`, vs `step2_retarget`,
   vs `heuristic_fair`.
3. Ajouter un signal d'entrainement plus stable que le reward brut:
   advantage-to-parent, AWR/AWAC, ou distillation depuis les parties ou le
   candidat bat son parent.
4. Garder `champion_cbp` comme champion actif tant qu'aucun candidat ne passe le
   gate Elo.
