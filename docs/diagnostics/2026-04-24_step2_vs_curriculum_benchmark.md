# Benchmark Step2 vs Curriculum Phase 1

Date: 2026-04-24, Europe/Paris.

## Objectif

Comparer le nouveau checkpoint step2 au champion historique `curriculum_phase1.pth`, sans faire de ce checkpoint historique le centre de la prochaine pipeline.

Question: est-ce que le modele step2, qui bat maintenant `HeuristicBot`, bat aussi l'ancien champion ?

## Checkpoints

Modele step2:

```text
step2_rl_finetune/checkpoints/step2_retarget_distilled_attempt1.pth
```

Champion historique:

```text
models/checkpoints/curriculum_phase1.pth
```

Rapport brut:

```text
step2_rl_finetune/reports/step2_vs_curriculum_benchmark_3000.json
```

Log:

```text
step2_rl_finetune/logs/2026-04-24_step2_vs_curriculum_benchmark_3000.md
```

## Arena Commune

Chaque checkpoint joue `player_0` contre les memes compositions d'adversaires, sur `3000` parties par configuration. C'est la comparaison la plus propre.

| Modele player_0 | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 retarget distille | 50.20% | 35.60% | 25.10% | 15.30% | 0.25790 |
| `curriculum_phase1.pth` | 55.73% | 42.10% | 29.23% | 16.53% | 0.29377 |

Conclusion: `curriculum_phase1.pth` reste devant au score global.

## Tables Directes

Ces tables sont indicatives: Love Letter a 4 joueurs est sensible au siege et aux adversaires presents.

| Table | Step2 | Curriculum | Autres |
|---|---:|---:|---:|
| Step2 + Curriculum + 2 randoms | 42.93% | 41.07% | randoms 15.70% |
| Step2 vs 3 Curriculum | 24.93% | 29.38% | - |
| Curriculum vs 3 Step2 | Step2 26.33% | 32.00% | - |
| 2 Step2 vs 2 Curriculum | 23.20% | 31.93% | - |

Lecture: en face-a-face isole avec deux randoms, step2 tient tres bien. Des qu'il y a plusieurs copies du curriculum a table, curriculum garde l'avantage.

## KPIs Tactiques

Agreges sur l'arena commune.

| KPI | Step2 | Curriculum |
|---|---:|---:|
| Garde hit rate | 35.24% | 28.15% |
| Garde suit le top belief | 89.28% | 80.56% |
| Baron favorable | 77.48% | 80.33% |
| Baron perdant | 20.05% | 17.15% |
| Prince touche Princesse | 9.64% | 8.49% |
| Roi meilleur echange | 40.56% | 23.30% |
| Roi pire echange | 55.30% | 71.89% |
| Chancelier garde la meilleure carte visible | 98.71% | 51.83% |

## Lecture

Step2 est tactiquement meilleur sur plusieurs gestes precis:

- il devine mieux avec le Garde;
- il suit beaucoup mieux son belief;
- il joue tres bien le Chancelier apres la correction d'observation;
- il fait de meilleurs Rois que curriculum.

Mais curriculum reste meilleur en score global. Les raisons probables:

- curriculum a une politique globale plus equilibree;
- step2 est tres fort sur les cibles/devinettes, mais pas encore assez fort sur le choix de carte lui-meme;
- step2 semble jouer beaucoup plus de Chancelier et Prince, moins de Garde/Espionne que curriculum;
- curriculum fait de meilleurs Barons en moyenne;
- les gains tactiques locaux ne suffisent pas encore a compenser les decisions de tempo, conservation de main, et survie.

Distribution d'actions observee:

| Carte/action | Step2 | Curriculum |
|---|---:|---:|
| Garde | 26.8% | 32.6% |
| Espionne | 5.9% | 11.2% |
| Pretre | 10.8% | 11.1% |
| Baron | 8.5% | 8.3% |
| Servante | 9.9% | 10.9% |
| Prince | 10.0% | 9.5% |
| Chancelier + choix | 20.6% | 10.2% |
| Roi | 4.3% | 3.0% |
| Comtesse | 3.1% | 3.2% |

## Verdict

Step2 est une vraie avance par rapport a `HeuristicBot`, mais pas encore un nouveau champion global.

Il ne faut pas abandonner la pipeline step1/step2: elle a produit des competences tactiques tres nettes. Mais l'etape suivante doit viser le choix de carte et le tempo global, pas seulement les cibles/devinettes.

## Suite Recommandee

Pour l'etape 3, ne pas simplement "copier curriculum".

Meilleure direction:

1. garder step2 comme base;
2. ajouter une distillation tactical sur les choix de carte quand le signal est fort;
3. reduire les mauvais Barons;
4. controler l'exces de Chancelier/Prince;
5. lancer ensuite du RL contre une league incluant `HeuristicBot`, step2, et curriculum;
6. choisir le champion uniquement sur arena longue.

