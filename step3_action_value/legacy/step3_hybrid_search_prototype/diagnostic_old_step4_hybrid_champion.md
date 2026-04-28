# Diagnostic Step4 - Champion Hybride Actor + Belief + Search

Date: 25 avril 2026.

## Etat De Depart

L'etape 3 avait valide une idee forte: les rollouts ameliorent les coups
tactiques du modele Step2. En revanche, la distillation dans une petite tete
rapide n'a pas recupere tout ce gain.

Decision prise: pour battre des humains, on privilegie un joueur qui reflechit
quelques secondes plutot qu'un actor instantane mais moins fort.

## Ce Qu'on A Fait

Creation du dossier:

```text
step4_hybrid_champion/
```

Ajout du script:

```text
step4_hybrid_champion/evaluate_hybrid_search.py
```

Le joueur Step4:

- charge `step2_retarget_distilled_attempt1.pth`;
- recupere les probabilites du belief a chaque decision;
- construit des candidats tactiques guides par le belief;
- sample les mains adverses cachees avec le belief et les cartes restantes;
- evalue les actions candidates par rollouts;
- override l'actor seulement si le meilleur candidat a une marge suffisante.

Cartes/cas couverts:

```text
guard, priest, spy, king, prince, chancellor_card, chancellor_choice, baron
```

## Resultats

Validation principale: `300` parties par configuration, seed `802000`,
`12` rollouts par action candidate.

| Politique | vs 3 randoms | vs 1H+2R | vs 2H+1R | vs 3H | Composite |
|---|---:|---:|---:|---:|---:|
| Step2 brut | 52.00% | 32.00% | 26.33% | 16.00% | 0.25900 |
| Step3 rollout uniforme | 56.33% | 46.67% | 29.00% | 25.00% | 0.33667 |
| Step4 hybride belief-search | 54.67% | 43.00% | 32.67% | 26.00% | 0.34267 |

Gains:

- Step4 vs Step2: `+8.37` points composite.
- Step4 vs Step3 uniforme: `+0.60` point composite.

## Lecture

Le succes principal n'est pas "le belief explose tout seul le score". Le vrai
succes est plus fin:

- le search tactique est devenu un vrai module de jeu;
- le belief aide a orienter les candidats et les determinizations;
- le gain du belief par rapport au search uniforme existe, mais reste modeste;
- Step4 est surtout plus fort parce qu'il accepte de calculer les coups
  importants au lieu de tout compresser dans l'actor.

## Points A Surveiller

- L'override rate est eleve, autour de `36-39%` des decisions player_0 selon
  la config.
- Les categories les plus modifiees sont surtout Garde, Baron, Chancelier,
  Prince et Roi.
- Le gain Step4 vs Step3 uniforme est petit sur 300 parties; il faut confirmer
  sur `1000` parties/config.
- Le joueur est trop lent pour de l'entrainement massif, mais acceptable pour
  une interface humaine.

## Conclusion

Verdict: succes Step4.

On a maintenant un candidat champion hybride, pas un simple checkpoint. Pour le
but "jouer fort contre un humain", cette direction est plus pertinente que
forcer immediatement une tete rapide.

Prochaine etape naturelle: brancher Step4 dans les interfaces jouables, puis
faire une evaluation longue et des diagnostics tactiques plus humains.

