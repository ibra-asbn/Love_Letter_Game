# Diagnostic Step3 v2 - Advantage Head + Verification CRN

Date: 25 avril 2026.

## Contexte

L'objectif de Step3 est de depasser Step2 sans ajouter de nouvelles regles
strategiques manuelles. Step2 (`step2_retarget_distilled_attempt1.pth`) sait
deja battre `HeuristicBot`, mais il laisse encore de la valeur sur certains
choix tactiques: cible de Garde, Baron risque, Prince, Roi, Pretre, Chancelier.

Les avis experts ajoutes dans `step3_action_value/` allaient dans le meme sens:

- ne pas apprendre une Q-value absolue bruitee;
- comparer les actions avec des Common Random Numbers / rollouts apparies;
- apprendre un avantage relatif a l'action Step2;
- traiter les coups ambigus comme des egalites, pas comme des labels durs;
- verifier statistiquement les corrections avant de les faire confiance.

## Ce Qui A Ete Fait

Deux nouveaux scripts ont ete ajoutes:

```text
step3_action_value/train_advantage_head_v2.py
step3_action_value/evaluate_advantage_head_v2.py
```

`train_advantage_head_v2.py` collecte des etats Step2, construit plusieurs
actions candidates, puis evalue ces actions par rollouts apparies. Pour chaque
decision, les actions candidates partagent les memes mondes caches echantillonnes
et les memes seeds de playout. Cela reduit fortement la variance par rapport a
des rollouts independants.

La tete apprise predit:

```text
advantage(s, action_candidate) = valeur(action_candidate) - valeur(action_step2)
```

Elle recoit:

- l'observation du jeu;
- le hidden recurrent de l'actor Step2;
- le belief adversaire;
- des features de phase de partie;
- l'action candidate, l'action Step2 et l'action HeuristicBot.

`evaluate_advantage_head_v2.py` garde Step2 comme action par defaut. La tete
peut proposer un override, mais dans la version retenue ce coup est ensuite
verifie par un petit rollout CRN qui compare seulement:

```text
action Step2 vs action proposee par la tete
```

## Tentative 1

Checkpoint:

```text
step3_action_value/checkpoints/step3_advantage_v2_attempt1.pth
```

La collecte trouvait bien un signal, mais la policy rapide seule etait trop
enthousiaste. Elle corrigeait souvent Step2 vers des petits coups type Garde ou
Espionne dans des contextes Chancelier/Prince, puis regressait contre 3
heuristiques.

Meilleur test court:

| Reglage | Composite | Baseline Step2 | Delta |
|---|---:|---:|---:|
| Attempt1, marge 0.16 + entropy 0.06 | 0.27400 | 0.27200 | +0.00200 |
| Attempt1, meme reglage + max 3 joueurs actifs | 0.27600 | 0.27200 | +0.00400 |

Verdict: signal reel, mais pas un succes. Trop fragile.

## Tentative 2 Stricte

Checkpoint retenu:

```text
step3_action_value/checkpoints/step3_advantage_v2_attempt2_strict.pth
```

Changements:

- `24` rollouts par action pendant la collecte;
- seuil d'effet plus dur: `min_win_delta=0.125`;
- seuil statistique plus dur: `t_threshold=1.65`;
- poids plus fort sur les actions ambigues pour eviter les hallucinations
  d'avantage;
- entrainement plus court pour limiter l'overfit.

Collecte:

| Mesure | Valeur |
|---|---:|
| Etats collectes | 280 |
| Lignes action | 1741 |
| Actions significatives non-Step2 | 275 / 1461 |
| Taux actions significatives | 18.82% |
| Etats avec correction positive | 34 / 280 |
| Taux correction positive | 12.14% |
| Correlation CRN moyenne | 0.382 |

La tete rapide seule reste instable:

| Evaluation 1000/config | Composite Step3 | Composite Step2 | Delta |
|---|---:|---:|---:|
| Seed 134000, marge 0.10 | 0.28000 | 0.26720 | +0.01280 |
| Seed 135000, marge 0.10 | 0.25890 | 0.26580 | -0.00690 |

Lecture: le modele a appris un signal, mais pas assez calibre pour etre laisse
seul. On n'aurait pas du declarer cela comme succes.

## Succes: Advantage + Verification CRN

Reglage retenu:

```bash
python3 -m step3_action_value.evaluate_advantage_head_v2 \
  --checkpoint step3_advantage_v2_attempt2_strict.pth \
  --override-margin 0.10 \
  --verify-rollouts 16 \
  --verify-min-win-delta 0.125 \
  --verify-min-score-delta 0.05 \
  --verify-t-threshold 0.75
```

La tete propose une correction, puis le verificateur CRN accepte seulement si
la correction bat l'action Step2 dans des rollouts apparies. Cela transforme la
tete en proposeur rapide et les rollouts en filet de securite.

Validation sur trois seeds independants, `1000` parties par composition:

| Seed | Composite Step3 v2 verifie | Composite Step2 | Delta |
|---|---:|---:|---:|
| 134000 | 0.28230 | 0.26720 | +0.01510 |
| 135000 | 0.27890 | 0.26580 | +0.01310 |
| 136000 | 0.27890 | 0.26330 | +0.01560 |
| Moyenne | 0.28003 | 0.26543 | +0.01460 |

Moyenne par composition:

| Composition | Step3 v2 verifie | Step2 | Delta |
|---|---:|---:|---:|
| vs 3 randoms | 53.30% | 51.57% | +1.73 pts |
| vs 1H+2R | 38.87% | 37.50% | +1.37 pts |
| vs 2H+1R | 26.47% | 24.73% | +1.73 pts |
| vs 3H | 17.40% | 16.17% | +1.23 pts |

Statistiques de verification agregees:

| Mesure | Valeur |
|---|---:|
| Propositions verifiees | 5603 |
| Overrides acceptes | 1409 |
| Overrides rejetes | 4194 |
| Taux d'acceptation | 25.15% |
| Override / decision | 3.53% |

Acceptations par categorie:

| Categorie | Verifications | Acceptations | Taux |
|---|---:|---:|---:|
| Garde | 2311 | 451 | 19.5% |
| Baron | 973 | 274 | 28.2% |
| Pretre | 829 | 261 | 31.5% |
| Roi | 511 | 183 | 35.8% |
| Prince | 637 | 174 | 27.3% |
| Chancelier | 254 | 46 | 18.1% |
| Espionne | 88 | 20 | 22.7% |

## Conclusion

Step3 v2 est un succes sous forme hybride:

```text
Step2 actor + belief
  -> tete advantage v2 propose une correction
  -> verification CRN locale
  -> override seulement si le gain est confirme
```

Ce n'est pas encore une tete rapide autonome. La tete seule reste trop fragile.
Mais c'est un vrai nouveau joueur officiel de la pipeline: il ameliore Step2
sur trois validations longues, dans toutes les compositions adverses, avec un
gain moyen de `+1.46` point de composite.

La suite logique n'est pas d'ouvrir une nouvelle etape. La suite propre est de
continuer Step3:

1. utiliser ce Step3 v2 verifie comme joueur/oracle actuel;
2. collecter les decisions acceptees/rejetees par le verificateur;
3. distiller cette decision finale, pas seulement les rollouts bruts;
4. viser une tete rapide autonome qui reproduit le verificateur sans perdre la
   stabilite.
