# Step5 - Baron Attempt1

Date: 2026-04-26.

## Contexte

Apres le succes Chancelier V1, on a traite Baron comme carte suivante. La
premiere erreur methodologique a ete corrigee pendant cette etape: une tete
locale doit etre comparee aussi sur les **parties ou la carte cible est
effectivement jouee**, pas seulement sur le composite global.

## Donnees Teacher

Dataset Baron global:

```text
step5_execution_heads/cards/baron/datasets/baron_target_teacher_240x12.json
```

Collecte:

```bash
python3 -m step5_execution_heads.collect_execution_teacher \
  --games 1800 \
  --states-per-config-kind 60 \
  --rollouts-per-action 12 \
  --kinds baron_target
```

Synthese:

| Mesure | Valeur |
|---|---:|
| Etats Baron collectes | 240 |
| Etats par composition | 60 |
| Etats entrainables | 173 |
| Regrets clairs | 55 |
| Best action differente | 100 / 240 = 41.67% |
| Mean score regret | 0.0738 |
| Mean win regret | 0.0688 |

Lecture: le teacher voit bien un signal Baron, mais il est moins fort que le
signal Chancelier.

## Tentative Low-Only

On a d'abord teste `baron_low_target`, c'est-a-dire Baron avec petite carte
gardee. C'etait trop rare en arena:

| Mesure | Valeur |
|---|---:|
| Checks sur 2000 parties | 25 |
| Overrides tete | 4 |
| Delta global | -0.00040 |

Conclusion: `baron_low_target` est trop fin comme module autonome. Il reste un
sous-cas utile a analyser, mais pas une bonne unite d'entrainement seule.

## Tete Baron Globale

Checkpoint:

```text
step5_execution_heads/cards/baron/checkpoints/baron_target_head_attempt1.pth
```

Validation entrainement:

| Mesure | Train | Val |
|---|---:|---:|
| Top1 action | 68.35% | 50.00% |
| Override agreement | 71.22% | 70.59% |
| Pred override rate | 21.58% | 14.71% |
| Target override rate | 47.48% | 38.24% |

Lecture: la tete apprend un signal, mais reste prudente et ne couvre qu'une
partie des corrections oracle.

## Evaluation Arena

Evaluation conditionnelle finale:

```text
step5_execution_heads/cards/baron/reports/baron_target_head_attempt1_eval_1000_m010_seed970000_conditional.md
```

### Global

| Politique | Composite |
|---|---:|
| Step3 rapide | 0.39610 |
| Baron random | 0.39770 |
| Step3 + tete Baron | 0.39650 |

Globalement, la tete est presque neutre: `+0.00040`.

### Parties Ou Baron Est Joue

| Politique | Winrate pondere |
|---|---:|
| Step3 rapide | 49.08% |
| Baron random | 49.50% |
| Step3 + tete Baron | 48.75% |

### Parties Ou La Cible Baron Est Randomisable

| Politique | Winrate pondere |
|---|---:|
| Step3 rapide | 43.32% |
| Baron random | 43.87% |
| Step3 + tete Baron | 42.89% |

## Conclusion

Baron Attempt1 n'est **pas** un succes.

La lecture sincere:

- le teacher rollout detecte bien du regret sur les cibles Baron;
- randomiser Baron degradeait dans certains runs courts, mais pas dans la
  validation conditionnelle finale;
- la tete actuelle ne bat ni Step3 rapide ni le controle random sur les parties
  ou Baron est vraiment joue;
- donc elle ne doit pas etre branchee dans le joueur Step5 actif.

## Hypotheses De Blocage

1. Le label rollout Baron est plus bruite que Chancelier, car l'issue d'un Baron
   depend beaucoup de cartes cachees et de continuations adverses.
2. L'observation/features de la tete cible ne capturent pas assez bien le belief
   implicite sur les cartes adverses.
3. La tete globale melange des situations differentes: Baron avec carte faible,
   moyenne, haute, fin de manche, cible deja connue, cible protegee juste avant,
   etc.
4. Le bon apprentissage Baron devra probablement utiliser explicitement les
   probabilites de belief ou des features de duel attendues.

## Suite Pour Baron

Ne pas poursuivre Baron en boucle aveugle maintenant.

Prochaine tentative Baron recommandee:

- ajouter des features explicites de duel:
  - carte gardee;
  - probabilite cible plus basse / egale / plus haute;
  - cible vue par Pretre;
  - cible ayant joue Comtesse, Roi, Prince, etc.;
- entrainer une tete Baron conditionnee au belief;
- evaluer seulement sur:
  - parties ou Baron est joue;
  - parties ou la cible est randomisable;
  - sous-cas `kept <= 4`, `kept 5-6`, `kept >= 7`.

Pour l'instant, on passe a la carte suivante seulement apres avoir note que
Baron Attempt1 est un echec utile, pas un module valide.
