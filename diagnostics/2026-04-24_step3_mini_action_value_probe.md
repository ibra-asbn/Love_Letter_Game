# Mini Test Step 3 - Action-Value Rollouts

Date: 2026-04-24, Europe/Paris.

## Objectif

Verifier rapidement si une approche `obs + action -> valeur estimee` est pertinente pour la suite, au lieu de continuer a ecrire des regles tactiques manuelles.

Script cree:

```text
step3_action_value/mini_rollout_probe.py
```

Rapports:

```text
step3_action_value/reports/mini_rollout_probe_step2_60.json
step3_action_value/reports/mini_rollout_probe_step2_baron_80.json
```

Logs:

```text
step3_action_value/logs/2026-04-24_mini_rollout_probe_step2_60.md
step3_action_value/logs/2026-04-24_mini_rollout_probe_step2_baron_80.md
```

## Methode

Le script:

1. collecte des etats critiques joues par `step2_retarget_distilled_attempt1.pth`;
2. liste les actions legales;
3. sample des determinisations des cartes cachees coherentes avec ce que `player_0` sait;
4. joue des fins de manche depuis chaque action candidate;
5. estime un winrate et un reward moyen par action.

Ce n'est pas encore une Q-value parfaite:

- peu d'etats;
- peu de rollouts;
- belief sampler approximatif;
- continuation actuelle simple avec `HeuristicBot`.

Mais c'est suffisant pour voir si le signal existe.

## Mini Probe General

Configuration:

- `5` etats critiques;
- `60` rollouts par action;
- categories: Baron, Roi, Chancelier, Prince, Garde;
- continuation: heuristique.

Resultat:

| Mesure | Valeur |
|---|---:|
| Etats evalues | 5 |
| Regret moyen du modele vs meilleure action rollout | 7.67 pts winrate |
| Regret max observe | 16.67 pts winrate |

Exemples:

| Etat | Main | Action modele | Meilleure action rollout | Regret |
|---|---|---|---|---:|
| Roi/Garde early | Roi + Garde | Garde p3 Baron | Garde p1 Comtesse | 6.67 pts |
| Prince/Garde | Garde + Prince | Prince sur soi | Garde p1 Chancelier | 6.67 pts |
| Garde/Servante | Servante + Garde | Garde p1 Baron | Garde p2 Baron | 16.67 pts |
| Chancelier/Espionne | Espionne + Chancelier | Chancelier | Espionne | 8.33 pts |

Lecture: meme avec un mini-test bruite, les rollouts trouvent des alternatives plausibles et parfois nettement meilleures. C'est exactement le signal qu'il faut pour apprendre une action-value.

## Zoom Baron

Configuration:

- `5` etats Baron;
- `80` rollouts par action.

Resultat:

| Mesure | Valeur |
|---|---:|
| Etats Baron evalues | 5 |
| Regret moyen du modele | 4.5 pts winrate |
| Regret max | 10 pts winrate |

Observations:

- quand le modele a `Servante + Baron`, il choisit correctement Servante dans un etat teste;
- quand il a `Pretre + Baron` en fin de manche, il choisit correctement Pretre;
- dans plusieurs etats `Prince + Baron`, les rollouts preferent parfois Prince ou une autre cible Baron;
- le probleme Baron n'est donc pas seulement "jouer Baron trop souvent", mais surtout "choisir le bon moment et la bonne cible".

Exemples Baron:

| Main | Action modele | Meilleure action rollout | Regret |
|---|---|---|---:|
| Servante + Baron | Servante | Servante | 0 pt |
| Baron + Chancelier | Baron p2 | Baron p1 | 5 pts |
| Pretre + Baron | Pretre p1 | Pretre p1 | 0 pt |
| Prince + Baron | Baron p1 | Baron p3 | 10 pts |
| Prince + Baron | Baron p2 | Prince p3 | 7.5 pts |

## Conclusion

Oui, c'est pertinent.

Le mini-test montre que des rollouts peuvent produire des labels plus riches que nos KPIs tactiques manuels:

- ils ne disent pas "Baron toujours bon/mauvais";
- ils distinguent le moment, la cible, le deck, les cartes jouees et l'information;
- ils peuvent trouver des actions meilleures que le modele sur des etats concrets;
- ils peuvent servir a entrainer une tete `Q(obs, action)` ou a distiller une politique amelioree.

La prochaine vraie etape devrait etre une version robuste de ce script:

1. collecter beaucoup plus d'etats critiques;
2. utiliser un belief sampler plus propre;
3. augmenter les rollouts sur les etats importants;
4. stocker les valeurs par action;
5. entrainer une action-value head;
6. tester le reranking d'actions avant de faire du RL lourd.

