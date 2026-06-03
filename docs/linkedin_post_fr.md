# Brouillon LinkedIn

Version courte, a adapter au ton personnel avant publication.

```text
J'ai passe les dernieres semaines a construire un agent IA pour jouer a Love Letter.

Le projet est parti d'une question simple: est-ce qu'un jeu court, cache, tres tactique, peut devenir un bon terrain d'experimentation pour du reinforcement learning ?

J'ai avance par paliers:

1. verifier que le jeu etait exploitable avec une heuristique forte;
2. entrainer un modele a imiter cette heuristique;
3. le faire depasser l'heuristique via un actor qui exploite mieux son belief;
4. ajouter une tete action-value pour corriger les decisions;
5. analyser les faiblesses carte par carte;
6. construire des modules locaux pour mieux executer Chancelier, Baron et Prince;
7. tester le champion dans une ligue de self-play.

Le resultat final n'est pas "un gros modele magique".

C'est plutot une pipeline lisible:
- un moteur de jeu audite;
- des evaluations reproductibles;
- des rapports d'erreur;
- un champion compose, `champion_cbp`;
- une app web jouable en FastAPI + React/Vite.

Le plus interessant pour moi: les meilleures ameliorations ne sont pas venues d'un entrainement global plus long, mais de corrections locales bien diagnostiquees. Quand le modele savait deja quelle carte jouer, il fallait surtout lui apprendre a mieux l'executer.

Le projet est maintenant dans un etat propre:
- documentation consolidee;
- resultats experimentaux conserves;
- checkpoints gardes hors Git classique;
- app jouable;
- prochaine reprise possible autour du self-play et du polish produit.

C'etait un bon rappel qu'en IA appliquee, la partie importante n'est pas seulement d'entrainer. C'est de comprendre ce qu'on mesure, pourquoi on l'ameliore, et quand il faut s'arreter proprement.
```

## Variante Plus Courte

```text
J'ai construit un agent IA pour jouer a Love Letter, de l'heuristique au self-play.

Le projet a suivi une pipeline assez complete:
- imitation d'un bot heuristique;
- fine-tuning pour le depasser;
- action-value locale;
- analyse des faiblesses par carte;
- modules specialises pour Chancelier, Baron et Prince;
- ligue de self-play;
- app web jouable en FastAPI + React/Vite.

Le plus beau resultat: les gains les plus stables sont venus de diagnostics precis, pas d'un entrainement global plus long.

Le champion final, `champion_cbp`, est une composition de modeles et de tetes locales. Le repo est maintenant documente proprement, avec les resultats, les decisions, et les limites.

Prochaine etape possible: publier les checkpoints en release externe et continuer le polish produit.
```

