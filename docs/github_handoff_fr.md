# Handoff GitHub

Etat consolide le 2026-06-03.

Ce fichier explique quoi publier dans Git et quoi garder comme artefact externe.

## Objectif Du Handoff

Le repo doit raconter proprement le projet et permettre de:

- comprendre la progression IA et les decisions;
- lancer les tests et l'app web;
- retrouver les rapports principaux;
- eviter de versionner des sorties generees ou checkpoints lourds.

## Fichiers A Mettre En Avant

| Fichier | Role |
|---|---|
| `README.md` | Vue globale du projet |
| `docs/project_journal_fr.md` | Recap canonique etape par etape |
| `docs/github_handoff_fr.md` | Consignes de publication |
| `docs/linkedin_post_fr.md` | Brouillon de communication |
| `love_letter_web/README.md` | Lancer l'app FastAPI + React/Vite |
| `tests/test_love_letter_web_backend.py` | Couverture backend produit |

## A Versionner

- Code source Python et React.
- README et rapports Markdown utiles.
- Tests.
- Petits assets web necessaires a l'experience.
- JSON de resultats seulement quand ils sont importants pour reproduire un
  rapport ou une decision.

## A Garder Hors Git Classique

- Checkpoints `.pth`.
- Datasets volumineux.
- Logs locaux.
- `node_modules/`, `dist/`, `build/`.
- `storybook-static/`.
- Captures Playwright locales dans `output/playwright/`.
- Liens locaux Vercel ou fichiers secrets.

## Artefacts Externes Recommandes

Pour publier completement le projet, les checkpoints peuvent etre distribues
via une de ces options:

- GitHub Release;
- Git LFS;
- Hugging Face Hub;
- stockage cloud avec liens documentes.

Minimum utile:

- `step2_retarget_distilled_attempt1.pth`;
- `step3_advantage_v2_dagger_attempt1_iter1.pth`;
- tetes Step5 Chancelier/Baron/Prince utilisees par `champion_cbp`;
- `curriculum_phase1.pth` comme sparring partner historique.

## Nettoyage Avant Commit

Commandes utiles:

```bash
git -c core.fsmonitor=false ls-files -m -d
git -c core.fsmonitor=false ls-files --others --exclude-standard
python3 -m pytest tests/test_love_letter_web_backend.py
cd love_letter_web/frontend
npm run build
```

Notes:

- Sur ce workspace OneDrive, `git status` global peut etre lent. Preferer les
  commandes ciblees ci-dessus.
- Ne pas supprimer les rapports Markdown: ils sont la memoire experimentale du
  projet.
- Ne pas ajouter les checkpoints lourds au commit final sans decision explicite.

## Etat De Pause

Le point d'arret propre est:

- champion final documente: `champion_cbp`;
- web app documentee;
- projet lisible dans GitHub;
- post LinkedIn prepare;
- pas de nouvelle experimentation lancee.

