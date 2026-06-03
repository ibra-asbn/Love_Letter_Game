# Love Letter Web App

Interface web jouable du projet Love Letter RL.

- backend: FastAPI, moteur Love Letter, champion `champion_cbp`;
- frontend: React/Vite, menu Qadi, tutoriel, regles, cartes, parametres IA;
- produit: profils joueurs, stats locales, logs structures et replay omniscient
  en fin de partie.

## Lancer

Depuis la racine du projet:

```bash
uvicorn love_letter_web.backend.main:app --host 127.0.0.1 --port 8000
```

Dans un second terminal:

```bash
cd love_letter_web/frontend
npm install
npm run dev
```

Puis ouvrir:

```text
http://127.0.0.1:5173/
```

## Etat

L'objectif jeu est maintenant `FastAPI + React/Vite`.

## Hebergement Et Donnees

| Service | Lien |
|---|---|
| Jeu complet Vercel | https://love-letter-game-pi.vercel.app/ |
| Backend Hugging Face Spaces | https://ibra-asbn-love-letter-api.hf.space |
| Health check backend | https://ibra-asbn-love-letter-api.hf.space/api/health |
| Space Hugging Face | https://huggingface.co/spaces/ibra-asbn/love-letter-api |

Architecture de production:

- backend FastAPI heberge sur Hugging Face Spaces;
- frontend React/Vite heberge sur Vercel;
- variable frontend de production:
  `VITE_API_BASE=https://ibra-asbn-love-letter-api.hf.space`;
- en local, `vite.config.js` proxy `/api` vers `http://127.0.0.1:8000`.

Flux de donnees:

- `frontend/src/api.js` construit tous les appels API;
- `POST /api/games` cree une partie;
- `GET /api/games/{game_id}` recupere l'etat courant;
- `POST /api/games/{game_id}/play` envoie l'action humaine;
- `POST /api/games/{game_id}/ai-step` avance les IA;
- `GET /api/games/{game_id}/replay` recupere le replay final;
- les profils, les choix d'IA et les reglages audio sont stockes en
  `localStorage`;
- les stats et evenements backend sont stockes dans `love_letter_web/logs/`,
  ignore par Git.

## Fonctionnalites Actuelles

- Nouvelle partie contre trois adversaires IA configurables.
- Champion par defaut: `champion_cbp`.
- Parametres par adversaire: `champion_cbp`, `step3_fast`, `step2_retarget`,
  `heuristic_fair`, `random`.
- Profil joueur avec prenom, nom, motif d'entree et dialogues du Qadi.
- Rappel des regles et preview des cartes.
- Tutoriel guide.
- Journal de partie, bulles de dialogue et effets visuels.
- Replay omniscient disponible uniquement a la fin d'une partie.
- Stats locales stockees dans `love_letter_web/logs/`, ignorees par Git.

## Tests

Depuis la racine:

```bash
python3 -m pytest tests/test_love_letter_web_backend.py
```

Depuis `love_letter_web/frontend`:

```bash
npm run build
```
