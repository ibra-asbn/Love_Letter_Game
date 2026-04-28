# Love Letter Web App

Nouvelle interface web jeu:

- backend: FastAPI, moteur Love Letter, champion `champion_cbp`;
- frontend: React/Vite, direction artistique pixel art, menu et rappel des regles.

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

Cette app remplace progressivement Streamlit pour l'experience de jeu finale.
Streamlit reste utile comme prototype/debug, mais l'objectif jeu est maintenant
`FastAPI + React/Vite`.
