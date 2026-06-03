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

Cette app remplace progressivement Streamlit pour l'experience de jeu finale.
Streamlit reste utile comme prototype/debug, mais l'objectif jeu est maintenant
`FastAPI + React/Vite`.

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
