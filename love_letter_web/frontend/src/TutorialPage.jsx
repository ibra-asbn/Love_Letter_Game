import React from "react";
import { ArrowLeft, ArrowRight, CheckCircle, Eye, Play, Shield, Target, XCircle } from "lucide-react";
import { api } from "./api";
import { cardArtById } from "./assets/cards";
import { profileRequestPayload } from "./playerProfile";
import { pushView } from "./navigation";

const PLAYERS = {
  human: "player_0",
  sultane: "player_1",
  sultan: "player_2",
  amira: "player_3",
};

const names = {
  [PLAYERS.human]: "Qadi",
  [PLAYERS.sultane]: "La Sultane",
  [PLAYERS.sultan]: "Le Sultan",
  [PLAYERS.amira]: "L'Amira",
};

function player(id, hand, played = [], extras = {}) {
  return {
    id,
    name: names[id],
    hand,
    played,
    alive: extras.alive ?? true,
    protected: extras.protected ?? false,
    note: extras.note || "",
  };
}

const tutorialScenes = [
  {
    title: "Je prépare une partie-école",
    qadi: "Avant d'ouvrir l'arène, je ressors une ancienne partie. Chaque choix mérite d'être observé: je commenterai mes décisions une à une.",
    action: "Leçon: j'installe la table",
    current: PLAYERS.human,
    recommendation: "J'observe d'abord mes deux cartes.",
    journal: "Je ressors une ancienne partie du registre.",
    players: [
      player(PLAYERS.human, [2, 1]),
      player(PLAYERS.sultane, [8]),
      player(PLAYERS.sultan, [5]),
      player(PLAYERS.amira, [3]),
    ],
    targetHints: {},
  },
  {
    title: "J'ai Qadi et Garde",
    qadi: "Le Garde peut éliminer, mais seulement si j'annonce juste. La carte Qadi est plus sûre: elle me donne une information avant de prendre un risque.",
    action: "Tour de Qadi: je choisis une carte",
    current: PLAYERS.human,
    recommendedCard: 2,
    recommendation: "Je joue Qadi plutôt que Garde.",
    journal: "Ma main: Qadi (2) et Garde (1).",
    players: [
      player(PLAYERS.human, [2, 1]),
      player(PLAYERS.sultane, [8]),
      player(PLAYERS.sultan, [5]),
      player(PLAYERS.amira, [3]),
    ],
    targetHints: {
      [PLAYERS.sultane]: { legal: true, text: "Cible possible, mais moins urgente." },
      [PLAYERS.sultan]: { legal: true, recommended: true, text: "Cible possible et utile: il joue bientôt." },
      [PLAYERS.amira]: { legal: true, text: "Cible possible, mais pas recommandée ici." },
    },
  },
  {
    title: "J'inspecte Le Sultan",
    qadi: "Le Sultan n'est pas protégé: je peux le cibler. Je regarde sa carte et je garde l'information pour plus tard.",
    action: "Tour de Qadi: je joue Qadi (2) sur Le Sultan",
    current: PLAYERS.sultane,
    recommendedTarget: PLAYERS.sultan,
    recommendation: "Info obtenue: Le Sultan garde Wali.",
    journal: "Je vois la carte du Sultan: Wali (5).",
    secretNote: {
      title: "Information obtenue",
      text: "J'ai ciblé Le Sultan: sa carte est Wali (5).",
    },
    players: [
      player(PLAYERS.human, [1], [2]),
      player(PLAYERS.sultane, [8]),
      player(PLAYERS.sultan, [5], [], { note: "Carte vue" }),
      player(PLAYERS.amira, [3]),
    ],
    targetHints: {
      [PLAYERS.sultan]: { legal: true, recommended: true, text: "Cible choisie: Le Sultan révèle Wali (5)." },
    },
  },
  {
    title: "La Sultane rate son Garde",
    qadi: "L'adversaire peut se tromper. Le Garde est puissant, mais il punit les suppositions fragiles.",
    action: "Tour de La Sultane: elle joue Garde (1) sur Qadi et annonce Amira (9)",
    current: PLAYERS.sultan,
    recommendation: "Elle rate: le Qadi garde Garde (1).",
    journal: "Le Garde rate: le Qadi n'avait pas Amira (9).",
    players: [
      player(PLAYERS.human, [1]),
      player(PLAYERS.sultane, [8], [1]),
      player(PLAYERS.sultan, [5]),
      player(PLAYERS.amira, [3]),
    ],
    targetHints: {},
  },
  {
    title: "Le Sultan donne peu d'information",
    qadi: "Le Sultan utilise aussi la carte Qadi, mais il ne change pas ma main. L'information que j'ai sur son Wali reste exploitable.",
    action: "Tour du Sultan: il joue Qadi (2) sur Qadi",
    current: PLAYERS.amira,
    recommendation: "Il garde toujours Wali (5).",
    journal: "Le Sultan regarde la main du Qadi.",
    players: [
      player(PLAYERS.human, [1]),
      player(PLAYERS.sultane, [8], [1]),
      player(PLAYERS.sultan, [5], [2], { note: "Wali confirmé" }),
      player(PLAYERS.amira, [3]),
    ],
    targetHints: {},
  },
  {
    title: "L'Amira se protège",
    qadi: "Le Hajib rend une cible impossible jusqu'au prochain tour de ce joueur. Je dois donc éviter L'Amira pour l'instant.",
    action: "Tour de L'Amira: elle joue Hajib (4)",
    current: PLAYERS.human,
    recommendation: "Je ne cible pas L'Amira: elle est protégée.",
    journal: "L'Amira est protégée jusqu'à son prochain tour.",
    players: [
      player(PLAYERS.human, [1, 3]),
      player(PLAYERS.sultane, [8], [1]),
      player(PLAYERS.sultan, [5], [2], { note: "Wali connu" }),
      player(PLAYERS.amira, [3], [4], { protected: true }),
    ],
    targetHints: {
      [PLAYERS.sultane]: { legal: true, text: "Cible possible." },
      [PLAYERS.sultan]: { legal: true, recommended: true, text: "Cible possible. Je connais sa carte." },
      [PLAYERS.amira]: { legal: false, text: "Cible interdite: protégée par Hajib." },
    },
  },
  {
    title: "J'utilise l'information",
    qadi: "Maintenant le Garde devient fort: je ne devine plus au hasard, j'utilise une information déjà obtenue.",
    action: "Tour de Qadi: je joue Garde (1) sur Le Sultan et annonce Wali (5)",
    current: PLAYERS.human,
    recommendedCard: 1,
    recommendedTarget: PLAYERS.sultan,
    recommendation: "Je joue Garde sur Le Sultan et j'annonce Wali.",
    journal: "Le Garde touche: Le Sultan avait Wali (5).",
    players: [
      player(PLAYERS.human, [3], [2, 1]),
      player(PLAYERS.sultane, [8], [1]),
      player(PLAYERS.sultan, [], [2, 5], { alive: false }),
      player(PLAYERS.amira, [3], [4], { protected: true }),
    ],
    targetHints: {
      [PLAYERS.sultan]: { legal: true, recommended: true, text: "Cible possible et connue: Wali (5)." },
      [PLAYERS.amira]: { legal: false, text: "Toujours protégée, donc interdite." },
    },
  },
  {
    title: "Fin de ma démonstration",
    qadi: "Voilà ce que je voulais montrer: je prends d'abord l'information avec Qadi, puis j'utilise Garde seulement quand je peux frapper juste.",
    action: "Leçon: je clôture la séquence",
    current: PLAYERS.human,
    recommendation: "La leçon est terminée. La vraie partie peut commencer.",
    journal: "Je clôture la démonstration et désigne la table de jeu.",
    players: [
      player(PLAYERS.human, [3], [2, 1]),
      player(PLAYERS.sultane, [8], [1]),
      player(PLAYERS.sultan, [], [2, 5], { alive: false }),
      player(PLAYERS.amira, [3], [4], { protected: true }),
    ],
    targetHints: {},
  },
];

function TutorialCard({ card, muted = false, selected = false }) {
  const art = cardArtById[card];
  if (!art) return <span className="tutorial-empty-card">vide</span>;
  return (
    <figure className={`tutorial-card ${muted ? "is-muted" : ""} ${selected ? "is-selected" : ""}`}>
      <img src={art.image} alt={`${art.gameName} (${card})`} />
      <figcaption>{art.gameName}</figcaption>
    </figure>
  );
}

function TutorialTarget({ player: target, hint, active }) {
  const legal = target.alive ? (hint?.legal ?? true) : false;
  const legalityLabel = !target.alive ? "Joueur sorti" : legal ? "Cible possible" : "Cible interdite";
  return (
    <article className={`tutorial-target ${active ? "is-active" : ""} ${legal ? "is-legal" : "is-illegal"} ${target.alive ? "" : "is-dead"}`}>
      <header>
        <strong>{target.name}</strong>
        {target.protected ? (
          <span>
            <Shield size={14} />
            Protégée
          </span>
        ) : null}
        {!target.alive ? <span>Sorti</span> : null}
      </header>
      <div className="tutorial-target-card">
        {target.hand.length ? target.hand.map((card) => (
          <TutorialCard key={`${target.id}-hand-${card}`} card={card} muted={!active && !hint?.recommended} />
        )) : <TutorialCard card={null} />}
      </div>
      <div className="tutorial-target-played">
        {target.played.length ? target.played.map((card, index) => (
          <TutorialCard key={`${target.id}-played-${index}-${card}`} card={card} muted />
        )) : <span>aucune carte jouée</span>}
      </div>
      <p>{hint?.text || target.note || "Main visible pour suivre la leçon."}</p>
      <em>
        {legal ? <CheckCircle size={14} /> : <XCircle size={14} />}
        {legalityLabel}
      </em>
    </article>
  );
}

export function TutorialPage({ onNavigate }) {
  const [stepIndex, setStepIndex] = React.useState(0);
  const scene = tutorialScenes[stepIndex];
  const human = scene.players.find((item) => item.id === PLAYERS.human);
  const opponents = scene.players.filter((item) => item.id !== PLAYERS.human);
  const isFirst = stepIndex === 0;
  const isLast = stepIndex === tutorialScenes.length - 1;

  React.useEffect(() => {
    let active = true;
    api("/api/games", {
      method: "POST",
      body: JSON.stringify({
        ...profileRequestPayload(),
        is_tutorial: true,
      }),
    })
      .then(() => {
        if (!active) return;
      })
      .catch(() => {
        if (!active) return;
      });
    return () => {
      active = false;
    };
  }, []);

  function navigate(view) {
    if (onNavigate) {
      onNavigate(view);
      return;
    }
    pushView(view);
  }

  return (
    <main className="tutorial-page-screen">
      <section className="tutorial-shell" aria-label="Tutoriel guidé du Qadi">
        <aside className="tutorial-qadi-panel">
          <div className="panel-kicker">Partie-école</div>
          <h1>{scene.title}</h1>
          <p>{scene.qadi}</p>
        </aside>

        <section className="tutorial-board" aria-label="Table du tutoriel guidé">
          <div className="tutorial-scripted-event">
            <span>Déroulé</span>
            <strong>{scene.action}</strong>
          </div>
          <div className="tutorial-target-grid">
            {opponents.map((opponent) => {
              const hint = scene.targetHints[opponent.id];
              return (
                <TutorialTarget
                  key={opponent.id}
                  player={opponent}
                  hint={hint}
                  active={scene.recommendedTarget === opponent.id || Boolean(hint?.recommended)}
                />
              );
            })}
          </div>

          <div className="tutorial-human-zone">
            <div>
              <strong>{human.name}</strong>
              <span>{scene.current === PLAYERS.human ? "Mon tour" : `Tour: ${names[scene.current]}`}</span>
            </div>
            <div className="tutorial-hand">
              {human.hand.map((card, index) => (
                <TutorialCard
                  key={`${card}-${index}`}
                  card={card}
                  selected={scene.recommendedCard === card}
                  muted={Boolean(scene.recommendedCard && scene.recommendedCard !== card)}
                />
              ))}
            </div>
            <div className="tutorial-recommendation">
              <Eye size={16} />
              <span>{scene.recommendation}</span>
            </div>
          </div>
        </section>

        <aside className="tutorial-lesson-panel">
          <div className="tutorial-step-counter">
            <span>{stepIndex + 1}</span>
            <em>/ {tutorialScenes.length}</em>
          </div>
          <div className="tutorial-rule-callout">
            <Target size={18} />
            <p>{scene.journal}</p>
          </div>
          {scene.secretNote ? (
            <div className="tutorial-secret-note">
              <strong>{scene.secretNote.title}</strong>
              <span>{scene.secretNote.text}</span>
            </div>
          ) : null}
          <div className="tutorial-controls">
            <button type="button" disabled={isFirst} onClick={() => setStepIndex((index) => Math.max(0, index - 1))}>
              <ArrowLeft size={16} />
              Précédent
            </button>
            {isLast ? (
              <button className="primary wide" type="button" onClick={() => navigate("game")}>
                <Play size={16} />
                Lancer une partie
              </button>
            ) : (
              <button className="primary" type="button" onClick={() => setStepIndex((index) => Math.min(tutorialScenes.length - 1, index + 1))}>
                Suivant
                <ArrowRight size={16} />
              </button>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}
