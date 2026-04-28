import React from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, Menu, Play, RotateCcw, Sparkles, Swords, X } from "lucide-react";
import stitchAmiraSpriteUrl from "./assets/stitch-amira-fullbody-cutout.png";
import stitchSultanSpriteUrl from "./assets/stitch-sultan-fullbody-cutout.png";
import stitchSultanaSpriteUrl from "./assets/stitch-sultana-fullbody-cutout.png";
import { cardArtById, cardBackArt } from "./assets/cards";
import "./styles.css";

const CARD_NAMES = {
  0: "Espionne",
  1: "Garde",
  2: "Qadi",
  3: "Émir",
  4: "Hajib",
  5: "Wali",
  6: "Vizir",
  7: "Sultan",
  8: "Sultane",
  9: "Amira",
};

const JOURNAL_NAME_REPLACEMENTS = [
  ["Pretre", "Qadi"],
  ["Baron", "Émir"],
  ["Servante", "Hajib"],
  ["Prince", "Wali"],
  ["Chancelier", "Vizir"],
  ["Roi", "Sultan"],
  ["Comtesse", "Sultane"],
  ["Princesse", "Amira"],
];

const API_BASE = import.meta.env.VITE_API_BASE || "";
const AI_POLICY_OPTIONS = [
  { id: "champion_cbp", label: "Champion CBP" },
  { id: "step3_fast", label: "Step3 seul" },
  { id: "step2_retarget", label: "Step2" },
  { id: "heuristic_fair", label: "Heuristique" },
  { id: "random", label: "Random" },
];
const DEFAULT_AI_POLICIES = {
  player_1: "champion_cbp",
  player_2: "champion_cbp",
  player_3: "champion_cbp",
};
const AI_POLICY_CHARACTERS = [
  { id: "player_1", name: "La Sultane" },
  { id: "player_2", name: "Le Sultan" },
  { id: "player_3", name: "L'Amira" },
];
const STITCH_ROYAL_DECOR_URL =
  "https://lh3.googleusercontent.com/aida/ADBb0ugzGhPx0fwWk5nHrVacmS07mwMIh-MqLosJW5ukDp9PFUIqSkLQMM6osejhMswJE0tNsImcsE5ENWvnELSyRXx-T0eW_w1ShNVedrNmzirTNFqqdDqWDG5yjsPV7KF-FSoCq6p1UfCfBD-18lyPYRS9V8xaG_q8nbs2W0BHEMQoMUWLZrGPqk90nl-4wJvX5lKAlfLGPuB-ulYgYc7XjHJOxTGLXkXCfj6Eqm-owGtox6rnmBKkjWg60Xo";
const STITCH_MENU_DOORS_URL =
  "https://lh3.googleusercontent.com/aida/ADBb0uhNDvdkcAW04e-yxilbMdIeBja_L93neHPY6L2RZR9Zvaj4OlA3uUR5wS3-5yz3b7iCfQHoASud3G_PDhGNQ3SkhJMStaoacrvznPU0T--3Y_yzz0L2gcY34cgbQkxjt2hTkaSn8Wseghu6r_TRIljRlzRnaZcEA2QvjzvYNlYFozAZBqVzEJdyZj2nYDEKZrFqPPk5U_H8ziGfVza7zLSmkO_4MENTo5k06vQ0Ljy4RhVsRqDq49iGhvaG";
const STITCH_LIBRARY_QADI_URL =
  "https://lh3.googleusercontent.com/aida/ADBb0ug-PQLDIPJv0orPOF_nL3NE5wDI2X4sF_WOQkt84Fsj8S3M84GXy44rgPww_KAbKx7gMBVQJxqSuMmTdZfGNC43PqB415MB6gUYlfMz-liNULFIGwdu7dG77hUYp3lt53VabN--yPsw_l7EzSdRX5X3O1KW2_00D6bp67VKUtSq3WAsllbqTK2khIQHM1FkQAeeskSoeYriKxjGNGDm4ou4oFXkFbL44M07b4-yPfxSjdjB_m4fJHuofw8";
const PALACE_DOORS_INTRO_VIDEO = "/palace_zoom_intro.mp4";
const PALACE_DOORS_INTRO_RATE = 1.12;
const PALACE_OST_AUDIO = "/palace_ost.mp3";
const PALACE_OST_MAX_VOLUME = 0.22;
const PALACE_OST_FADE_MS = 1600;
const DEFAULT_SOUND_SETTINGS = {
  enabled: true,
  volume: 0.06,
};

let palaceOstAudio = null;
let palaceOstFadeTimer = null;

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Erreur API ${response.status}`);
  }
  return response.json();
}

function getPalaceOstAudio() {
  if (!window.__palaceOstAudio) {
    window.__palaceOstAudio = new Audio(PALACE_OST_AUDIO);
    window.__palaceOstAudio.loop = true;
    window.__palaceOstAudio.volume = DEFAULT_SOUND_SETTINGS.volume;
    window.__palaceOstAudio.preload = "auto";
  }
  palaceOstAudio = window.__palaceOstAudio;
  if (!palaceOstAudio.src.includes(PALACE_OST_AUDIO)) {
    palaceOstAudio.pause();
    palaceOstAudio = new Audio(PALACE_OST_AUDIO);
    palaceOstAudio.loop = true;
    palaceOstAudio.volume = DEFAULT_SOUND_SETTINGS.volume;
    palaceOstAudio.preload = "auto";
    window.__palaceOstAudio = palaceOstAudio;
  }
  return palaceOstAudio;
}

function clampVolume(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return DEFAULT_SOUND_SETTINGS.volume;
  return Math.min(PALACE_OST_MAX_VOLUME, Math.max(0, numberValue));
}

function readSoundSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("palaceSoundSettings") || "{}");
    return {
      enabled: stored.enabled !== false,
      volume: clampVolume(stored.volume ?? DEFAULT_SOUND_SETTINGS.volume),
    };
  } catch (_error) {
    return { ...DEFAULT_SOUND_SETTINGS };
  }
}

function applyPalaceOstSettings(settings = readSoundSettings()) {
  const audio = getPalaceOstAudio();
  audio.volume = settings.enabled ? clampVolume(settings.volume) : 0;
  if (!settings.enabled) {
    audio.pause();
  }
  return audio;
}

function startPalaceOst() {
  const settings = readSoundSettings();
  if (!settings.enabled) return;
  const audio = getPalaceOstAudio();
  const targetVolume = clampVolume(settings.volume);
  window.clearInterval(palaceOstFadeTimer);
  audio.volume = Math.min(audio.volume || 0, targetVolume);
  audio.play().catch(() => {
    // Les navigateurs bloquent parfois l'audio hors geste utilisateur.
  });
  const startedAt = performance.now();
  palaceOstFadeTimer = window.setInterval(() => {
    const progress = Math.min(1, (performance.now() - startedAt) / PALACE_OST_FADE_MS);
    audio.volume = targetVolume * progress;
    if (progress >= 1) {
      window.clearInterval(palaceOstFadeTimer);
    }
  }, 80);
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    window.clearInterval(palaceOstFadeTimer);
  });
}

function readCurrentView() {
  return new URLSearchParams(window.location.search).get("view");
}

function pushView(view) {
  const url = new URL(window.location.href);
  if (view) {
    url.searchParams.set("view", view);
  } else {
    url.searchParams.delete("view");
  }
  window.history.pushState({}, "", url);
  window.dispatchEvent(new Event("palace:viewchange"));
}

function readAiPolicySettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("palaceAiPolicies") || "{}");
    const validIds = new Set(AI_POLICY_OPTIONS.map((policy) => policy.id));
    return Object.fromEntries(
      AI_POLICY_CHARACTERS.map((character) => {
        const storedId = stored?.[character.id];
        return [character.id, validIds.has(storedId) ? storedId : DEFAULT_AI_POLICIES[character.id]];
      }),
    );
  } catch (_error) {
    return { ...DEFAULT_AI_POLICIES };
  }
}

function cardLabel(card) {
  if (card === null || card === undefined) return "?";
  return `${CARD_NAMES[card] || "?"} (${card})`;
}

function palaceText(text = "") {
  let output = text;
  for (const [from, to] of JOURNAL_NAME_REPLACEMENTS) {
    output = output.replace(new RegExp(`\\b${from}\\b`, "g"), to);
  }
  return output.trimEnd().replace(/\.+$/g, "");
}

function isHiddenJournalLine(text = "") {
  const cardNames = [
    "Espionne",
    "Garde",
    "Pretre",
    "Qadi",
    "Baron",
    "Emir",
    "Servante",
    "Hajib",
    "Prince",
    "Wali",
    "Chancelier",
    "Vizir",
    "Roi",
    "Sultan",
    "Comtesse",
    "Sultane",
    "Princesse",
    "Amira",
  ];
  return new RegExp(`^.+ garde (${cardNames.join("|")}) \\([0-9]\\)\\.?$`).test(text.trim());
}

function PixelCard({ card, hidden = false }) {
  return (
    <span className={`pixel-card ${hidden ? "hidden" : ""}`}>
      {hidden ? "" : card}
    </span>
  );
}

function RulesPanel({ open, rules, onClose }) {
  if (!open) return null;
  return (
    <aside className="rules-panel" aria-label="Règles du jeu">
      <button className="icon-button close-button" type="button" onClick={onClose} aria-label="Fermer">
        <X size={18} />
      </button>
      <div className="panel-kicker">Menu</div>
      <h2>Règles de Love Letter</h2>
      <div className="rules-list">
        {(rules || []).map((rule, index) => (
          <p key={index}>{rule}</p>
        ))}
      </div>
    </aside>
  );
}

function StartScreen({ onStart, loading, rules }) {
  const [name, setName] = React.useState("Hafsa");
  const [rulesOpen, setRulesOpen] = React.useState(false);

  return (
    <main className="start-screen">
      <button className="top-menu-button" type="button" onClick={() => setRulesOpen(true)}>
        <Menu size={18} />
        Menu
      </button>
      <section className="start-card">
        <div className="panel-kicker">Champion CBP</div>
        <h1>Le Palais du Sultan</h1>
        <p>
          Affronte trois agents dans un palais de conte en 2 points. Une victoire de manche vaut un point,
          et l'Espionne peut voler le second.
        </p>
        <label className="name-field">
          Ton nom
          <input value={name} onChange={(event) => setName(event.target.value)} maxLength={18} />
        </label>
        <div className="start-actions">
          <button className="primary-button" type="button" onClick={() => onStart(name)} disabled={loading}>
            <Play size={18} />
            {loading ? "Chargement..." : "Commencer"}
          </button>
          <button className="secondary-button" type="button" onClick={() => setRulesOpen(true)}>
            <BookOpen size={18} />
            Règles
          </button>
        </div>
      </section>
      <div className="start-characters" aria-hidden="true">
        <div className="tiny-sprite s1" />
        <div className="tiny-sprite s2" />
        <div className="tiny-sprite s3" />
      </div>
      <RulesPanel open={rulesOpen} rules={rules} onClose={() => setRulesOpen(false)} />
    </main>
  );
}

function PlayerSprite({ player, active, position }) {
  const played = player.played.slice(-4);
  return (
    <div className={`seat ${position} ${active ? "active" : ""} ${player.alive ? "" : "dead"}`}>
      <div className="turn-cursor">▼</div>
      {player.speech?.text ? (
        <div className={`speech ${player.speech.tone || "normal"}`}>{player.speech.text}</div>
      ) : null}
      <div className="sprite" aria-hidden="true">
        <span className="sprite-shadow" />
        <span className="sprite-body" />
        <span className="sprite-head" />
        <span className="sprite-hair" />
        <span className="sprite-eye left" />
        <span className="sprite-eye right" />
        <span className="sprite-mouth" />
      </div>
      <div className="nameplate">{player.name}</div>
      <div className="status-dots" aria-label="Statut">
        <span className={`dot ${player.alive ? "alive" : "dead"}`} title={player.alive ? "En jeu" : "Eliminee"} />
        {player.protected ? <span className="dot protected" title="Protegee" /> : null}
        {Array.from({ length: player.score }).map((_, index) => (
          <span key={index} className="dot point" title="Point" />
        ))}
      </div>
      <div className="mini-cards">
        {player.is_human
          ? player.hand.map((card, index) => <PixelCard key={`${card}-${index}`} card={card} />)
          : Array.from({ length: player.hand_count }).map((_, index) => <PixelCard key={index} hidden />)}
      </div>
      <div className="played-row">
        {played.map((card, index) => (
          <PixelCard key={`${card}-${index}`} card={card} />
        ))}
      </div>
    </div>
  );
}

function GameBoard({ state }) {
  const playerMap = Object.fromEntries(state.players.map((player) => [player.id, player]));
  const current = state.current_agent;
  return (
    <section className="game-board" aria-label="Plateau Love Letter">
      <div className="palace-arch left" aria-hidden="true" />
      <div className="palace-arch right" aria-hidden="true" />
      <div className="palace-column left" aria-hidden="true" />
      <div className="palace-column right" aria-hidden="true" />
      <div className="palace-lantern left" aria-hidden="true" />
      <div className="palace-lantern right" aria-hidden="true" />
      <div className="palace-palm left" aria-hidden="true" />
      <div className="palace-palm right" aria-hidden="true" />
      <div className="palace-cushion left" aria-hidden="true" />
      <div className="palace-cushion right" aria-hidden="true" />
      <div className="top-piles">
        <div className="table-pile deck-pile" aria-label={`Pioche ${state.deck_size} cartes`}>
          <span className="table-pile-label">Pioche</span>
          <span className="table-card table-card-back">
            <strong>{state.deck_size}</strong>
          </span>
        </div>
        <div className="table-pile discard-pile" aria-label="Défausse">
          <span className="table-pile-label">Défausse</span>
          <span className={`table-card table-card-face ${state.discard_top === null ? "empty" : ""}`}>
            {state.discard_top === null ? "vide" : state.discard_top}
          </span>
        </div>
      </div>
      <PlayerSprite player={playerMap.player_1} active={current === "player_1"} position="p1" />
      <PlayerSprite player={playerMap.player_2} active={current === "player_2"} position="p2" />
      <PlayerSprite player={playerMap.player_3} active={current === "player_3"} position="p3" />
      <PlayerSprite player={playerMap.player_0} active={current === "player_0"} position="p0" />
    </section>
  );
}

function ActionControls({ state, onPlay, onAiStep, onNextRound, onNewGame, busy }) {
  const [selectedCard, setSelectedCard] = React.useState(null);
  const [selectedTarget, setSelectedTarget] = React.useState(null);
  const actions = state.valid_actions || [];

  React.useEffect(() => {
    setSelectedCard(null);
    setSelectedTarget(null);
  }, [state.current_agent, state.round_index]);

  if (state.match_over) {
    const winners = state.players.filter((player) => player.score >= state.target_points);
    return (
      <section className="control-panel">
        <h2>Partie terminee</h2>
        <p>{winners.map((player) => player.name).join(", ")} gagne.</p>
        <button className="primary-button" type="button" onClick={onNewGame}>
          <RotateCcw size={18} />
          Nouvelle partie
        </button>
      </section>
    );
  }

  if (state.round_over) {
    return (
      <section className="control-panel">
        <h2>Manche terminee</h2>
        <button className="primary-button" type="button" onClick={onNextRound} disabled={busy}>
          <Sparkles size={18} />
          Manche suivante
        </button>
      </section>
    );
  }

  if (!state.can_human_act) {
    return (
      <section className="control-panel">
        <h2>Tour IA</h2>
        <p>{state.current_name} reflechit.</p>
        <button className="primary-button" type="button" onClick={onAiStep} disabled={busy}>
          <Swords size={18} />
          Faire jouer {state.current_name}
        </button>
      </section>
    );
  }

  const chancellorActions = actions.filter((action) => action.action >= 900);
  if (chancellorActions.length) {
    return (
      <section className="control-panel">
        <h2>Vizir</h2>
        <div className="hand-preview">
          {state.chancellor_pool.map((card, index) => (
            <span key={index}>{cardLabel(card)}</span>
          ))}
        </div>
        <div className="action-list">
          {chancellorActions.map((action) => (
            <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
              {palaceText(action.label)}
            </button>
          ))}
        </div>
      </section>
    );
  }

  const cards = [...new Map(actions.map((action) => [action.card, action])).values()];
  const cardActions = selectedCard === null ? [] : actions.filter((action) => action.card === selectedCard);
  const needsTarget = selectedCard !== null && [1, 2, 3, 5, 7].includes(selectedCard);
  const targetActions = selectedTarget === null
    ? []
    : cardActions.filter((action) => action.target === selectedTarget);

  return (
    <section className="control-panel">
      <h2>A toi</h2>
      {state.private_notes?.length ? (
        <div className="private-note">
          {state.private_notes[state.private_notes.length - 1].text}
        </div>
      ) : null}
      {selectedCard === null ? (
        <div className="action-list">
          {cards.map((action) => (
            <button key={action.card} type="button" onClick={() => setSelectedCard(action.card)}>
              <span className="button-card">{action.card}</span>
              {action.card_name}
            </button>
          ))}
        </div>
      ) : needsTarget && selectedTarget === null ? (
        <>
          <button className="back-button" type="button" onClick={() => setSelectedCard(null)}>
            Changer de carte
          </button>
          <h3>{cardLabel(selectedCard)} - cible</h3>
          <div className="action-list">
            {[...new Map(cardActions.map((action) => [action.target, action])).values()].map((action) => (
              <button key={action.target || action.action} type="button" onClick={() => setSelectedTarget(action.target)}>
                {action.target_name || "Sans cible"}
              </button>
            ))}
          </div>
        </>
      ) : selectedCard === 1 ? (
        <>
          <button className="back-button" type="button" onClick={() => setSelectedTarget(null)}>
            Changer de cible
          </button>
          <h3>Annonce</h3>
          <div className="action-list guesses">
            {targetActions.map((action) => (
              <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
                {palaceText(action.guess_name)}
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          <button className="back-button" type="button" onClick={() => setSelectedCard(null)}>
            Changer de carte
          </button>
          <div className="action-list">
            {(needsTarget ? targetActions : cardActions).map((action) => (
              <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
                {action.label}
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function Journal({ logs }) {
  return (
    <section className="journal">
      <h2>Journal</h2>
      <div className="journal-lines">
        {(logs || []).slice(-18).map((log, index) => (
          <p key={`${log.ts}-${index}`} className={log.tone || "normal"}>
            {log.text}
          </p>
        ))}
      </div>
    </section>
  );
}

function ScoreBar({ state }) {
  return (
    <section className="score-bar">
      {state.players.map((player) => (
        <div key={player.id}>
          <span>{player.name}</span>
          <strong>
            {player.score}/{state.target_points}
          </strong>
        </div>
      ))}
    </section>
  );
}

function DecorGameCard({ card, hidden = false, className = "", style }) {
  const art = hidden ? cardBackArt : cardArtById[card] || cardBackArt;
  return (
    <img
      src={art.image}
      alt={hidden ? "Carte cachee" : `${CARD_NAMES[card] || art.gameName} (${card})`}
      className={className}
      style={style}
    />
  );
}

function DecorPlayedZone({ player, className = "" }) {
  const played = player?.played || [];
  const lastServantIndex = played.lastIndexOf(4);
  const statusCards = played.filter((card, index) => (
    card === 0 || (card === 4 && player.protected && index === lastServantIndex)
  ));
  if (!statusCards.length) return null;
  const lastCard = statusCards[statusCards.length - 1];
  const humanClass = player?.is_human ? "is-human" : "";

  return (
    <div className={`decor-played-card-zone ${className} ${humanClass}`} aria-label={`Cartes actives devant ${player.name}`}>
      <span className="decor-played-owner">{player.name}</span>
      <div className="decor-played-mini-stack">
        {statusCards.map((card, index) => (
          <DecorGameCard
            key={`${player.id}-${card}-${index}`}
            card={card}
            className="decor-played-card"
            style={{ "--played-index": index }}
          />
        ))}
      </div>
      <span className="decor-played-label">{CARD_NAMES[lastCard] || cardArtById[lastCard]?.gameName} jouée</span>
    </div>
  );
}

function discardCardsForTable(players = [], discardEvents = []) {
  const playerMap = Object.fromEntries(players.map((player) => [player.id, player]));
  const events = discardEvents.length
    ? discardEvents.map((event) => ({
        id: event.id,
        card: event.card,
        playerId: event.owner,
        playedIndex: event.played_index,
        isHuman: event.is_human,
      }))
    : players.flatMap((player) => {
    const played = player.played || [];
    const lastServantIndex = played.lastIndexOf(4);
    return played.flatMap((card, index) => {
      if (card === 0) return [];
      if (card === 4 && player.protected && index === lastServantIndex) return [];
      return [{
        id: `${player.id}-${index}-${card}`,
        card,
        playerId: player.id,
        playedIndex: index,
        isHuman: player.is_human,
      }];
    });
  });

  return events.filter((event) => {
    const player = playerMap[event.playerId];
    if (!player) return true;
    if (event.card === 0) return false;
    if (
      event.card === 4
      && player.protected
      && (player.played || []).lastIndexOf(4) === event.playedIndex
    ) {
      return false;
    }
    return true;
  });
}

function DecorHeldHand({ player, active = false, className = "" }) {
  if (!player || player.is_human || !player.alive) return null;
  const count = Math.max(1, Math.min(player.hand_count || 1, 2));

  return (
    <div className={`decor-held-hand ${className} ${active ? "is-active" : ""}`} aria-label={`Main cachee de ${player.name}`}>
      {Array.from({ length: count }).map((_, index) => (
        <DecorGameCard
          key={`${player.id}-hidden-${index}-${count}`}
          hidden
          className={`decor-held-card held-${index}`}
        />
      ))}
    </div>
  );
}

function DecorDrawFlight({ current, roundIndex, deckSize, roundOver, matchOver }) {
  if (!current || current === "player_0" || roundOver || matchOver) return null;

  return (
    <DecorGameCard
      key={`${current}-${roundIndex}-${deckSize}`}
      hidden
      className={`decor-draw-flight to-${current}`}
    />
  );
}

function DecorPlayerBadge({ player, className = "" }) {
  if (!player) return null;
  return (
    <div className={`decor-player-badge ${className} ${player.alive ? "" : "is-dead"}`}>
      <strong>{player.name}</strong>
      <span>{player.score}/2</span>
      {player.protected ? <em>protégée</em> : null}
    </div>
  );
}

function DecorSpeechBubble({ player, className = "", visible = true }) {
  if (!player?.speech?.text) return null;
  const speechKey = `${player.id}-${player.speech.ts || ""}-${player.speech.text}`;
  const [autoVisible, setAutoVisible] = React.useState(true);

  React.useEffect(() => {
    setAutoVisible(true);
    const timeoutId = window.setTimeout(() => setAutoVisible(false), 3300);
    return () => window.clearTimeout(timeoutId);
  }, [speechKey]);

  if (!visible || !autoVisible) return null;
  const tone = player.speech.tone || "normal";
  const text = palaceText(player.speech.text);
  return (
    <div
      key={speechKey}
      className={`decor-speech-bubble ${className} ${tone}`}
      aria-label={`Dialogue de ${player.name}`}
    >
      {text}
    </div>
  );
}

function DecorJournal({ logs }) {
  const fallbackLogs = [{ ts: "loading", text: "Le palais prepare une nouvelle partie.", tone: "info" }];
  const rawEntries = logs?.length ? logs : fallbackLogs;
  const entries = (rawEntries.length ? rawEntries : fallbackLogs)
    .slice(-24)
    .map((log) => ({ ...log, displayText: palaceText(log.text) }));
  const latest = entries[entries.length - 1];
  const latestKey = latest ? `${latest.ts}-${entries.length}-${latest.displayText}` : "empty";
  const latestText = latest?.displayText || "";
  const [typingKey, setTypingKey] = React.useState(latestKey);
  const [typedText, setTypedText] = React.useState(latestText);
  const journalRef = React.useRef(null);

  React.useEffect(() => {
    if (!latestText) return undefined;
    const fullText = latestText;
    let index = 0;
    let timeoutId;
    setTypingKey(latestKey);
    setTypedText("");

    const tick = () => {
      index += 1;
      setTypedText(fullText.slice(0, index));
      if (index < fullText.length) {
        timeoutId = window.setTimeout(tick, 42);
      }
    };

    timeoutId = window.setTimeout(tick, 160);
    return () => window.clearTimeout(timeoutId);
  }, [latestKey, latestText]);

  React.useEffect(() => {
    const node = journalRef.current;
    if (!node) return;
    window.requestAnimationFrame(() => {
      node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
    });
  }, [entries.length, typedText]);

  return (
    <div className="stitch-decor-note" ref={journalRef}>
      <strong>Journal du Palais</strong>
      {entries.map((log, index) => {
        const key = `${log.ts}-${index}-${log.displayText}`;
        const isLatest = index === entries.length - 1 && typingKey === latestKey;
        return (
          <span key={key} className={`${log.tone || "normal"} ${isLatest ? "typing" : ""}`}>
            {isLatest ? typedText : log.displayText}
          </span>
        );
      })}
    </div>
  );
}

function DecorActionPanel({ state, busy, onPlay, onAiStep, onNextRound, onNewGame }) {
  const [selectedCard, setSelectedCard] = React.useState(null);
  const [selectedTarget, setSelectedTarget] = React.useState(null);

  React.useEffect(() => {
    setSelectedCard(null);
    setSelectedTarget(null);
  }, [state?.current_agent, state?.round_index]);

  if (!state) {
    return (
      <section className="decor-action-panel">
        <h2>Installation</h2>
        <p>Le palais prepare la manche.</p>
      </section>
    );
  }

  if (state.match_over) {
    const winners = state.players.filter((player) => player.score >= state.target_points);
    return (
      <section className="decor-action-panel">
        <h2>Partie terminee</h2>
        <p>{winners.map((player) => player.name).join(", ")} gagne.</p>
        <button type="button" onClick={onNewGame} disabled={busy}>
          Nouvelle partie
        </button>
      </section>
    );
  }

  if (state.round_over) {
    return (
      <section className="decor-action-panel">
        <h2>Manche terminee</h2>
        <button type="button" onClick={onNextRound} disabled={busy}>
          Manche suivante
        </button>
      </section>
    );
  }

  if (!state.can_human_act) {
    return (
      <section className="decor-action-panel">
        <h2>Tour du palais</h2>
        <p>{state.current_name} reflechit.</p>
        <button type="button" onClick={onAiStep} disabled={busy}>
          Faire jouer {state.current_name}
        </button>
      </section>
    );
  }

  const actions = state.valid_actions || [];
  const chancellorActions = actions.filter((action) => action.action >= 900);
  if (chancellorActions.length) {
    return (
      <section className="decor-action-panel">
        <h2>Vizir</h2>
        <div className="decor-chancellor-pool">
          {state.chancellor_pool.map((card, index) => (
            <DecorGameCard key={`${card}-${index}`} card={card} className="decor-choice-card" />
          ))}
        </div>
        <div className="decor-action-list">
          {chancellorActions.map((action) => (
            <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
              {action.label}
            </button>
          ))}
        </div>
      </section>
    );
  }

  const cards = [...new Map(actions.map((action) => [action.card, action])).values()];
  const cardActions = selectedCard === null ? [] : actions.filter((action) => action.card === selectedCard);
  const needsTarget = selectedCard !== null && [1, 2, 3, 5, 7].includes(selectedCard);
  const targetActions = selectedTarget === null
    ? []
    : cardActions.filter((action) => action.target === selectedTarget);

  return (
    <section className="decor-action-panel">
      <h2>A toi</h2>
      {state.private_notes?.length ? (
        <p className="decor-private-note">{state.private_notes[state.private_notes.length - 1].text}</p>
      ) : null}
      {selectedCard === null ? (
        <div className="decor-action-list decor-card-action-list">
          {cards.map((action) => (
            <button key={action.card} type="button" onClick={() => setSelectedCard(action.card)}>
              <DecorGameCard card={action.card} className="decor-action-card-thumb" />
              <span>{CARD_NAMES[action.card] || action.card_name}</span>
            </button>
          ))}
        </div>
      ) : needsTarget && selectedTarget === null ? (
        <>
          <button className="decor-back-button" type="button" onClick={() => setSelectedCard(null)}>
            Changer de carte
          </button>
          <h3>{cardLabel(selectedCard)} - cible</h3>
          <div className="decor-action-list">
            {[...new Map(cardActions.map((action) => [action.target, action])).values()].map((action) => (
              <button key={action.target || action.action} type="button" onClick={() => setSelectedTarget(action.target)}>
                {action.target_name || "Sans cible"}
              </button>
            ))}
          </div>
        </>
      ) : selectedCard === 1 ? (
        <>
          <button className="decor-back-button" type="button" onClick={() => setSelectedTarget(null)}>
            Changer de cible
          </button>
          <h3>Annonce</h3>
          <div className="decor-action-list decor-guess-list">
            {targetActions.map((action) => (
              <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
                {action.guess_name}
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          <button className="decor-back-button" type="button" onClick={() => setSelectedCard(null)}>
            Changer de carte
          </button>
          <div className="decor-action-list">
            {(needsTarget ? targetActions : cardActions).map((action) => (
              <button key={action.action} type="button" onClick={() => onPlay(action.action)} disabled={busy}>
                {palaceText(action.label)}
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function VisualNovelPrototype() {
  const scenes = [
    {
      id: "terrace",
      label: "Terrasse",
      speaker: "Mazeer",
      text: "Je la lui rendrai moi-meme.",
      actor: "mazeer",
    },
    {
      id: "courtyard",
      label: "Cour",
      speaker: "",
      text: "Je ne la vois plus. Je traverse la cour et file vers la porte.",
      actor: "none",
    },
    {
      id: "chamber",
      label: "Chambre",
      speaker: "Tiziri",
      text: "Entre.",
      actor: "tiziri",
    },
  ];
  const [sceneIndex, setSceneIndex] = React.useState(0);
  const scene = scenes[sceneIndex];

  return (
    <main className="vn-screen">
      <a className="vn-back-link" href="/" aria-label="Retour au jeu">
        Retour au palais jouable
      </a>
      <section className="vn-frame vn-refined" aria-label="Prototype visual novel pixel art">
        <div className={`vn-main-scene vn-${scene.id}`}>
          <div className="vn-border top" />
          <div className="vn-border bottom" />
          <div className="vn-pillar left" />
          <div className="vn-pillar right" />
          <div className="vn-lantern floor left" />
          <div className="vn-lantern floor right" />
          <div className="vn-scene-architecture">
            <span className="vn-arch a1" />
            <span className="vn-arch a2" />
            <span className="vn-arch a3" />
            <span className="vn-arch a4" />
            <span className="vn-door" />
            <span className="vn-water" />
          </div>
          <div className="vn-sunbeams" />
          <div className="vn-lattice-window left" />
          <div className="vn-lattice-window right" />
          <div className="vn-palms">
            <span className="vn-palm p1" />
            <span className="vn-palm p2" />
          </div>
          {scene.actor !== "none" ? (
            <div className={`vn-person ${scene.actor}`} aria-label={scene.speaker}>
              <span className="vn-person-shadow" />
              <span className="vn-robe" />
              <span className="vn-sash" />
              <span className="vn-skin torso" />
              <span className="vn-skin neck" />
              <span className="vn-head" />
              <span className="vn-eye left" />
              <span className="vn-eye right" />
              <span className="vn-nose" />
              <span className="vn-mouth" />
              <span className="vn-hair" />
              <span className="vn-veil" />
              <span className="vn-jewel" />
              <span className="vn-arm left" />
              <span className="vn-arm right" />
              <span className="vn-staff" />
            </div>
          ) : null}
          <div className="vn-dialogue">
            {scene.speaker ? <strong>{scene.speaker}</strong> : null}
            <p>{scene.text}</p>
          </div>
        </div>
        <aside className="vn-side-panel">
          <div className="vn-zellij" />
          <div className="vn-hanging-ornaments">
            <span />
            <span />
            <span />
          </div>
          <div className="vn-portrait">
            <div className="vn-portrait-veil" />
            <div className="vn-portrait-head" />
            <div className="vn-portrait-eye left" />
            <div className="vn-portrait-eye right" />
            <div className="vn-portrait-mouth" />
          </div>
          <div className="vn-side-card">
            <span>Prototype</span>
            <strong>{scene.label}</strong>
          </div>
          <nav className="vn-scene-nav" aria-label="Changer de scene">
            {scenes.map((item, index) => (
              <button
                key={item.id}
                type="button"
                className={index === sceneIndex ? "active" : ""}
                onClick={() => setSceneIndex(index)}
              >
                {index + 1}
              </button>
            ))}
          </nav>
        </aside>
      </section>
    </main>
  );
}

function StitchRoyalDecorPreview({ rules = [] }) {
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [state, setState] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const busyRef = React.useRef(false);

  async function run(action) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError("");
    try {
      setState(await action());
    } catch (err) {
      setError(err.message || "Erreur pendant la partie");
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function startDecorGame() {
    await run(() => api("/api/games", {
      method: "POST",
      body: JSON.stringify({
        human_name: localStorage.getItem("palacePlayerName") || "Hafsa",
        ai_policies: readAiPolicySettings(),
      }),
    }));
  }

  React.useEffect(() => {
    startDecorGame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const playerMap = Object.fromEntries((state?.players || []).map((player) => [player.id, player]));
  const human = playerMap.player_0;
  const allPlayed = discardCardsForTable(state?.players || [], state?.discard_events || []);
  const discardStack = allPlayed.length ? allPlayed.slice(-9) : [];
  const latestDiscardId = discardStack.length ? discardStack[discardStack.length - 1].id : null;
  const drawStack = Array.from({ length: Math.min(Math.max(state?.deck_size || 0, 0), 10) });
  const current = state?.current_agent;

  return (
    <main className="stitch-decor-screen">
      <section className="stitch-decor-frame" aria-label="Decor principal Stitch">
        <button
          className="stitch-scene-menu-button"
          type="button"
          onClick={() => setMenuOpen((open) => !open)}
          aria-expanded={menuOpen}
          aria-label="Ouvrir le menu de la scene"
        >
          <Menu size={18} />
          Menu
        </button>
        {menuOpen && (
          <nav className="stitch-scene-menu" aria-label="Menu de la scene">
            <button type="button" onClick={startDecorGame} disabled={busy}>Nouvelle partie</button>
            <a href="/">Menu principal</a>
            <a href="?view=cards-preview">Cartes</a>
            <a href="?view=rules">Règles</a>
            <a href="?view=settings">Paramètres</a>
          </nav>
        )}
        <img className="stitch-decor-backdrop" src={STITCH_ROYAL_DECOR_URL} alt="Salon du Palais - decor pur" />
        <img
          className={`decor-stitch-character decor-sultana-stitch-art ${current === "player_1" ? "is-active" : ""} ${playerMap.player_1?.alive === false ? "is-dead" : ""}`}
          src={stitchSultanaSpriteUrl}
          alt="La Sultane, sprite pixel art Stitch"
          aria-label="La Sultane, illustration Stitch"
        />
        <img
          className={`decor-stitch-character decor-sultan-stitch-art ${current === "player_2" ? "is-active" : ""} ${playerMap.player_2?.alive === false ? "is-dead" : ""}`}
          src={stitchSultanSpriteUrl}
          alt="Le Sultan, sprite pixel art Stitch"
          aria-label="Le Sultan, illustration Stitch"
        />
        <img
          className={`decor-stitch-character decor-amira-stitch-art ${current === "player_3" ? "is-active" : ""} ${playerMap.player_3?.alive === false ? "is-dead" : ""}`}
          src={stitchAmiraSpriteUrl}
          alt="L'Amira, sprite pixel art Stitch"
          aria-label="L'Amira, illustration Stitch"
        />
        <DecorPlayerBadge player={playerMap.player_1} className="decor-badge-sultana" />
        <DecorPlayerBadge player={playerMap.player_2} className="decor-badge-sultan" />
        <DecorPlayerBadge player={playerMap.player_3} className="decor-badge-amira" />
        <DecorSpeechBubble player={playerMap.player_1} visible={state?.last_speaker === "player_1"} className="decor-speech-sultana" />
        <DecorSpeechBubble player={playerMap.player_2} visible={state?.last_speaker === "player_2"} className="decor-speech-sultan" />
        <DecorSpeechBubble player={playerMap.player_3} visible={state?.last_speaker === "player_3"} className="decor-speech-amira" />
        <DecorHeldHand player={playerMap.player_1} active={current === "player_1"} className="decor-held-sultana" />
        <DecorHeldHand player={playerMap.player_2} active={current === "player_2"} className="decor-held-sultan" />
        <DecorHeldHand player={playerMap.player_3} active={current === "player_3"} className="decor-held-amira" />
        <div className="decor-palace-table" aria-hidden="true">
          <span className="decor-palace-table-side left" />
          <span className="decor-palace-table-side right" />
          <div className="decor-palace-table-inlay" />
        </div>
        <div className="decor-table-state" aria-label="Etat simule du centre de table">
          <div className="decor-card-pile decor-draw-pile" aria-label="Pioche, 10 cartes">
            <div className="decor-card-stack">
              {drawStack.map((_, index) => (
                <img
                  key={`draw-${index}`}
                  src={cardBackArt.image}
                  alt=""
                  aria-hidden="true"
                  className="decor-pile-card decor-pile-card-back"
                  style={{ "--pile-index": index }}
                />
              ))}
              {!drawStack.length ? <span className="decor-empty-pile">0</span> : null}
            </div>
            <span className="decor-pile-label">Pioche {state?.deck_size ?? "..."}</span>
          </div>
          <div className="decor-card-pile decor-discard-pile" aria-label="Défausse simulée">
            <div className="decor-card-stack">
              {discardStack.map((entry, index) => (
                <img
                  key={entry.id}
                  src={cardArtById[entry.card].image}
                  alt=""
                  aria-hidden="true"
                  className={`decor-pile-card decor-pile-card-face from-${entry.playerId} ${entry.id === latestDiscardId ? "is-new-discard" : ""} ${entry.isHuman ? "is-human-discard" : ""}`}
                  style={{ "--pile-index": index }}
                />
              ))}
              {!discardStack.length ? <span className="decor-empty-pile">vide</span> : null}
            </div>
            <span className="decor-pile-label">Défausse</span>
          </div>
        </div>
        <DecorDrawFlight
          current={current}
          roundIndex={state?.round_index}
          deckSize={state?.deck_size}
          roundOver={state?.round_over}
          matchOver={state?.match_over}
        />
        <DecorPlayedZone player={playerMap.player_1} className="decor-sultana-played-zone" />
        <DecorPlayedZone player={playerMap.player_2} className="decor-sultan-played-zone" />
        <DecorPlayedZone player={playerMap.player_3} className="decor-amira-played-zone" />
        <DecorPlayedZone player={human} className="decor-human-played-zone" />
        <div className="decor-human-hand" aria-label="Tes cartes">
          {(human?.hand || []).map((card, index) => (
            <DecorGameCard
              key={`${card}-${index}`}
              card={card}
              className={`decor-human-card hand-${index}`}
            />
          ))}
        </div>
        {human ? <DecorPlayerBadge player={human} className="decor-badge-human" /> : null}
        <DecorJournal logs={state?.logs} />
        <DecorActionPanel
          state={state}
          busy={busy}
          onPlay={(action) => run(() => api(`/api/games/${state.game_id}/play`, {
            method: "POST",
            body: JSON.stringify({ action }),
          }))}
          onAiStep={() => run(() => api(`/api/games/${state.game_id}/ai-step`, { method: "POST" }))}
          onNextRound={() => run(() => api(`/api/games/${state.game_id}/next-round`, { method: "POST" }))}
          onNewGame={startDecorGame}
        />
        {error ? <div className="error-toast decor-error-toast">{error}</div> : null}
      </section>
    </main>
  );
}

function MainMenu({ onNavigate }) {
  const [stage, setStage] = React.useState("menu");
  const [dialogueIndex, setDialogueIndex] = React.useState(0);
  const [playerName, setPlayerName] = React.useState(
    () => localStorage.getItem("palacePlayerName") || "",
  );
  const cleanPlayerName = playerName.trim() || "Hafsa";
  const qadiDialogue = [
    {
      title: "Le Qadi",
      text: `Bienvenue, ${cleanPlayerName}. Le palais connaît déjà ton nom.`,
    },
    {
      title: "Le Qadi",
      text: "Tu entres ici sous les lanternes des Mille et Une Nuits, là où le Sultan garde sa cour et ses secrets.",
    },
    {
      title: "Le Qadi",
      text: "Sa fille, l'Amira, refuse chaque prétendant. Aucun poème, aucune fortune, aucune promesse ne l'a fait changer d'avis.",
    },
    {
      title: "Le Qadi",
      text: "Alors le Sultan a donné une épreuve: battre sa famille à leur jeu préféré, Love Letter.",
    },
    {
      title: "Le Qadi",
      text: `Tu es le 212e candidat, ${cleanPlayerName}. J'espère que tu es prêt à les affronter.`,
    },
    {
      title: "Le Qadi",
      text: "Moi-même, je ne les ai jamais battus. Le Sultan calcule tout, la Sultane voit tout, et l'Amira pardonne rarement.",
    },
    {
      title: "Le Qadi",
      text: "Souviens-toi: ici, une carte jouée trop tôt peut te condamner, et une carte gardée trop longtemps peut te trahir.",
    },
    {
      title: "Le Qadi",
      text: "Avance maintenant. La table est prête. Que ton nom soit plus qu'une ligne de plus dans mon registre.",
    },
  ];
  const activeDialogue = qadiDialogue[Math.min(dialogueIndex, qadiDialogue.length - 1)];
  const isFinalDialogue = dialogueIndex >= qadiDialogue.length - 1;

  React.useEffect(() => {
    if (stage !== "black") return undefined;
    const timeout = window.setTimeout(() => {
      setStage("library");
    }, 950);
    return () => window.clearTimeout(timeout);
  }, [stage]);

  React.useEffect(() => {
    if (stage === "library") {
      setDialogueIndex(0);
    }
  }, [stage]);

  function launchIntro(event) {
    event.preventDefault();
    const cleanName = playerName.trim() || "Hafsa";
    localStorage.setItem("palacePlayerName", cleanName);
    setPlayerName(cleanName);
    startPalaceOst();
    setStage("video");
  }

  function enterGame() {
    if (onNavigate) {
      onNavigate("game");
      return;
    }
    pushView("game");
  }

  function beginDuel() {
    setStage("duel");
    window.setTimeout(enterGame, 1450);
  }

  return (
    <main
      className={`main-menu-screen ${stage === "video" || stage === "black" ? "is-intro-running" : ""}`}
      style={{ "--main-menu-bg": `url("${STITCH_MENU_DOORS_URL}")` }}
    >
      {stage === "video" ? (
        <section className="main-menu-intro-video" aria-label="Ouverture du palais">
          <video
            src={PALACE_DOORS_INTRO_VIDEO}
            autoPlay
            muted
            playsInline
            preload="auto"
            onLoadedMetadata={(event) => {
              event.currentTarget.playbackRate = PALACE_DOORS_INTRO_RATE;
            }}
            onEnded={() => {
              window.setTimeout(() => setStage("black"), 260);
            }}
            onError={() => setStage("black")}
          />
        </section>
      ) : null}
      {stage === "black" ? (
        <section className="main-menu-black-transition" aria-label="Entrée dans le palais">
          <span>Le palais vous ouvre ses portes</span>
        </section>
      ) : null}
      {stage === "library" ? (
        <section
          className="main-menu-library-scene"
          aria-label="Grande archive du palais"
          style={{ "--library-qadi-bg": `url("${STITCH_LIBRARY_QADI_URL}")` }}
        >
          <div className="library-dialogue-panel">
            <div className="panel-kicker">La Grande Archive</div>
            <h1>{activeDialogue.title}</h1>
            <p>{activeDialogue.text}</p>
            <div className="library-dialogue-footer">
              <button
                className="primary"
                type="button"
                onClick={() => {
                  if (isFinalDialogue) {
                    beginDuel();
                    return;
                  }
                  setDialogueIndex((index) => index + 1);
                }}
              >
                <Play size={18} />
                {isFinalDialogue ? "Affronter la cour" : "Suivant"}
              </button>
            </div>
          </div>
        </section>
      ) : null}
      {stage === "duel" ? (
        <section className="main-menu-duel-transition" aria-label="Debut du duel">
          <div className="duel-flash-panel">
            <span>{cleanPlayerName}</span>
            <strong>défie le palais</strong>
            <span>Le duel commence</span>
          </div>
        </section>
      ) : null}
      {stage === "menu" ? (
        <section className="main-menu-panel" aria-label="Menu principal">
          <div className="panel-kicker">Love Letter</div>
          <h1>Le Palais du Sultan</h1>
          <p>
            Entre dans le salon, consulte les cartes ou relis les règles avant de lancer une manche
            contre le champion.
          </p>
          <nav className="main-menu-actions" aria-label="Actions principales">
            <button className="primary" type="button" onClick={() => setStage("name")}>
              <Play size={18} />
              Lancer une partie
            </button>
            <a href="?view=rules">
              <BookOpen size={18} />
              Consulter les règles
            </a>
            <a href="?view=cards-preview">
              <Sparkles size={18} />
              Voir les cartes
            </a>
            <a href="?view=settings">
              <Menu size={18} />
              Changer les paramètres
            </a>
          </nav>
        </section>
      ) : null}
      {stage === "name" ? (
        <section className="main-menu-panel main-menu-name-panel" aria-label="Choisir son prénom">
          <div className="panel-kicker">Entrée au palais</div>
          <h1>Ton prénom</h1>
          <p>Le Sultan veut savoir qui ose s'asseoir a sa table.</p>
          <form className="main-menu-name-form" onSubmit={launchIntro}>
            <label>
              Prenom
              <input
                type="text"
                value={playerName}
                maxLength={18}
                autoFocus
                onChange={(event) => setPlayerName(event.target.value)}
                placeholder="Hafsa"
              />
            </label>
            <div className="main-menu-actions main-menu-form-actions">
              <button className="primary" type="submit">
                <Play size={18} />
                Entrer dans le palais
              </button>
              <button type="button" onClick={() => setStage("menu")}>
                Retour
              </button>
            </div>
          </form>
        </section>
      ) : null}
    </main>
  );
}

function CardsPreview() {
  return (
    <main className="cards-preview-screen">
      <nav className="simple-page-nav" aria-label="Navigation">
        <a href="/">Menu principal</a>
        <a href="?view=rules">Règles</a>
        <a href="?view=settings">Paramètres</a>
      </nav>
      <section className="cards-preview-panel" aria-label="Cartes Love Letter decoupees">
        <h1>Cartes du Palais</h1>
        <div className="cards-preview-grid">
          <figure className="cards-preview-card cards-preview-card-back">
            <img src={cardBackArt.image} alt={cardBackArt.gameName} />
            <figcaption>
              <strong>{cardBackArt.gameName}</strong>
              <span>{cardBackArt.artName}</span>
            </figcaption>
          </figure>
          {Object.entries(cardArtById).map(([id, card]) => (
            <figure className="cards-preview-card" key={id}>
              <img src={card.image} alt={`${card.gameName} - ${card.artName}`} />
              <figcaption>
                <strong>{id} - {card.gameName}</strong>
                <span>{card.artName}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </main>
  );
}

function RulesPage({ rules = [] }) {
  return (
    <main className="rules-page-screen">
      <nav className="simple-page-nav" aria-label="Navigation">
        <a href="/">Menu principal</a>
        <a href="?view=cards-preview">Cartes</a>
        <a href="?view=settings">Paramètres</a>
      </nav>
      <section className="rules-page-panel" aria-label="Règles de Love Letter">
        <div className="rules-scroll-roll rules-scroll-roll-top" aria-hidden="true" />
        <div className="rules-scroll-content">
          <div className="panel-kicker">Le Palais du Sultan</div>
          <h1>Règles du jeu</h1>
          <div className="rules-page-list">
            {(rules.length ? rules : ["Chargement des règles..."]).map((rule, index) => (
              <p key={index}>{rule}</p>
            ))}
          </div>
        </div>
        <div className="rules-scroll-roll rules-scroll-roll-bottom" aria-hidden="true" />
      </section>
    </main>
  );
}

function SettingsPage({ onNavigate }) {
  const [playerName, setPlayerName] = React.useState(
    () => localStorage.getItem("palacePlayerName") || "Hafsa",
  );
  const [aiPolicies, setAiPolicies] = React.useState(readAiPolicySettings);
  const [soundSettings, setSoundSettings] = React.useState(readSoundSettings);
  const [saved, setSaved] = React.useState(false);

  function persistSettings() {
    const cleanName = playerName.trim() || "Hafsa";
    localStorage.setItem("palacePlayerName", cleanName);
    localStorage.setItem("palaceAiPolicies", JSON.stringify(aiPolicies));
    localStorage.setItem("palaceSoundSettings", JSON.stringify(soundSettings));
    setPlayerName(cleanName);
    applyPalaceOstSettings(soundSettings);
  }

  function saveSettings(event) {
    event.preventDefault();
    persistSettings();
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1800);
  }

  function saveAndLaunch() {
    persistSettings();
    if (onNavigate) {
      onNavigate("game");
      return;
    }
    pushView("game");
  }

  function updateAiPolicy(agentId, policyId) {
    setAiPolicies((current) => ({ ...current, [agentId]: policyId }));
  }

  function updateSoundSettings(patch) {
    setSoundSettings((current) => {
      const next = { ...current, ...patch };
      const audioWasLoaded = Boolean(palaceOstAudio);
      const audio = applyPalaceOstSettings(next);
      if (next.enabled && audioWasLoaded) {
        audio.play().catch(() => {});
      }
      return next;
    });
  }

  return (
    <main className="settings-page-screen">
      <nav className="simple-page-nav" aria-label="Navigation">
        <a href="/">Menu principal</a>
        <a href="?view=game">Jouer</a>
        <a href="?view=rules">Règles</a>
        <a href="?view=cards-preview">Cartes</a>
      </nav>
      <section className="settings-page-panel" aria-label="Paramètres">
        <div className="panel-kicker">Le Palais du Sultan</div>
        <h1>Paramètres</h1>
        <form className="settings-form" onSubmit={saveSettings}>
          <label>
            Nom du joueur
            <input
              type="text"
              value={playerName}
              maxLength={18}
              onChange={(event) => setPlayerName(event.target.value)}
            />
          </label>
          <p>La prochaine partie utilisera ce nom dans le journal et autour de la table.</p>
          <fieldset className="settings-policy-grid">
            <legend>Modèles des adversaires</legend>
            {AI_POLICY_CHARACTERS.map((character) => (
              <label key={character.id} className="settings-select-label">
                {character.name}
                <select
                  value={aiPolicies[character.id] || DEFAULT_AI_POLICIES[character.id]}
                  onChange={(event) => updateAiPolicy(character.id, event.target.value)}
                >
                  {AI_POLICY_OPTIONS.map((policy) => (
                    <option key={policy.id} value={policy.id}>
                      {policy.label}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </fieldset>
          <p>
            Chaque adversaire peut avoir son propre cerveau. Une nouvelle partie recharge les modèles choisis ici.
          </p>
          <fieldset className="settings-sound-grid">
            <legend>Ambiance sonore</legend>
            <label className="settings-toggle-label">
              <input
                type="checkbox"
                checked={soundSettings.enabled}
                onChange={(event) => updateSoundSettings({ enabled: event.target.checked })}
              />
              Musique activée
            </label>
            <label className="settings-volume-label">
              Volume
              <input
                type="range"
                min="0"
                max={PALACE_OST_MAX_VOLUME}
                step="0.01"
                value={soundSettings.volume}
                onChange={(event) => updateSoundSettings({ volume: clampVolume(event.target.value) })}
                disabled={!soundSettings.enabled}
              />
              <span>{Math.round((soundSettings.volume / PALACE_OST_MAX_VOLUME) * 100)}%</span>
            </label>
          </fieldset>
          <div className="settings-actions">
            <button type="submit">Enregistrer</button>
            <button type="button" onClick={saveAndLaunch}>Lancer une partie</button>
          </div>
          {saved ? <strong className="settings-saved">Paramètres enregistrés</strong> : null}
        </form>
      </section>
    </main>
  );
}

function GameScreen({ state, setState, onNewGame, rules }) {
  const [rulesOpen, setRulesOpen] = React.useState(false);
  const [busy, setBusy] = React.useState(false);

  async function run(action) {
    setBusy(true);
    try {
      setState(await action());
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="game-screen">
      <button className="top-menu-button" type="button" onClick={() => setRulesOpen(true)}>
        <Menu size={18} />
        Menu
      </button>
      <header className="game-header">
        <div>
          <div className="panel-kicker">Love Letter au palais</div>
          <h1>Le Palais du Sultan</h1>
        </div>
        <ScoreBar state={state} />
      </header>
      <div className="game-layout">
        <GameBoard state={state} />
        <aside className="right-column">
          <ActionControls
            state={state}
            busy={busy}
            onPlay={(action) => run(() => api(`/api/games/${state.game_id}/play`, {
              method: "POST",
              body: JSON.stringify({ action }),
            }))}
            onAiStep={() => run(() => api(`/api/games/${state.game_id}/ai-step`, { method: "POST" }))}
            onNextRound={() => run(() => api(`/api/games/${state.game_id}/next-round`, { method: "POST" }))}
            onNewGame={onNewGame}
          />
          <Journal logs={state.logs} />
        </aside>
      </div>
      <RulesPanel open={rulesOpen} rules={rules} onClose={() => setRulesOpen(false)} />
    </main>
  );
}

function App() {
  const [view, setView] = React.useState(readCurrentView);
  const [rules, setRules] = React.useState([]);

  React.useEffect(() => {
    const syncView = () => setView(readCurrentView());
    window.addEventListener("popstate", syncView);
    window.addEventListener("palace:viewchange", syncView);
    return () => {
      window.removeEventListener("popstate", syncView);
      window.removeEventListener("palace:viewchange", syncView);
    };
  }, []);

  React.useEffect(() => {
    api("/api/rules")
      .then((payload) => setRules(payload.rules || []))
      .catch(() => setRules([]));
  }, []);

  function navigate(viewName) {
    pushView(viewName);
  }

  if (view === "cards-preview") {
    return <CardsPreview />;
  }
  if (view === "rules") {
    return <RulesPage rules={rules} />;
  }
  if (view === "settings") {
    return <SettingsPage onNavigate={navigate} />;
  }
  if (view === "game" || view === "stitch-decor") {
    return <StitchRoyalDecorPreview rules={rules} />;
  }

  return <MainMenu onNavigate={navigate} />;
}

createRoot(document.getElementById("root")).render(<App />);
