import React from "react";
import { Eye, Pause, Play, RotateCcw, SkipBack, SkipForward, X } from "lucide-react";
import { api } from "./api";
import { profileRequestPayload } from "./playerProfile";
import { PalaceMenu } from "./PalaceMenu";
import stitchAmiraSpriteUrl from "./assets/stitch-amira-fullbody-cutout.png";
import stitchSultanSpriteUrl from "./assets/stitch-sultan-fullbody-cutout.png";
import stitchSultanaSpriteUrl from "./assets/stitch-sultana-fullbody-cutout.png";
import { cardArtById, cardBackArt } from "./assets/cards";
import {
  CARD_NAMES,
  cardLabel,
  isChancellorChoiceAction,
  palaceText,
  readAiPolicySettings,
} from "./gameConfig";

const ACTION_TO_DRAW_DELAY_MS = 720;
const DRAW_SEQUENCE_MS = 980;

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
          key={`${player.id}-hidden-${index}`}
          hidden
          className={`decor-held-card held-${index}`}
        />
      ))}
    </div>
  );
}

function concealDrawnCard(player) {
  if (!player || !player.alive) return player;
  if (player.is_human) {
    const hand = player.hand || [];
    return {
      ...player,
      hand: hand.slice(0, Math.max(0, hand.length - 1)),
    };
  }
  return {
    ...player,
    hand_count: Math.max(1, (player.hand_count || 1) - 1),
  };
}

function DecorDrawFlight({ actor, animationKey, roundOver, matchOver }) {
  if (!actor || roundOver || matchOver) return null;

  return (
    <DecorGameCard
      key={animationKey}
      hidden
      className={`decor-draw-flight to-${actor}`}
    />
  );
}

function DecorPlayerBadge({ player, className = "" }) {
  if (!player) return null;
  return (
    <div className={`decor-player-badge ${className} ${player.alive ? "" : "is-dead"}`}>
      <strong>{player.name}</strong>
      <span>{player.score}/2</span>
      {player.protected ? (
        <em className="decor-protection-label" title="Protégée">
          <span className="decor-protection-full">protégée</span>
          <span className="decor-protection-short">prot.</span>
        </em>
      ) : null}
    </div>
  );
}

function DecorSpeechBubble({ player, className = "", visible = true }) {
  const speechText = player?.speech?.text || "";
  const speechKey = speechText ? `${player.id}-${player.speech.ts || ""}-${speechText}` : "";
  const [autoVisible, setAutoVisible] = React.useState(false);

  React.useEffect(() => {
    if (!speechKey) {
      setAutoVisible(false);
      return undefined;
    }
    setAutoVisible(true);
    const timeoutId = window.setTimeout(() => setAutoVisible(false), 3300);
    return () => window.clearTimeout(timeoutId);
  }, [speechKey]);

  if (!speechText || !visible || !autoVisible) return null;
  const tone = player.speech.tone || "normal";
  const text = palaceText(speechText);
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

function DecorActionPanel({
  state,
  busy,
  presentationText = "",
  onPlay,
  onAiStep,
  onNextRound,
  onNewGame,
  onShowReplay,
}) {
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

  if (presentationText) {
    return (
      <section className="decor-action-panel">
        <h2>Tour du palais</h2>
        <p>{presentationText}</p>
      </section>
    );
  }

  if (state.match_over) {
    const winners = state.players.filter((player) => player.score >= state.target_points);
    return (
      <section className="decor-action-panel">
        <h2>Partie terminee</h2>
        <p>{winners.map((player) => player.name).join(", ")} gagne.</p>
        {state.replay_available ? (
          <button type="button" onClick={onShowReplay} disabled={busy}>
            <Eye size={14} />
            Voir le replay
          </button>
        ) : null}
        <button type="button" onClick={onNewGame} disabled={busy}>
          Nouvelle partie
        </button>
        <a className="decor-panel-link" href="/">
          Retour au palais
        </a>
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
  const chancellorActions = actions.filter((action) => isChancellorChoiceAction(action.action));
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
      {selectedCard === null ? (
        <div className="decor-action-list decor-card-action-list">
          {cards.map((action) => (
            <button key={action.card} type="button" onClick={() => setSelectedCard(action.card)} disabled={busy}>
              <DecorGameCard card={action.card} className="decor-action-card-thumb" />
              <span>{CARD_NAMES[action.card] || action.card_name}</span>
            </button>
          ))}
        </div>
      ) : needsTarget && selectedTarget === null ? (
        <>
          <button className="decor-back-button" type="button" onClick={() => setSelectedCard(null)} disabled={busy}>
            Changer de carte
          </button>
          <h3>{cardLabel(selectedCard)} - cible</h3>
          <div className="decor-action-list">
            {[...new Map(cardActions.map((action) => [action.target, action])).values()].map((action) => (
              <button key={action.target || action.action} type="button" onClick={() => setSelectedTarget(action.target)} disabled={busy}>
                {action.target_name || "Sans cible"}
              </button>
            ))}
          </div>
        </>
      ) : selectedCard === 1 ? (
        <>
          <button className="decor-back-button" type="button" onClick={() => setSelectedTarget(null)} disabled={busy}>
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
          <button className="decor-back-button" type="button" onClick={() => setSelectedCard(null)} disabled={busy}>
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

function replayStateForStep(replay, step) {
  const events = replay?.events || [];
  const event = events[Math.max(0, Math.min(step, events.length - 1))];
  const state = event?.payload?.state_after || event?.payload?.state_before;
  return { event, state };
}

function replayPlayerName(state, playerId) {
  return state?.players?.find((player) => player.id === playerId)?.name || playerId;
}

function roundReasonLabel(reason) {
  if (reason === "last_alive") return "dernier survivant";
  if (reason === "highest_card") return "plus haute carte";
  return "résolution";
}

function replayRoundWinners(event, state) {
  const winners = event?.payload?.winners || [];
  return winners.map((winnerId) => replayPlayerName(state, winnerId)).join(", ");
}

function replayEventTitle(event, state) {
  if (!event) return "Replay";
  if (event.type === "round_started") return `Manche ${event.round_index} - ouverture`;
  if (event.type === "round_finished") {
    const winners = replayRoundWinners(event, state);
    return winners ? `Manche ${event.round_index} - ${winners} marque` : `Manche ${event.round_index} - conclusion`;
  }
  if (event.type === "match_finished") return "Partie terminée";
  return event.payload?.label || event.type;
}

function replayEventDescription(event, state) {
  if (!event) return "";
  if (event.type === "round_finished") {
    const winners = replayRoundWinners(event, state);
    const reason = roundReasonLabel(event.payload?.reason);
    return winners ? `${winners} gagne la manche pour ${reason}.` : "La manche est terminée.";
  }
  if (event.type === "match_finished") {
    const winners = event.payload?.winner_names?.join(", ");
    return winners ? `${winners} gagne la partie.` : "La partie est terminée.";
  }
  return event.payload?.label || replayEventTitle(event, state);
}

function latestActionEvent(events = []) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.type === "action_played") return events[index];
  }
  return null;
}

function DecorReplayPlayer({ player }) {
  const playedCards = (player.played || []).slice(-5);
  return (
    <article className={`decor-replay-player ${player.winner ? "is-winner" : ""}`}>
      <header>
        <strong>{player.name}</strong>
        <span>{player.score}/2</span>
      </header>
      <div className="decor-replay-player-cards">
        <div className="decor-replay-card-row decor-replay-hand-row" aria-label={`Main omnisciente de ${player.name}`}>
          {player.hand?.length ? player.hand.map((card, index) => (
            <DecorGameCard key={`${player.id}-hand-${index}-${card}`} card={card} className="decor-replay-card" />
          )) : <em>main vide</em>}
        </div>
        <div className="decor-replay-played-row" aria-label={`Défausse de ${player.name}`}>
          {playedCards.map((card, index) => (
            <DecorGameCard key={`${player.id}-played-${index}-${card}`} card={card} className="decor-replay-mini-card" />
          ))}
          {(player.played || []).length > playedCards.length ? (
            <span className="decor-replay-more">+{player.played.length - playedCards.length}</span>
          ) : null}
        </div>
      </div>
    </article>
  );
}

function DecorReplayOverlay({
  replay,
  step,
  playing,
  error,
  onClose,
  onPrev,
  onNext,
  onTogglePlay,
  onStepChange,
}) {
  const events = replay?.events || [];
  const { event, state } = replayStateForStep(replay, step);
  const finalStep = Math.max(events.length - 1, 0);
  const winners = replay?.winners?.map((winner) => winner.name).join(", ");

  return (
    <section className="decor-replay-overlay" aria-label="Replay omniscient de la partie">
      <div className="decor-replay-panel">
        <button className="decor-replay-close" type="button" onClick={onClose} aria-label="Fermer le replay">
          <X size={16} />
        </button>
        <div className="decor-replay-heading">
          <span>Replay omniscient</span>
          <strong>{replay ? replayEventTitle(event, state) : "Chargement du registre"}</strong>
          {winners ? <em>Vainqueur final: {winners}</em> : null}
        </div>
        {error ? <p className="decor-replay-error">{error}</p> : null}
        {state ? (
          <>
            <div className="decor-replay-board">
              <div className="decor-replay-players">
                {state.players.map((player) => (
                  <DecorReplayPlayer
                    key={player.id}
                    player={{
                      ...player,
                      winner: replay?.players?.find((item) => item.id === player.id)?.winner,
                    }}
                  />
                ))}
              </div>
              <aside className="decor-replay-deck">
                <strong>Pioche visible</strong>
                <div className="decor-replay-card-row">
                  {(state.deck || []).slice(0, 12).map((card, index) => (
                    <DecorGameCard key={`deck-${index}-${card}`} card={card} className="decor-replay-mini-card" />
                  ))}
                  {!state.deck?.length ? <em>vide</em> : null}
                </div>
                <span>{state.deck_size} cartes restantes</span>
                <strong>Défausse visible</strong>
                <div className="decor-replay-card-row">
                  {(state.discard || []).slice(-12).map((entry, index) => (
                    <DecorGameCard
                      key={`discard-${entry.owner}-${entry.played_index}-${index}-${entry.card}`}
                      card={entry.card}
                      className="decor-replay-mini-card"
                    />
                  ))}
                  {!state.discard?.length ? <em>vide</em> : null}
                </div>
              </aside>
            </div>
            <div className="decor-replay-event">
              <strong>{event?.payload?.actor_name || event?.type}</strong>
              <span>{replayEventDescription(event, state)}</span>
            </div>
          </>
        ) : (
          <p className="decor-replay-empty">Le registre prépare les cartes.</p>
        )}
        <div className="decor-replay-controls">
          <button type="button" onClick={onPrev} disabled={!replay || step <= 0}>
            <SkipBack size={16} />
            Précédent
          </button>
          <button type="button" onClick={onTogglePlay} disabled={!replay || !events.length}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
            {playing ? "Pause" : "Lecture"}
          </button>
          <button type="button" onClick={onNext} disabled={!replay || step >= finalStep}>
            <SkipForward size={16} />
            Suivant
          </button>
        </div>
        <input
          className="decor-replay-range"
          type="range"
          min="0"
          max={finalStep}
          value={Math.min(step, finalStep)}
          disabled={!replay || !events.length}
          onChange={(eventChange) => onStepChange(Number(eventChange.target.value))}
          aria-label="Position du replay"
        />
        <span className="decor-replay-counter">
          {events.length ? `${Math.min(step + 1, events.length)} / ${events.length}` : "0 / 0"}
        </span>
      </div>
    </section>
  );
}

export function StitchRoyalDecorPreview({ onNavigate }) {
  const [state, setState] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [replayOpen, setReplayOpen] = React.useState(false);
  const [replayData, setReplayData] = React.useState(null);
  const [replayStep, setReplayStep] = React.useState(0);
  const [replayPlaying, setReplayPlaying] = React.useState(false);
  const [replayError, setReplayError] = React.useState("");
  const [visualCurrent, setVisualCurrent] = React.useState(null);
  const [drawFlight, setDrawFlight] = React.useState(null);
  const [concealedDrawActor, setConcealedDrawActor] = React.useState(null);
  const [presentationBusy, setPresentationBusy] = React.useState(false);
  const busyRef = React.useRef(false);
  const presentationBusyRef = React.useRef(false);
  const presentationEventRef = React.useRef("");

  async function run(action) {
    if (busyRef.current || presentationBusyRef.current) return;
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
    setReplayOpen(false);
    setReplayData(null);
    setReplayStep(0);
    setReplayPlaying(false);
    await run(() => api("/api/games", {
      method: "POST",
      body: JSON.stringify({
        ...profileRequestPayload(),
        ai_policies: readAiPolicySettings(),
      }),
    }));
  }

  async function showReplay() {
    if (!state?.game_id) return;
    setReplayOpen(true);
    setReplayError("");
    setReplayPlaying(false);
    if (replayData?.game_id === state.game_id) return;
    try {
      const payload = await api(`/api/games/${state.game_id}/replay`);
      setReplayData(payload);
      setReplayStep(0);
    } catch (err) {
      setReplayError(err.message || "Replay indisponible");
    }
  }

  React.useEffect(() => {
    if (!replayOpen || !replayPlaying || !replayData?.events?.length) return undefined;
    const finalStep = replayData.events.length - 1;
    const timeoutId = window.setTimeout(() => {
      setReplayStep((current) => {
        if (current >= finalStep) {
          setReplayPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1050);
    return () => window.clearTimeout(timeoutId);
  }, [replayOpen, replayPlaying, replayData, replayStep]);

  React.useEffect(() => {
    startDecorGame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    if (!state) {
      setVisualCurrent(null);
      setDrawFlight(null);
      setConcealedDrawActor(null);
      setPresentationBusy(false);
      presentationBusyRef.current = false;
      presentationEventRef.current = "";
      return undefined;
    }

    const lastAction = latestActionEvent(state.analytics_events || []);
    const eventId = lastAction?.event_id || "";
    const actor = lastAction?.payload?.actor || null;
    const nextCurrent = state.current_agent || null;

    if (!eventId || eventId === presentationEventRef.current) {
      if (!presentationBusyRef.current) {
        setVisualCurrent(nextCurrent);
      }
      return undefined;
    }

    presentationEventRef.current = eventId;
    setDrawFlight(null);
    setVisualCurrent(actor || nextCurrent);

    const shouldDrawNext = Boolean(
      nextCurrent
      && actor
      && nextCurrent !== actor
      && !state.round_over
      && !state.match_over,
    );

    if (!shouldDrawNext) {
      setConcealedDrawActor(null);
      setPresentationBusy(false);
      presentationBusyRef.current = false;
      const settleTimeout = window.setTimeout(() => {
        setVisualCurrent(nextCurrent);
      }, ACTION_TO_DRAW_DELAY_MS);
      return () => window.clearTimeout(settleTimeout);
    }

    setConcealedDrawActor(nextCurrent);
    setPresentationBusy(true);
    presentationBusyRef.current = true;
    const drawTimeout = window.setTimeout(() => {
      setVisualCurrent(nextCurrent);
      setDrawFlight({
        actor: nextCurrent,
        animationKey: `${eventId}-draw-${nextCurrent}`,
      });
    }, ACTION_TO_DRAW_DELAY_MS);

    const finishTimeout = window.setTimeout(() => {
      setDrawFlight(null);
      setConcealedDrawActor(null);
      setPresentationBusy(false);
      presentationBusyRef.current = false;
    }, ACTION_TO_DRAW_DELAY_MS + DRAW_SEQUENCE_MS);

    return () => {
      window.clearTimeout(drawTimeout);
      window.clearTimeout(finishTimeout);
    };
  }, [state]);

  const displayPlayers = (state?.players || []).map((player) => (
    player.id === concealedDrawActor ? concealDrawnCard(player) : player
  ));
  const playerMap = Object.fromEntries(displayPlayers.map((player) => [player.id, player]));
  const human = playerMap.player_0;
  const allPlayed = discardCardsForTable(state?.players || [], state?.discard_events || []);
  const discardStack = allPlayed.length ? allPlayed.slice(-9) : [];
  const latestDiscardId = discardStack.length ? discardStack[discardStack.length - 1].id : null;
  const drawStack = Array.from({ length: Math.min(Math.max(state?.deck_size || 0, 0), 10) });
  const current = visualCurrent || state?.current_agent;
  const drawActor = drawFlight?.actor;
  const drawAnimationKey = drawFlight?.animationKey || "";
  const presentationText = presentationBusy && concealedDrawActor
    ? `${playerMap[concealedDrawActor]?.name || "Le palais"} pioche.`
    : "";

  return (
    <main className="stitch-decor-screen">
      <div className="landscape-gate" role="status" aria-live="polite">
        <RotateCcw size={34} />
        <strong>Tourne ton téléphone en paysage</strong>
        <span>Le plateau du palais a besoin de largeur pour rester lisible.</span>
      </div>
      <section className="stitch-decor-frame" aria-label="Decor principal Stitch">
        <PalaceMenu
          variant="scene"
          onNavigate={onNavigate}
          onNewGame={startDecorGame}
          busy={busy || presentationBusy}
        />
        <div className="stitch-decor-backdrop" aria-hidden="true" />
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
          actor={drawActor}
          animationKey={drawAnimationKey}
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
          busy={busy || presentationBusy}
          presentationText={presentationText}
          onPlay={(action) => run(() => api(`/api/games/${state.game_id}/play`, {
            method: "POST",
            body: JSON.stringify({ action }),
          }))}
          onAiStep={() => run(() => api(`/api/games/${state.game_id}/ai-step`, { method: "POST" }))}
          onNextRound={() => run(() => api(`/api/games/${state.game_id}/next-round`, { method: "POST" }))}
          onNewGame={startDecorGame}
          onShowReplay={showReplay}
        />
        {replayOpen ? (
          <DecorReplayOverlay
            replay={replayData}
            step={replayStep}
            playing={replayPlaying}
            error={replayError}
            onClose={() => {
              setReplayOpen(false);
              setReplayPlaying(false);
            }}
            onPrev={() => setReplayStep((current) => Math.max(0, current - 1))}
            onNext={() => setReplayStep((current) => Math.min((replayData?.events?.length || 1) - 1, current + 1))}
            onTogglePlay={() => setReplayPlaying((playingNow) => !playingNow)}
            onStepChange={(nextStep) => {
              setReplayPlaying(false);
              setReplayStep(nextStep);
            }}
          />
        ) : null}
        {error ? <div className="error-toast decor-error-toast">{error}</div> : null}
      </section>
    </main>
  );
}
