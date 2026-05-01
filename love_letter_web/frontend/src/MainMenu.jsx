import React from "react";
import { BookOpen, CheckCircle, FastForward, GraduationCap, Menu, Pencil, Play, Sparkles, UserX } from "lucide-react";
import { api } from "./api";
import { primePalaceOst, startPalaceOst } from "./palaceAudio";
import qadiLibraryScene from "./assets/qadi-library-scene.png";
import {
  cleanProfilePart,
  DEFAULT_ENTRY_REASON_OPTIONS,
  DEFAULT_PLAYER_PROFILE,
  fallbackQadiDialogue,
  readPlayerProfile,
  savePlayerProfile,
} from "./playerProfile";
import {
  PALACE_DOORS_INTRO_RATE,
  PALACE_DOORS_INTRO_VIDEO,
} from "./gameConfig";
import { pushView } from "./navigation";

function formatPalmares(stats = {}) {
  const matches = Number(stats.matches_played || 0);
  if (!matches) return "Aucune partie terminée dans le registre.";
  const wins = Number(stats.wins || 0);
  const losses = Number(stats.losses || Math.max(matches - wins, 0));
  const winrate = Number(stats.winrate || 0);
  return `${matches} parties · ${wins} victoires · ${losses} défaites · ${winrate}% de victoire`;
}

export function MainMenu({ onNavigate }) {
  const [stage, setStage] = React.useState("menu");
  const [dialogueIndex, setDialogueIndex] = React.useState(0);
  const [playerProfile, setPlayerProfile] = React.useState(readPlayerProfile);
  const [entryReasonOptions, setEntryReasonOptions] = React.useState(DEFAULT_ENTRY_REASON_OPTIONS);
  const [qadiDialogue, setQadiDialogue] = React.useState(() => fallbackQadiDialogue(readPlayerProfile()));
  const [profileError, setProfileError] = React.useState("");
  const [profileBusy, setProfileBusy] = React.useState(false);
  const cleanPlayerName = cleanProfilePart(playerProfile.first_name, DEFAULT_PLAYER_PROFILE.first_name);
  const activeDialogue = qadiDialogue[Math.min(dialogueIndex, qadiDialogue.length - 1)];
  const isFinalDialogue = dialogueIndex >= qadiDialogue.length - 1;
  const canRushGame = stage === "video" || stage === "black" || stage === "library";

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

  async function loadProfileDialogue(payload) {
    return api("/api/player-profile/dialogue", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  function launchIntro(nextProfile, nextDialogue) {
    const savedProfile = savePlayerProfile(nextProfile);
    setPlayerProfile(savedProfile);
    setQadiDialogue(nextDialogue.length ? nextDialogue : fallbackQadiDialogue(savedProfile));
    startPalaceOst();
    setStage("video");
  }

  async function submitIdentity(event) {
    event.preventDefault();
    primePalaceOst();
    setProfileBusy(true);
    setProfileError("");
    const draftProfile = {
      ...playerProfile,
      first_name: cleanProfilePart(playerProfile.first_name, DEFAULT_PLAYER_PROFILE.first_name),
      last_name: cleanProfilePart(playerProfile.last_name),
    };
    setPlayerProfile(draftProfile);
    try {
      const result = await loadProfileDialogue({
        first_name: draftProfile.first_name,
        last_name: draftProfile.last_name,
      });
      const options = result.entry_reason_options?.length ? result.entry_reason_options : DEFAULT_ENTRY_REASON_OPTIONS;
      setEntryReasonOptions(options);
      const nextProfile = { ...draftProfile, ...(result.player_profile || {}) };
      if (result.requires_identity_confirmation) {
        setPlayerProfile(nextProfile);
        setQadiDialogue(result.qadi_dialogue?.length ? result.qadi_dialogue : fallbackQadiDialogue(nextProfile));
        setStage("confirm-identity");
        return;
      }
      if (result.requires_entry_reason) {
        setQadiDialogue(result.qadi_dialogue?.length ? result.qadi_dialogue : fallbackQadiDialogue(draftProfile));
        setStage("reason");
        return;
      }
      launchIntro(
        nextProfile,
        result.qadi_dialogue || [],
      );
    } catch (err) {
      setProfileError(err.message || "Impossible d'ouvrir le registre du Qadi");
    } finally {
      setProfileBusy(false);
    }
  }

  async function confirmIdentity() {
    primePalaceOst();
    setProfileBusy(true);
    setProfileError("");
    try {
      const result = await loadProfileDialogue({
        first_name: playerProfile.first_name,
        last_name: playerProfile.last_name,
        identity_confirmed: true,
      });
      launchIntro(
        {
          ...playerProfile,
          ...(result.player_profile || {}),
          identity_confirmed: true,
        },
        result.qadi_dialogue || [],
      );
    } catch (err) {
      setProfileError(err.message || "Le Qadi n'a pas pu confirmer cette identité");
    } finally {
      setProfileBusy(false);
    }
  }

  function rejectIdentity(clearName = false) {
    setPlayerProfile((current) => ({
      ...DEFAULT_PLAYER_PROFILE,
      first_name: clearName ? DEFAULT_PLAYER_PROFILE.first_name : current.first_name,
      last_name: "",
    }));
    setProfileError("");
    setStage("name");
  }

  async function chooseEntryReason(reasonId) {
    primePalaceOst();
    setProfileBusy(true);
    setProfileError("");
    try {
      const result = await loadProfileDialogue({
        first_name: playerProfile.first_name,
        last_name: playerProfile.last_name,
        entry_reason: reasonId,
      });
      const selectedReason = entryReasonOptions.find((reason) => reason.id === reasonId);
      launchIntro(
        {
          ...playerProfile,
          ...(result.player_profile || {}),
          entry_reason: reasonId,
          entry_reason_label: result.player_profile?.entry_reason_label || selectedReason?.label,
        },
        result.qadi_dialogue || [],
      );
    } catch (err) {
      setProfileError(err.message || "Le Qadi n'a pas pu noter ce motif");
    } finally {
      setProfileBusy(false);
    }
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

  function watchQadiGame() {
    if (onNavigate) {
      onNavigate("tutorial");
      return;
    }
    pushView("tutorial");
  }

  return (
    <main
      className={`main-menu-screen ${stage === "video" || stage === "black" ? "is-intro-running" : ""}`}
    >
      <video
        className="main-menu-palace-backdrop"
        src={PALACE_DOORS_INTRO_VIDEO}
        muted
        playsInline
        preload="auto"
        aria-hidden="true"
      />
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
        >
          <img
            className="library-qadi-backdrop"
            src={qadiLibraryScene}
            alt=""
            aria-hidden="true"
          />
          <div className="library-dialogue-panel">
            <div className="panel-kicker">La Grande Archive</div>
            <h1>{activeDialogue.title}</h1>
            <p>{activeDialogue.text}</p>
            <div className="library-dialogue-footer">
              {isFinalDialogue ? (
                <>
                  <button className="primary" type="button" onClick={beginDuel}>
                    <Play size={18} />
                    Jouer maintenant
                  </button>
                  <button type="button" onClick={watchQadiGame}>
                    <GraduationCap size={18} />
                    Regarder ma partie
                  </button>
                </>
              ) : (
                <button
                  className="primary"
                  type="button"
                  onClick={() => setDialogueIndex((index) => index + 1)}
                >
                  <Play size={18} />
                  Suivant
                </button>
              )}
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
      {canRushGame ? (
        <button className="main-menu-rush-game" type="button" onClick={enterGame}>
          <FastForward size={18} />
          Se précipiter dans le jeu
        </button>
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
            <a href="?view=tutorial">
              <GraduationCap size={18} />
              Tutoriel guidé
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
        <section className="main-menu-panel main-menu-name-panel" aria-label="Choisir son identité">
          <div className="panel-kicker">Entrée au palais</div>
          <h1>Ton identité</h1>
          <p>Le Sultan veut savoir qui ose s'asseoir a sa table.</p>
          <form className="main-menu-name-form" onSubmit={submitIdentity}>
            <label>
              Prenom
              <input
                type="text"
                value={playerProfile.first_name}
                maxLength={18}
                autoFocus
                onChange={(event) => setPlayerProfile((current) => ({ ...current, first_name: event.target.value }))}
                placeholder="Hafsa"
              />
            </label>
            <label>
              Nom
              <input
                type="text"
                value={playerProfile.last_name}
                maxLength={28}
                onChange={(event) => setPlayerProfile((current) => ({ ...current, last_name: event.target.value }))}
                placeholder="Nom de famille"
              />
            </label>
            <div className="main-menu-actions main-menu-form-actions">
              <button className="primary" type="submit" disabled={profileBusy}>
                <Play size={18} />
                {profileBusy ? "Le Qadi consulte..." : "Entrer dans le palais"}
              </button>
              <button type="button" onClick={() => setStage("menu")}>
                Retour
              </button>
            </div>
            {profileError ? <strong className="settings-saved">{profileError}</strong> : null}
          </form>
        </section>
      ) : null}
      {stage === "reason" ? (
        <section className="main-menu-panel main-menu-name-panel" aria-label="Motif d'entrée">
          <div className="panel-kicker">Le registre du Qadi</div>
          <h1>{qadiDialogue[0]?.title || "Le Qadi"}</h1>
          <p>{qadiDialogue[0]?.text || "Pour quelle raison entrez-vous au palais ?"}</p>
          <div className="main-menu-reason-grid">
            {entryReasonOptions.map((reason) => (
              <button
                key={reason.id}
                type="button"
                disabled={profileBusy}
                onClick={() => chooseEntryReason(reason.id)}
              >
                {reason.label}
              </button>
            ))}
          </div>
          <div className="main-menu-actions main-menu-form-actions">
            <button type="button" onClick={() => setStage("name")}>
              Retour
            </button>
          </div>
          {profileError ? <strong className="settings-saved">{profileError}</strong> : null}
        </section>
      ) : null}
      {stage === "confirm-identity" ? (
        <section className="main-menu-panel main-menu-name-panel" aria-label="Confirmer son identité">
          <div className="panel-kicker">Le registre du Qadi</div>
          <h1>{qadiDialogue[0]?.title || "Le Qadi"}</h1>
          <p>{qadiDialogue[0]?.text || `Le registre me dit que vous êtes ${playerProfile.full_name}. Est-ce bien vous ?`}</p>
          <div className="main-menu-palmares" aria-label="Palmarès joueur">
            <strong>Palmarès</strong>
            <span>{formatPalmares(playerProfile.stats)}</span>
          </div>
          <div className="main-menu-actions main-menu-form-actions">
            <button className="primary" type="button" disabled={profileBusy} onClick={confirmIdentity}>
              <CheckCircle size={18} />
              Oui, c’est moi
            </button>
            <button type="button" disabled={profileBusy} onClick={() => rejectIdentity(false)}>
              <UserX size={18} />
              Non, nouveau visiteur
            </button>
            <button type="button" disabled={profileBusy} onClick={() => rejectIdentity(true)}>
              <Pencil size={18} />
              Modifier mon nom
            </button>
          </div>
          {profileError ? <strong className="settings-saved">{profileError}</strong> : null}
        </section>
      ) : null}
    </main>
  );
}
