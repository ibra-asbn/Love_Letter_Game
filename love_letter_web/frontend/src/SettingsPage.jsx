import React from "react";
import {
  PALACE_OST_MAX_VOLUME,
  applyPalaceOstSettings,
  clampVolume,
  isPalaceOstLoaded,
  readSoundSettings,
} from "./palaceAudio";
import {
  DEFAULT_ENTRY_REASON_OPTIONS,
  readPlayerProfile,
  reasonLabel,
  savePlayerProfile,
} from "./playerProfile";
import {
  AI_POLICY_CHARACTERS,
  AI_POLICY_OPTIONS,
  DEFAULT_AI_POLICIES,
  readAiPolicySettings,
} from "./gameConfig";
import { pushView } from "./navigation";

export function SettingsPage({ onNavigate }) {
  const [playerProfile, setPlayerProfile] = React.useState(readPlayerProfile);
  const [aiPolicies, setAiPolicies] = React.useState(readAiPolicySettings);
  const [soundSettings, setSoundSettings] = React.useState(readSoundSettings);
  const [saved, setSaved] = React.useState(false);

  function persistSettings() {
    const savedProfile = savePlayerProfile(playerProfile);
    localStorage.setItem("palaceAiPolicies", JSON.stringify(aiPolicies));
    localStorage.setItem("palaceSoundSettings", JSON.stringify(soundSettings));
    setPlayerProfile(savedProfile);
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
      const audioWasLoaded = isPalaceOstLoaded();
      const audio = applyPalaceOstSettings(next);
      if (next.enabled && audioWasLoaded && audio) {
        audio.play().catch(() => {});
      }
      return next;
    });
  }

  return (
    <main className="settings-page-screen">
      <section className="settings-page-panel" aria-label="Paramètres">
        <div className="panel-kicker">Le Palais du Sultan</div>
        <h1>Paramètres</h1>
        <form className="settings-form" onSubmit={saveSettings}>
          <label>
            Prénom du joueur
            <input
              type="text"
              value={playerProfile.first_name}
              maxLength={18}
              onChange={(event) => setPlayerProfile((current) => ({ ...current, first_name: event.target.value }))}
            />
          </label>
          <label>
            Nom du joueur
            <input
              type="text"
              value={playerProfile.last_name}
              maxLength={28}
              onChange={(event) => setPlayerProfile((current) => ({ ...current, last_name: event.target.value }))}
            />
          </label>
          <label className="settings-select-label">
            Motif d'entrée
            <select
              value={playerProfile.entry_reason}
              onChange={(event) => {
                const entryReason = event.target.value;
                setPlayerProfile((current) => ({
                  ...current,
                  entry_reason: entryReason,
                  entry_reason_label: reasonLabel(entryReason),
                }));
              }}
            >
              {DEFAULT_ENTRY_REASON_OPTIONS.map((reason) => (
                <option key={reason.id} value={reason.id}>
                  {reason.label}
                </option>
              ))}
            </select>
          </label>
          <p>La prochaine partie utilisera le prénom autour de la table et gardera le nom complet dans les journaux locaux.</p>
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
