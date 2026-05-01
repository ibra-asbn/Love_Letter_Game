const PALACE_OST_AUDIO = import.meta.env.VITE_PALACE_OST_AUDIO || "/audio/palace-loop.wav";
const HAS_PALACE_OST_AUDIO = Boolean(PALACE_OST_AUDIO);
export const PALACE_OST_MAX_VOLUME = 0.22;
const PALACE_OST_FADE_MS = 1600;
const DEFAULT_SOUND_SETTINGS = {
  enabled: HAS_PALACE_OST_AUDIO,
  volume: 0.06,
};

let palaceOstAudio = null;
let palaceOstFadeTimer = null;
let palaceOstPrimer = null;

function getPalaceOstAudio() {
  if (!HAS_PALACE_OST_AUDIO) return null;
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

export function isPalaceOstLoaded() {
  return Boolean(palaceOstAudio || window.__palaceOstAudio);
}

export function clampVolume(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return DEFAULT_SOUND_SETTINGS.volume;
  return Math.min(PALACE_OST_MAX_VOLUME, Math.max(0, numberValue));
}

export function readSoundSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("palaceSoundSettings") || "{}");
    return {
      enabled: HAS_PALACE_OST_AUDIO && stored.enabled !== false,
      volume: clampVolume(stored.volume ?? DEFAULT_SOUND_SETTINGS.volume),
    };
  } catch (_error) {
    return { ...DEFAULT_SOUND_SETTINGS };
  }
}

export function applyPalaceOstSettings(settings = readSoundSettings()) {
  const audio = getPalaceOstAudio();
  if (!audio) return null;
  audio.volume = settings.enabled ? clampVolume(settings.volume) : 0;
  if (!settings.enabled) {
    audio.pause();
  }
  return audio;
}

export function primePalaceOst() {
  const settings = readSoundSettings();
  if (!settings.enabled) return null;
  const audio = getPalaceOstAudio();
  if (!audio) return null;
  window.clearInterval(palaceOstFadeTimer);
  audio.volume = 0;
  palaceOstPrimer = audio.play().catch(() => null);
  return palaceOstPrimer;
}

export function startPalaceOst() {
  const settings = readSoundSettings();
  if (!settings.enabled) return;
  const audio = getPalaceOstAudio();
  if (!audio) return;
  const targetVolume = clampVolume(settings.volume);
  window.clearInterval(palaceOstFadeTimer);
  audio.volume = Math.min(audio.volume || 0, targetVolume);
  const playPromise = audio.paused ? audio.play() : palaceOstPrimer;
  playPromise?.catch(() => {
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

export function disposePalaceOstTimers() {
  window.clearInterval(palaceOstFadeTimer);
}
