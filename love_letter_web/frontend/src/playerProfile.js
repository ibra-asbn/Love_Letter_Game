export const DEFAULT_ENTRY_REASON_OPTIONS = [
  { id: "challenge_family", label: "Défier le Sultan et sa famille", signal: "challenger" },
  { id: "observe_champions", label: "Observer les champions de la cour", signal: "recruiter_hint" },
  { id: "evaluate_contender", label: "Évaluer la force d’un prétendant", signal: "recruiter_hint" },
  { id: "support_close_one", label: "Accompagner un proche", signal: "close_one" },
  { id: "learn_rules", label: "Découvrir les règles du palais", signal: "learner" },
];

export const DEFAULT_PLAYER_PROFILE = {
  first_name: "Hafsa",
  last_name: "",
  display_name: "Hafsa",
  full_name: "Hafsa",
  player_identity_id: "",
  identity_confirmed: false,
  identity_found: false,
  stats: {
    matches_played: 0,
    wins: 0,
    losses: 0,
    winrate: 0,
    rounds_played: 0,
    rounds_won: 0,
  },
  entry_reason: "challenge_family",
  entry_reason_label: "Défier le Sultan et sa famille",
};

const PLAYER_PROFILE_STORAGE_KEY = "palacePlayerProfile";

export function cleanProfilePart(value, fallback = "") {
  const text = String(value || "").trim().replace(/\s+/g, " ");
  return text || fallback;
}

export function normalizeEntryReasonId(value) {
  const validIds = new Set(DEFAULT_ENTRY_REASON_OPTIONS.map((reason) => reason.id));
  return validIds.has(value) ? value : DEFAULT_PLAYER_PROFILE.entry_reason;
}

export function reasonLabel(reasonId, options = DEFAULT_ENTRY_REASON_OPTIONS) {
  return options.find((reason) => reason.id === reasonId)?.label || DEFAULT_PLAYER_PROFILE.entry_reason_label;
}

export function readPlayerProfile() {
  try {
    const stored = JSON.parse(localStorage.getItem(PLAYER_PROFILE_STORAGE_KEY) || "{}");
    const legacyName = localStorage.getItem("palacePlayerName") || "";
    const firstName = cleanProfilePart(stored.first_name, cleanProfilePart(legacyName, DEFAULT_PLAYER_PROFILE.first_name));
    const lastName = cleanProfilePart(stored.last_name);
    const entryReason = normalizeEntryReasonId(stored.entry_reason);
    return {
      ...DEFAULT_PLAYER_PROFILE,
      ...stored,
      first_name: firstName,
      last_name: lastName,
      display_name: firstName,
      full_name: [firstName, lastName].filter(Boolean).join(" ") || firstName,
      player_identity_id: stored.player_identity_id || "",
      identity_confirmed: Boolean(stored.identity_confirmed),
      identity_found: Boolean(stored.identity_found),
      stats: stored.stats || DEFAULT_PLAYER_PROFILE.stats,
      entry_reason: entryReason,
      entry_reason_label: stored.entry_reason_label || reasonLabel(entryReason),
    };
  } catch (_error) {
    const legacyName = cleanProfilePart(localStorage.getItem("palacePlayerName"), DEFAULT_PLAYER_PROFILE.first_name);
    return {
      ...DEFAULT_PLAYER_PROFILE,
      first_name: legacyName,
      display_name: legacyName,
      full_name: legacyName,
    };
  }
}

export function savePlayerProfile(profile) {
  const firstName = cleanProfilePart(profile.first_name, DEFAULT_PLAYER_PROFILE.first_name);
  const lastName = cleanProfilePart(profile.last_name);
  const entryReason = normalizeEntryReasonId(profile.entry_reason);
  const nextProfile = {
    ...DEFAULT_PLAYER_PROFILE,
    ...profile,
    first_name: firstName,
    last_name: lastName,
    display_name: firstName,
    full_name: [firstName, lastName].filter(Boolean).join(" ") || firstName,
    player_identity_id: profile.player_identity_id || "",
    identity_confirmed: Boolean(profile.identity_confirmed),
    identity_found: Boolean(profile.identity_found),
    stats: profile.stats || DEFAULT_PLAYER_PROFILE.stats,
    entry_reason: entryReason,
    entry_reason_label: profile.entry_reason_label || reasonLabel(entryReason),
  };
  localStorage.setItem(PLAYER_PROFILE_STORAGE_KEY, JSON.stringify(nextProfile));
  localStorage.setItem("palacePlayerName", nextProfile.display_name);
  return nextProfile;
}

export function profileRequestPayload(profile = readPlayerProfile()) {
  return {
    first_name: cleanProfilePart(profile.first_name, DEFAULT_PLAYER_PROFILE.first_name),
    last_name: cleanProfilePart(profile.last_name),
    entry_reason: normalizeEntryReasonId(profile.entry_reason),
    identity_confirmed: Boolean(profile.identity_confirmed),
  };
}

export function fallbackQadiDialogue(profile = DEFAULT_PLAYER_PROFILE) {
  const displayName = cleanProfilePart(profile.display_name || profile.first_name, DEFAULT_PLAYER_PROFILE.display_name);
  return [
    {
      title: "Le Qadi",
      text: `Bienvenue, ${displayName}. Le palais inscrit ton nom avec soin.`,
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
      text: "Avance maintenant. La table est prête. Que ton nom soit plus qu'une ligne de plus dans mon registre.",
    },
  ];
}
