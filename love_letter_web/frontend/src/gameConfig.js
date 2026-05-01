export const CARD_NAMES = {
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

export const AI_POLICY_OPTIONS = [
  { id: "champion_cbp", label: "Champion CBP" },
  { id: "step3_fast", label: "Step3 seul" },
  { id: "step2_retarget", label: "Step2" },
  { id: "heuristic_fair", label: "Heuristique" },
  { id: "random", label: "Random" },
];

export const DEFAULT_AI_POLICIES = {
  player_1: "champion_cbp",
  player_2: "champion_cbp",
  player_3: "champion_cbp",
};

export const AI_POLICY_CHARACTERS = [
  { id: "player_1", name: "La Sultane" },
  { id: "player_2", name: "Le Sultan" },
  { id: "player_3", name: "L'Amira" },
];

export const PALACE_DOORS_INTRO_VIDEO = "/palace_zoom_intro.mp4";
export const PALACE_DOORS_INTRO_RATE = 1.12;

const CHANCELLOR_ACTION_MIN = 900;
const CHANCELLOR_ACTION_MAX = 906;

export function isChancellorChoiceAction(action) {
  const value = Number(action);
  return value >= CHANCELLOR_ACTION_MIN && value < CHANCELLOR_ACTION_MAX;
}

export function readAiPolicySettings() {
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

export function cardLabel(card) {
  if (card === null || card === undefined) return "?";
  return `${CARD_NAMES[card] || "?"} (${card})`;
}

export function palaceText(text = "") {
  let output = text;
  for (const [from, to] of JOURNAL_NAME_REPLACEMENTS) {
    output = output.replace(new RegExp(`\\b${from}\\b`, "g"), to);
  }
  return output.trimEnd().replace(/\.+$/g, "");
}
