from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time
import unicodedata
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from love_letter.engine import LoveLetterRLEnv  # noqa: E402
from step7_self_play_league.league_policy import (  # noqa: E402
    LeaguePolicyFactory,
    LeagueRuntimeArgs,
    load_roster,
    policy_by_id,
)


HUMAN = "player_0"
TARGET_POINTS = 2
CHANCELLOR_ACTION_MIN = 900
CHANCELLOR_ACTION_MAX = 906
PARIS_TZ = ZoneInfo("Europe/Paris")
CARD_NAMES = {
    0: "Espionne",
    1: "Garde",
    2: "Qadi",
    3: "Emir",
    4: "Hajib",
    5: "Wali",
    6: "Vizir",
    7: "Sultan",
    8: "Sultane",
    9: "Amira",
}
DEFAULT_NAMES = {
    "player_0": "Hafsa",
    "player_1": "La Sultane",
    "player_2": "Le Sultan",
    "player_3": "L'Amira",
}
AI_AGENT_IDS = ["player_1", "player_2", "player_3"]
POLICY_LABELS = {
    "champion_cbp": "Champion CBP",
    "step3_fast": "Step3 seul",
    "step2_retarget": "Step2",
    "heuristic_fair": "Heuristique",
    "random": "Random",
}
DEFAULT_AI_POLICY_IDS = {agent: "champion_cbp" for agent in AI_AGENT_IDS}
ENTRY_REASON_OPTIONS = [
    {
        "id": "challenge_family",
        "label": "Défier le Sultan et sa famille",
        "signal": "challenger",
    },
    {
        "id": "observe_champions",
        "label": "Observer les champions de la cour",
        "signal": "recruiter_hint",
    },
    {
        "id": "evaluate_contender",
        "label": "Évaluer la force d’un prétendant",
        "signal": "recruiter_hint",
    },
    {
        "id": "support_close_one",
        "label": "Accompagner un proche",
        "signal": "close_one",
    },
    {
        "id": "learn_rules",
        "label": "Découvrir les règles du palais",
        "signal": "learner",
    },
]
ENTRY_REASON_BY_ID = {reason["id"]: reason for reason in ENTRY_REASON_OPTIONS}
DEFAULT_ENTRY_REASON_ID = "challenge_family"
SECRET_PROFILE_DIALOGUES = {
    "ibra|asbn": {
        "intro": [
            "Ibra. Le registre dit que tu connais déjà les couloirs. Fais semblant de découvrir, si cela t'amuse.",
            "Le palais a préparé une table sérieuse pour toi. Les regards seront plus attentifs que d'habitude.",
        ],
        "match_win": [
            "Victoire confirmée. Le Qadi note que le palais supporte plutôt bien tes expériences.",
        ],
        "match_loss": [
            "Défaite notée. Le palais garde le sourire, mais le registre n'oublie jamais tout à fait.",
        ],
    },
    "hafsa|secret": {
        "intro": [
            "Hafsa. Certaines portes ne grincent pas quand elles te reconnaissent.",
            "Le Sultan a demandé qu'on observe cette partie de très près. Rien d'inquiétant, naturellement.",
        ],
        "match_win": [
            "Le palais s'incline. Cette victoire ressemble moins à un hasard qu'à une signature.",
        ],
        "match_loss": [
            "Le palais gagne cette fois. Le Qadi évitera de trop sourire devant toi.",
        ],
    },
}
RULES_TEXT = [
    "But de la partie - Le premier joueur qui atteint 2 points gagne la partie. Une manche remportee vaut 1 point, et l'Espionne peut offrir un point bonus.",
    "Debut de manche - Chaque joueur recoit une carte secrete. Une carte est mise de cote face cachee, puis le reste forme la pioche.",
    "Tour de jeu - A son tour, un joueur pioche une carte, en a donc deux en main, puis en joue une devant tout le monde. Seule la carte jouee applique son effet.",
    "Fin de manche - La manche s'arrete quand il ne reste qu'un joueur en vie ou quand la pioche est vide. Si la pioche est vide, les joueurs encore en vie revelent leur carte: la plus haute valeur marque 1 point.",
    "Espionne (0) - Elle n'a pas d'effet immediat. Si un seul joueur encore en vie a joue une Espionne pendant la manche, il marque 1 point bonus.",
    "Garde (1) - Le joueur cible un adversaire et annonce une carte autre que Garde. Si l'annonce est juste, la cible est eliminee.",
    "Qadi (2) - Le joueur regarde secretement la carte d'un adversaire. Cette information n'est pas annoncee aux autres.",
    "Emir (3) - Le joueur compare sa carte gardee avec celle d'un adversaire. La plus petite carte elimine son proprietaire. En cas d'egalite, personne ne sort.",
    "Hajib (4) - Le joueur est protege jusqu'a son prochain tour. Il ne peut pas etre cible par Garde, Qadi, Emir, Wali ou Sultan.",
    "Wali (5) - Le joueur choisit un joueur, lui-meme compris. La cible defausse sa carte, sans appliquer son effet, puis repioche. Si elle defausse l'Amira, elle est eliminee.",
    "Vizir (6) - Le joueur pioche jusqu'a deux cartes, garde une carte, puis remet les autres au fond de la pioche dans l'ordre choisi.",
    "Sultan (7) - Le joueur echange sa carte avec celle d'un adversaire.",
    "Sultane (8) - Si un joueur a la Sultane avec le Sultan ou le Wali, il doit jouer la Sultane. Il peut aussi la jouer volontairement pour bluffer.",
    "Amira (9) - Si un joueur joue ou defausse l'Amira, il est immediatement elimine.",
]


class PlayerProfileRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    entry_reason: str | None = None
    human_name: str | None = None
    identity_confirmed: bool | None = False


class NewGameRequest(PlayerProfileRequest):
    ai_policies: dict[str, str] | None = None
    is_tutorial: bool | None = False


class PlayActionRequest(BaseModel):
    action: int


LOGS_DIR = PROJECT_ROOT / "love_letter_web" / "logs"
GAME_EVENTS_PATH = LOGS_DIR / "game_events.jsonl"
PLAYER_STATS_PATH = LOGS_DIR / "player_stats.json"
LOG_LOCK = threading.Lock()


def now_stamp() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def normalize_profile_part(value: str | None) -> str:
    raw = (value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_text.split())


def profile_key(first_name: str | None, last_name: str | None) -> str:
    return f"{normalize_profile_part(first_name)}|{normalize_profile_part(last_name)}"


def player_identity_id_from_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"player_{digest}"


def clean_display_part(value: str | None, fallback: str = "") -> str:
    text = " ".join((value or "").strip().split())
    return text or fallback


def normalize_entry_reason_id(value: str | None) -> str:
    if not value:
        return DEFAULT_ENTRY_REASON_ID
    normalized = normalize_profile_part(value)
    for reason in ENTRY_REASON_OPTIONS:
        if normalized in {normalize_profile_part(reason["id"]), normalize_profile_part(reason["label"])}:
            return reason["id"]
    return DEFAULT_ENTRY_REASON_ID


def empty_player_stats() -> dict:
    return {
        "matches_played": 0,
        "wins": 0,
        "losses": 0,
        "winrate": 0.0,
        "rounds_played": 0,
        "rounds_won": 0,
    }


def stats_summary(record: dict | None = None) -> dict:
    stats = empty_player_stats()
    if record:
        stats.update({
            "matches_played": int(record.get("matches_played", 0)),
            "wins": int(record.get("wins", 0)),
            "losses": int(record.get("losses", 0)),
            "rounds_played": int(record.get("rounds_played", 0)),
            "rounds_won": int(record.get("rounds_won", 0)),
        })
    matches = stats["matches_played"]
    stats["winrate"] = round((stats["wins"] / matches) * 100, 1) if matches else 0.0
    return stats


def load_player_stats_store() -> dict:
    try:
        if PLAYER_STATS_PATH.exists():
            with PLAYER_STATS_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    data.setdefault("version", 1)
                    data.setdefault("players", {})
                    data.setdefault("profile_key_index", {})
                    return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": 1, "players": {}, "profile_key_index": {}}


def save_player_stats_store(store: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with PLAYER_STATS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2, sort_keys=True)


def find_player_record(key: str) -> tuple[str, dict | None]:
    identity_id = player_identity_id_from_key(key)
    with LOG_LOCK:
        store = load_player_stats_store()
        indexed_id = store.get("profile_key_index", {}).get(key) or identity_id
        record = store.get("players", {}).get(indexed_id)
    return indexed_id, record


def ensure_player_record(profile: dict) -> dict:
    key = profile["profile_key"]
    identity_id = profile["player_identity_id"]
    with LOG_LOCK:
        store = load_player_stats_store()
        players = store.setdefault("players", {})
        index = store.setdefault("profile_key_index", {})
        identity_id = index.get(key) or identity_id
        now = now_stamp()
        record = players.get(identity_id) or {
            "player_identity_id": identity_id,
            "profile_key": key,
            "first_name": profile.get("first_name") or DEFAULT_NAMES[HUMAN],
            "last_name": profile.get("last_name") or "",
            "full_name": profile.get("full_name") or profile.get("display_name") or DEFAULT_NAMES[HUMAN],
            "created_at": now,
            **empty_player_stats(),
        }
        record.update({
            "profile_key": key,
            "first_name": profile.get("first_name") or record.get("first_name") or DEFAULT_NAMES[HUMAN],
            "last_name": profile.get("last_name") or record.get("last_name") or "",
            "full_name": profile.get("full_name") or record.get("full_name") or DEFAULT_NAMES[HUMAN],
            "last_entry_reason": profile.get("entry_reason") or record.get("last_entry_reason") or DEFAULT_ENTRY_REASON_ID,
            "last_entry_reason_label": profile.get("entry_reason_label") or record.get("last_entry_reason_label") or ENTRY_REASON_BY_ID[DEFAULT_ENTRY_REASON_ID]["label"],
            "last_seen": now,
        })
        players[identity_id] = record
        index[key] = identity_id
        save_player_stats_store(store)
    profile["player_identity_id"] = identity_id
    profile["stats"] = stats_summary(record)
    profile["identity_found"] = True
    return profile["stats"]


def update_player_stats_for_match(game: "GameSession", leaders: list[str]) -> dict:
    profile = game.player_profile or {}
    before = stats_summary(profile.get("stats"))
    human_won = HUMAN in leaders
    with LOG_LOCK:
        store = load_player_stats_store()
        players = store.setdefault("players", {})
        index = store.setdefault("profile_key_index", {})
        key = profile.get("profile_key") or profile_key(profile.get("first_name"), profile.get("last_name"))
        identity_id = index.get(key) or profile.get("player_identity_id") or player_identity_id_from_key(key)
        record = players.get(identity_id) or {
            "player_identity_id": identity_id,
            "profile_key": key,
            "first_name": profile.get("first_name") or DEFAULT_NAMES[HUMAN],
            "last_name": profile.get("last_name") or "",
            "full_name": profile.get("full_name") or profile.get("display_name") or DEFAULT_NAMES[HUMAN],
            "created_at": now_stamp(),
            **empty_player_stats(),
        }
        before = stats_summary(record)
        record["matches_played"] = before["matches_played"] + 1
        record["wins"] = before["wins"] + (1 if human_won else 0)
        record["losses"] = before["losses"] + (0 if human_won else 1)
        record["rounds_played"] = before["rounds_played"] + int(game.round_index)
        record["rounds_won"] = before["rounds_won"] + int(game.match_points.get(HUMAN, 0))
        record["last_entry_reason"] = profile.get("entry_reason") or record.get("last_entry_reason") or DEFAULT_ENTRY_REASON_ID
        record["last_entry_reason_label"] = profile.get("entry_reason_label") or record.get("last_entry_reason_label") or ENTRY_REASON_BY_ID[DEFAULT_ENTRY_REASON_ID]["label"]
        record["last_result"] = "win" if human_won else "loss"
        record["last_seen"] = now_stamp()
        record["last_game_id"] = game.game_id
        players[identity_id] = record
        index[key] = identity_id
        save_player_stats_store(store)
    after = stats_summary(record)
    profile["player_identity_id"] = identity_id
    profile["stats"] = after
    profile["identity_found"] = True
    return {"before": before, "after": after}


def build_player_profile(request: PlayerProfileRequest) -> dict:
    legacy_name = clean_display_part(request.human_name)
    first_name = clean_display_part(request.first_name, legacy_name or DEFAULT_NAMES[HUMAN])
    last_name = clean_display_part(request.last_name)
    key = profile_key(first_name, last_name)
    identity_id, record = find_player_record(key)
    reason_id = normalize_entry_reason_id(request.entry_reason)
    reason = ENTRY_REASON_BY_ID[reason_id]
    return {
        "first_name": first_name,
        "last_name": last_name,
        "display_name": first_name,
        "full_name": " ".join(part for part in [first_name, last_name] if part),
        "profile_key": key,
        "player_identity_id": identity_id,
        "identity_confirmed": bool(request.identity_confirmed),
        "identity_found": bool(record or SECRET_PROFILE_DIALOGUES.get(key)),
        "stats": stats_summary(record),
        "secret_profile": SECRET_PROFILE_DIALOGUES.get(key),
        "entry_reason": reason_id,
        "entry_reason_label": reason["label"],
        "entry_reason_signal": reason["signal"],
    }


def public_player_profile(profile: dict | None) -> dict:
    profile = profile or build_player_profile(PlayerProfileRequest(first_name=DEFAULT_NAMES[HUMAN]))
    return {
        "first_name": profile.get("first_name") or DEFAULT_NAMES[HUMAN],
        "last_name": profile.get("last_name") or "",
        "display_name": profile.get("display_name") or DEFAULT_NAMES[HUMAN],
        "full_name": profile.get("full_name") or profile.get("display_name") or DEFAULT_NAMES[HUMAN],
        "player_identity_id": profile.get("player_identity_id") or player_identity_id_from_key(profile.get("profile_key") or profile_key(profile.get("first_name"), profile.get("last_name"))),
        "identity_confirmed": bool(profile.get("identity_confirmed")),
        "identity_found": bool(profile.get("identity_found")),
        "stats": stats_summary(profile.get("stats")),
        "entry_reason": profile.get("entry_reason") or DEFAULT_ENTRY_REASON_ID,
        "entry_reason_label": profile.get("entry_reason_label") or ENTRY_REASON_BY_ID[DEFAULT_ENTRY_REASON_ID]["label"],
        "entry_reason_signal": profile.get("entry_reason_signal") or ENTRY_REASON_BY_ID[DEFAULT_ENTRY_REASON_ID]["signal"],
        "is_known_profile": bool(profile.get("secret_profile")),
    }


def format_palmares(stats: dict | None) -> str:
    summary = stats_summary(stats)
    if summary["matches_played"] == 0:
        return "Palmarès actuel: aucune partie terminée dans le registre."
    return (
        "Palmarès actuel: "
        f"{summary['matches_played']} parties, "
        f"{summary['wins']} victoires, "
        f"{summary['losses']} défaites, "
        f"{summary['winrate']}% de victoire."
    )


def build_identity_confirmation_dialogue(profile: dict) -> list[dict]:
    full_name = profile.get("full_name") or profile.get("display_name") or DEFAULT_NAMES[HUMAN]
    return [
        {
            "title": "Le Qadi",
            "text": f"Le registre me dit que vous êtes {full_name}. Est-ce bien vous ?",
        },
        {
            "title": "Le Qadi",
            "text": format_palmares(profile.get("stats")),
        },
    ]


def build_intro_qadi_dialogue(profile: dict, needs_entry_reason: bool = False) -> list[dict]:
    display_name = profile["display_name"]
    if needs_entry_reason:
        return [
            {
                "title": "Le Qadi",
                "text": f"{display_name}, avant que le registre ne s'ouvre: pour quelle raison entres-tu au palais ?",
            }
        ]
    secret_profile = profile.get("secret_profile")
    if secret_profile:
        secret_lines = secret_profile.get("intro", [])
        return [{"title": "Le Qadi", "text": line} for line in secret_lines] + [
            {
                "title": "Le Qadi",
                "text": "La table est prête. Que cette partie dise ce que les mots évitent.",
            }
        ]
    reason_label = profile.get("entry_reason_label") or ENTRY_REASON_BY_ID[DEFAULT_ENTRY_REASON_ID]["label"]
    return [
        {
            "title": "Le Qadi",
            "text": f"Bienvenue, {display_name}. Le palais inscrit ton nom avec soin.",
        },
        {
            "title": "Le Qadi",
            "text": "Tu entres ici sous les lanternes, là où le Sultan garde sa cour et ses secrets.",
        },
        {
            "title": "Le Qadi",
            "text": f"Motif déclaré: {reason_label}. Le Qadi note la nuance sans l'annoncer trop fort.",
        },
        {
            "title": "Le Qadi",
            "text": "Sa fille, l'Amira, refuse chaque prétendant. Aucun poème, aucune fortune, aucune promesse ne l'a fait changer d'avis.",
        },
        {
            "title": "Le Qadi",
            "text": "Alors le Sultan a donné une épreuve: battre sa famille à leur jeu préféré, Love Letter.",
        },
        {
            "title": "Le Qadi",
            "text": "Souviens-toi: ici, une carte jouée trop tôt peut te condamner, et une carte gardée trop longtemps peut te trahir.",
        },
        {
            "title": "Le Qadi",
            "text": "Avance maintenant. La table est prête. Que ton nom soit plus qu'une ligne de plus dans mon registre.",
        },
    ]


def build_match_end_qadi_dialogue(game: "GameSession", leaders: list[str]) -> list[dict]:
    profile = game.player_profile or {}
    secret_profile = profile.get("secret_profile")
    human_won = HUMAN in leaders
    stats_delta = game.match_stats_delta or {}
    after_stats = stats_delta.get("after") or profile.get("stats") or empty_player_stats()
    palmares_text = format_palmares(after_stats)
    if secret_profile:
        key = "match_win" if human_won else "match_loss"
        lines = secret_profile.get(key, [])
        if lines:
            return [{"title": "Le Qadi", "text": line} for line in lines] + [
                {"title": "Le Qadi", "text": palmares_text},
            ]
    if human_won:
        return [
            {
                "title": "Le Qadi",
                "text": f"{profile.get('display_name', DEFAULT_NAMES[HUMAN])}, la partie est gagnée. Le Sultan devra relire son propre règlement.",
            },
            {"title": "Le Qadi", "text": palmares_text},
            {"title": "Le Qadi", "text": "Si tu le souhaites, le registre peut maintenant rejouer chaque coup, sans voile ni secret."},
        ]
    winners = ", ".join(game.names[agent] for agent in leaders)
    return [
        {
            "title": "Le Qadi",
            "text": f"Partie terminée. {winners} garde l'avantage, mais le registre laisse toujours une revanche.",
        },
        {"title": "Le Qadi", "text": palmares_text},
        {"title": "Le Qadi", "text": "Le replay omniscient est disponible si tu veux comprendre où la partie a basculé."},
    ]


def card_name(card: int | None, with_value: bool = False) -> str:
    if card is None:
        return "?"
    value = int(card)
    name = CARD_NAMES.get(value, "?")
    return f"{name} ({value})" if with_value else name


def action_card(action: int) -> int:
    return int(action) // 100


def action_target(action: int) -> int:
    return (int(action) % 100) // 10


def action_guess(action: int) -> int:
    return int(action) % 10


def is_chancellor_choice(action: int | None) -> bool:
    if action is None:
        return False
    value = int(action)
    return CHANCELLOR_ACTION_MIN <= value < CHANCELLOR_ACTION_MAX


def compared_card_after_play(hand: list[int], played_card: int) -> int | None:
    remaining = list(hand)
    if played_card in remaining:
        remaining.remove(played_card)
    return remaining[0] if remaining else None


def snapshot(env: LoveLetterRLEnv) -> dict:
    return {
        "alive": {
            agent: agent in env.agents and not env.terminations.get(agent, True)
            for agent in env.possible_agents
        },
        "hands": {agent: list(env._hands.get(agent, [])) for agent in env.possible_agents},
        "played": {agent: list(env._played_cards.get(agent, [])) for agent in env.possible_agents},
        "protected": {agent: bool(env._protected.get(agent, False)) for agent in env.possible_agents},
        "terminations": {agent: bool(env.terminations.get(agent, True)) for agent in env.possible_agents},
        "agent_selection": env.agent_selection,
        "deck": list(env._deck),
        "deck_size": len(env._deck),
        "chancellor_pending": bool(env._chancellor_pending),
        "chancellor_pool": list(env._chancellor_pool),
    }


def replay_state_from_snapshot(game: "GameSession", snap: dict | None = None) -> dict:
    env = game.env
    snap = snap or snapshot(env)
    players = []
    all_terminated = all(snap.get("terminations", {}).values()) if snap.get("terminations") else False
    discard = []
    for agent in env.possible_agents:
        hand = [int(card) for card in snap.get("hands", {}).get(agent, [])]
        played = [int(card) for card in snap.get("played", {}).get(agent, [])]
        players.append({
            "id": agent,
            "name": game.names.get(agent, agent),
            "is_human": agent == HUMAN,
            "policy_id": "human" if agent == HUMAN else game.ai_policy_ids.get(agent),
            "policy_label": "Humain" if agent == HUMAN else POLICY_LABELS.get(game.ai_policy_ids.get(agent), "?"),
            "alive": bool(hand) if all_terminated else bool(snap.get("alive", {}).get(agent, False)),
            "protected": bool(snap.get("protected", {}).get(agent, False)),
            "score": int(game.match_points.get(agent, 0)),
            "hand": hand,
            "played": played,
        })
        discard.extend({
            "owner": agent,
            "owner_name": game.names.get(agent, agent),
            "card": card,
            "played_index": index,
        } for index, card in enumerate(played))
    return {
        "round_index": int(game.round_index),
        "turn_index": int(game.turn_index),
        "current_agent": snap.get("agent_selection"),
        "current_name": game.names.get(snap.get("agent_selection"), snap.get("agent_selection")),
        "deck": [int(card) for card in snap.get("deck", [])],
        "deck_size": int(snap.get("deck_size", len(snap.get("deck", [])))),
        "discard": discard,
        "chancellor_pending": bool(snap.get("chancellor_pending", False)),
        "chancellor_pool": [int(card) for card in snap.get("chancellor_pool", [])],
        "match_points": {agent: int(points) for agent, points in game.match_points.items()},
        "players": players,
    }


def public_state_from_snapshot(game: "GameSession", snap: dict | None = None) -> dict:
    state = replay_state_from_snapshot(game, snap)
    public_players = []
    for player in state["players"]:
        hand = player["hand"] if player["is_human"] else []
        public_players.append({
            **player,
            "hand": hand,
            "hand_count": len(player["hand"]),
        })
    return {
        **state,
        "deck": [],
        "players": public_players,
    }


def decode_action_label(action: int, names: dict[str, str], actor: str | None = None) -> str:
    if is_chancellor_choice(action):
        text = "choisit avec le Vizir"
        return f"{names.get(actor, actor)} {text}" if actor else text
    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    text = f"joue {card_name(card, True)}"
    if card in {1, 2, 3, 5, 7}:
        target = f"player_{target_idx}" if target_idx < 4 else None
        if target is not None:
            text += f" sur {names.get(target, target)}"
    if card == 1:
        text += f" et annonce {card_name(guess, True)}"
    return f"{names.get(actor, actor)} {text}" if actor else text


def chancellor_action_label(env: LoveLetterRLEnv, action: int) -> str:
    pool = list(env._chancellor_pool)
    offset = int(action) - 900
    if not pool:
        return "Choix Vizir"
    if len(pool) >= 3:
        keep_idx = offset // 2
        order_idx = offset % 2
        keep_idx = min(max(keep_idx, 0), len(pool) - 1)
        returned = [card for idx, card in enumerate(pool) if idx != keep_idx]
        if order_idx == 1:
            returned = list(reversed(returned))
        return (
            f"Garder {card_name(pool[keep_idx], True)}; "
            f"fond {', '.join(card_name(card, True) for card in returned)}"
        )
    keep_idx = min(max(offset, 0), len(pool) - 1)
    return f"Garder {card_name(pool[keep_idx], True)}"


def action_payload(env: LoveLetterRLEnv, action: int, names: dict[str, str], actor: str) -> dict:
    if is_chancellor_choice(action):
        return {
            "action": int(action),
            "card": 6,
            "card_name": CARD_NAMES[6],
            "target": None,
            "guess": None,
            "label": chancellor_action_label(env, action),
            "full_label": decode_action_label(action, names, actor),
        }
    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    target = f"player_{target_idx}" if target_idx < env.num_players else None
    return {
        "action": int(action),
        "card": int(card),
        "card_name": CARD_NAMES.get(card, "?"),
        "target": target,
        "target_name": names.get(target, target) if target else None,
        "guess": int(guess) if card == 1 else None,
        "guess_name": card_name(guess, True) if card == 1 else None,
        "label": decode_action_label(action, names),
        "full_label": decode_action_label(action, names, actor),
    }


def valid_action_payloads(env: LoveLetterRLEnv, agent: str, names: dict[str, str]) -> list[dict]:
    obs = env.observe(agent)
    actions = [int(action) for action in np.where(obs["action_mask"] == 1)[0]]
    return [action_payload(env, action, names, agent) for action in actions]


def say(lines: list[str], **values) -> str:
    return random.choice(lines).format(**values)


def build_speech(names: dict[str, str], actor: str, action: int | None, prev: dict) -> tuple[str, str]:
    if action is None:
        return "", "normal"
    if is_chancellor_choice(action):
        return say([
            "Je range le paquet. Tu verras plus tard",
            "Le Vizir remet de l'ordre dans le destin",
            "Une carte pour moi, deux secrets pour le palais",
            "Le fond de la pioche a parfois meilleure memoire que nous",
        ]), "normal"
    card = action_card(action)
    target_idx = action_target(action)
    target = f"player_{target_idx}" if target_idx < 4 else None
    target_name = names.get(target, target) if target else ""
    target_tone = "taunt" if target == HUMAN else "normal"
    if card == 0:
        return say([
            "Je passe dans l'ombre",
            "Personne ne regarde jamais assez les Espionnes",
            "Un petit pas discret vaut parfois un point",
            "Silence. Les murs ecoutent",
        ]), "normal"
    if card == 1 and target:
        guess = action_guess(action)
        real = prev["hands"].get(target, [None])[0]
        if real == guess:
            return say([
                "Je t'avais lu, {target}",
                "{target}, ton secret etait mal cache",
                "Exactement ce que je pensais, {target}",
                "Le Garde n'a pas frappe au hasard, {target}",
            ], target=target_name), "taunt" if target == HUMAN else "good"
        return say([
            "Hmm. Pas ca. Je note",
            "Tu gardes encore ton mystere, {target}",
            "Ratage elegant. Ca arrive meme au palais",
            "Mauvaise intuition. Mais la prochaine sera meilleure",
        ], target=target_name), "normal"
    if card == 2 and target:
        return say([
            "Montre-moi ta carte, {target}",
            "Un regard suffit, {target}",
            "Le Qadi demande la verite en silence",
            "Je veux savoir ce que tu caches, {target}",
        ], target=target_name), target_tone
    if card == 3 and target:
        actor_card = compared_card_after_play(prev["hands"].get(actor, []), 3)
        target_card = prev["hands"].get(target, [None])[0]
        if actor_card is not None and target_card is not None and actor_card > target_card:
            return say([
                "Allez dehors, {target}",
                "L'Emir tranche. Tu tombes, {target}",
                "Plus petit que moi, {target}. Cruel mais clair",
                "Le duel est pour moi, {target}",
            ], target=target_name), "taunt" if target == HUMAN else "good"
        if actor_card is not None and target_card is not None and actor_card < target_card:
            return say([
                "Oups. Mauvais duel",
                "J'ai provoque plus grand que moi",
                "L'Emir s'incline. Quelle humiliation",
                "Trop ambitieux. Je paie le prix",
            ]), "bad"
        return say([
            "Egalite. On continue",
            "Deux forces egales. Le palais soupire",
            "Ni vainqueur, ni vaincu",
            "Egalite parfaite. Rien ne bouge",
        ]), "normal"
    if card == 4:
        return say([
            "Personne ne me touche ce tour",
            "Le Hajib ferme les portes",
            "Je reste derriere les rideaux",
            "Approcher serait impoli",
        ]), "normal"
    if card == 5 and target:
        old_card = prev["hands"].get(target, [None])[0]
        if old_card == 9:
            return say([
                "{target}, quelle carte dangereuse tu portais",
                "L'Amira tombe avec fracas, {target}",
                "Un Wali bien place peut etre fatal, {target}",
            ], target=target_name), "taunt" if target == HUMAN else "good"
        return say([
            "Jette-moi ca, {target}",
            "Changeons ton avenir, {target}",
            "Le Wali ordonne, {target} obeit",
            "Cette carte ne me plaisait pas, {target}",
        ], target=target_name), target_tone
    if card == 6:
        return say([
            "Je vais organiser l'avenir",
            "Le Vizir aime les choix impossibles",
            "Une bonne main se construit dans l'ombre",
            "Le paquet va apprendre la discipline",
        ]), "normal"
    if card == 7 and target:
        return say([
            "On echange, {target}",
            "Ton destin me tente, {target}",
            "Le Sultan prend ce qui l'intrigue",
            "Nos secrets changent de mains, {target}",
        ], target=target_name), target_tone
    if card == 8:
        return say([
            "Ce n'est pas forcement ce que tu crois",
            "La Sultane sort avec dignite",
            "Obligation ou bluff ? A toi de trembler",
            "Je pose la Sultane, et je ne dirai rien de plus",
        ]), "normal"
    if card == 9:
        return say([
            "Bon. Ca, c'etait dramatique",
            "L'Amira ne pardonne jamais",
            "Je viens de signer ma sortie",
            "Le palais retiendra cette erreur",
        ]), "bad"
    return "", "normal"


def consequence_logs(env: LoveLetterRLEnv, names: dict[str, str], actor: str, action: int | None, prev: dict) -> list[dict]:
    if action is None:
        return []
    logs = []
    if is_chancellor_choice(action):
        if actor == HUMAN and env._hands.get(actor):
            logs.append({"text": f"{names[actor]} garde {card_name(env._hands[actor][0], True)}.", "tone": "info"})
        return logs

    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    target = f"player_{target_idx}" if target_idx < env.num_players else None

    if card == 1 and target:
        old_card = prev["hands"].get(target, [None])[0]
        if old_card == guess:
            logs.append({"text": f"Le Garde touche: {names[target]} avait {card_name(old_card, True)}.", "tone": "good"})
        else:
            logs.append({"text": f"Le Garde rate: {names[target]} n'avait pas {card_name(guess, True)}.", "tone": "muted"})
    elif card == 3 and target:
        target_card = prev["hands"].get(target, [None])[0]
        actor_card = compared_card_after_play(prev["hands"].get(actor, []), 3)
        human_involved = actor == HUMAN or target == HUMAN
        if actor_card is not None and target_card is not None and human_involved:
            if actor_card > target_card:
                logs.append({"text": f"Emir gagne pour {names[actor]}: {names[actor]} avait {card_name(actor_card, True)}, {names[target]} avait {card_name(target_card, True)}.", "tone": "good"})
            elif actor_card < target_card:
                logs.append({"text": f"Emir perdu par {names[actor]}: {names[actor]} avait {card_name(actor_card, True)}, {names[target]} avait {card_name(target_card, True)}.", "tone": "bad"})
            else:
                logs.append({"text": f"Emir: egalite entre {names[actor]} et {names[target]}: {names[actor]} avait {card_name(actor_card, True)}, {names[target]} avait {card_name(target_card, True)}.", "tone": "muted"})
        elif actor_card is not None and target_card is not None and actor_card > target_card:
            logs.append({"text": f"Emir: {names[target]} sort avec {card_name(target_card, True)}. {names[actor]} avait une carte plus forte.", "tone": "good"})
        elif actor_card is not None and target_card is not None and actor_card < target_card:
            logs.append({"text": f"Emir: {names[actor]} sort avec {card_name(actor_card, True)}. {names[target]} avait une carte plus forte.", "tone": "bad"})
        else:
            logs.append({"text": f"Emir: egalite entre {names[actor]} et {names[target]}. Aucune carte n'est revelee.", "tone": "muted"})
    elif card == 4:
        logs.append({"text": f"{names[actor]} est protegee jusqu'a son prochain tour.", "tone": "info"})
    elif card == 5 and target:
        old_card = prev["hands"].get(target, [None])[0]
        logs.append({"text": f"{names[target]} defausse {card_name(old_card, True)}.", "tone": "normal"})
        if old_card == 9:
            logs.append({"text": f"{names[target]} defausse l'Amira et sort.", "tone": "bad"})
    elif card == 7 and target:
        logs.append({"text": f"{names[actor]} et {names[target]} echangent leurs mains.", "tone": "normal"})
    elif card == 9:
        logs.append({"text": f"{names[actor]} joue ou defausse l'Amira et sort.", "tone": "bad"})
    return logs


@dataclass
class GameSession:
    game_id: str
    names: dict[str, str]
    player_profile: dict = field(default_factory=dict)
    is_tutorial: bool = False
    ai_policy_ids: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_AI_POLICY_IDS))
    env: LoveLetterRLEnv = field(default_factory=lambda: LoveLetterRLEnv(num_players=4))
    match_points: dict[str, int] = field(default_factory=lambda: {f"player_{idx}": 0 for idx in range(4)})
    round_index: int = 0
    next_starter: str | None = None
    round_over: bool = False
    match_over: bool = False
    logs: list[dict] = field(default_factory=list)
    discard_events: list[dict] = field(default_factory=list)
    speeches: dict[str, dict] = field(default_factory=dict)
    qadi_dialogue: list[dict] = field(default_factory=list)
    last_speaker: str | None = None
    private_notes: list[dict] = field(default_factory=list)
    analytics_events: list[dict] = field(default_factory=list)
    structured_events: list[dict] = field(default_factory=list)
    ai_policies: dict[str, object] = field(default_factory=dict)
    turn_index: int = 0
    match_stats_recorded: bool = False
    match_stats_delta: dict | None = None
    seed: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, text: str, tone: str = "normal") -> None:
        self.logs.append({"ts": now_stamp(), "text": text, "tone": tone})

    def set_speech(self, agent: str, text: str, tone: str = "normal") -> None:
        if text:
            self.speeches[agent] = {"text": text, "tone": tone, "ts": time.time()}
            self.last_speaker = agent


app = FastAPI(title="Love Letter Champion API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

POLICY_FACTORY = LeaguePolicyFactory(LeagueRuntimeArgs(device="cpu"))
ROSTER_POLICIES = policy_by_id(load_roster())
GAMES: dict[str, GameSession] = {}


class RandomSeat:
    def act(self, _env, obs_dict, _agent: str) -> int:
        valid = np.flatnonzero(obs_dict["action_mask"])
        if len(valid) == 0:
            return 0
        return int(random.choice(valid.tolist()))


def normalize_ai_policy_ids(payload: dict[str, str] | None) -> dict[str, str]:
    selected = dict(DEFAULT_AI_POLICY_IDS)
    for agent, policy_id in (payload or {}).items():
        if agent not in AI_AGENT_IDS:
            continue
        if policy_id not in POLICY_LABELS:
            raise HTTPException(status_code=400, detail=f"Modèle inconnu: {policy_id}")
        selected[agent] = policy_id
    return selected


def policy_role(policy_id: str) -> str:
    if policy_id == "heuristic_fair":
        return "heuristic"
    if policy_id == "random":
        return "random"
    return "model"


def make_ai_policies(policy_ids: dict[str, str]) -> dict[str, object]:
    roles = {HUMAN: "human"}
    roles.update({agent: policy_role(policy_ids.get(agent, "champion_cbp")) for agent in AI_AGENT_IDS})
    policies = {}
    for agent in AI_AGENT_IDS:
        policy_id = policy_ids.get(agent, "champion_cbp")
        if policy_id == "random":
            policies[agent] = RandomSeat()
            continue
        spec = ROSTER_POLICIES.get(policy_id)
        if spec is None:
            raise HTTPException(status_code=400, detail=f"Modèle absent du roster: {policy_id}")
        policies[agent] = POLICY_FACTORY.make(spec, agent, roles)
    return policies


def compact_event_payload(payload: dict) -> dict:
    compact = {}
    for key, value in payload.items():
        if key in {"state_before", "state_after", "public_state_before", "public_state_after"}:
            continue
        compact[key] = value
    return compact


def record_game_event(game: GameSession, event_type: str, payload: dict | None = None) -> None:
    payload = payload or {}
    event = {
        "event_id": f"{game.game_id}-{len(game.structured_events) + 1:04d}",
        "ts": now_stamp(),
        "type": event_type,
        "game_id": game.game_id,
        "is_tutorial": bool(game.is_tutorial),
        "round_index": int(game.round_index),
        "turn_index": int(game.turn_index),
        "player_profile": public_player_profile(game.player_profile),
        "ai_policy_ids": dict(game.ai_policy_ids),
        "match_points": {agent: int(points) for agent, points in game.match_points.items()},
        "payload": payload,
    }
    game.structured_events.append(event)
    game.analytics_events.append({
        **event,
        "payload": compact_event_payload(payload),
    })
    try:
        with LOG_LOCK:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with GAME_EVENTS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        game.add_log("Journal local indisponible: l'evenement n'a pas ete ecrit sur disque.", "muted")


def start_round(game: GameSession, reset_match: bool = False) -> None:
    if reset_match:
        game.match_points = {f"player_{idx}": 0 for idx in range(4)}
        game.round_index = 0
        game.next_starter = None
        game.logs.clear()
        game.discard_events.clear()
        game.speeches.clear()
        game.last_speaker = None
        game.private_notes.clear()
        game.analytics_events.clear()
        game.structured_events.clear()
        game.turn_index = 0
        game.match_stats_recorded = False
        game.match_stats_delta = None
        game.match_over = False
    game.round_over = False
    game.round_index += 1
    game.discard_events.clear()
    game.private_notes.clear()
    game.last_speaker = None
    game.env = LoveLetterRLEnv(num_players=4)
    game.seed = int(time.time() * 1000) % 2_000_000_000
    options = {"tokens": dict(game.match_points)}
    if game.next_starter:
        options["starting_agent"] = game.next_starter
    game.env.reset(seed=game.seed, options=options)
    game.ai_policies = make_ai_policies(game.ai_policy_ids)
    game.add_log(f"Manche {game.round_index}.", "info")
    if reset_match and game.player_profile:
        profile = public_player_profile(game.player_profile)
        game.add_log(f"Joueur: {profile['full_name']}.", "info")
        game.add_log(f"Motif d'entree: {profile['entry_reason_label']}.", "info")
    composition = ", ".join(
        f"{game.names[agent]}: {POLICY_LABELS.get(game.ai_policy_ids.get(agent), game.ai_policy_ids.get(agent))}"
        for agent in AI_AGENT_IDS
    )
    game.add_log(f"Adversaires: {composition}.", "info")
    game.add_log(f"Premier joueur: {game.names[game.env.agent_selection]}.", "info")
    record_game_event(game, "round_started", {
        "seed": game.seed,
        "starting_agent": game.env.agent_selection,
        "starting_name": game.names.get(game.env.agent_selection, game.env.agent_selection),
        "state_after": replay_state_from_snapshot(game),
        "public_state_after": public_state_from_snapshot(game),
    })


def record_discard_events(game: GameSession, actor: str, action: int | None, prev: dict) -> None:
    if action is None or is_chancellor_choice(action):
        return

    env = game.env
    previous_played = prev.get("played", {})
    ordered_agents = [actor] + [agent for agent in env.possible_agents if agent != actor]
    for owner in ordered_agents:
        before_count = len(previous_played.get(owner, []))
        after_cards = list(env._played_cards.get(owner, []))
        for played_index, card in enumerate(after_cards[before_count:], start=before_count):
            game.discard_events.append({
                "id": f"r{game.round_index}-d{len(game.discard_events)}",
                "round": int(game.round_index),
                "owner": owner,
                "actor": actor,
                "card": int(card),
                "played_index": int(played_index),
                "is_human": owner == HUMAN,
            })


def get_game(game_id: str) -> GameSession:
    if game_id not in GAMES:
        raise HTTPException(status_code=404, detail="Partie introuvable")
    return GAMES[game_id]


def finalize_round_if_needed(game: GameSession) -> None:
    env = game.env
    if game.round_over or not all(env.terminations.values()):
        return
    game.round_over = True
    winners = list(getattr(env, "_round_winners", []))
    spy = getattr(env, "_round_spy_winner", None)
    reason = getattr(env, "_round_win_reason", None)
    points_awarded = {agent: 0 for agent in env.possible_agents}
    if winners:
        reason_text = "dernier survivant" if reason == "last_alive" else "plus haute carte"
        names = ", ".join(game.names[agent] for agent in winners)
        game.add_log(f"Fin de manche: {names} marque pour {reason_text}.", "good")
        if reason == "highest_card":
            revealed = [
                f"{game.names[agent]} {card_name(env._hands[agent][0], True)}"
                for agent in env.possible_agents
                if env._hands.get(agent)
            ]
            game.add_log(f"Cartes finales revelees: {', '.join(revealed)}.", "info")
        for winner in winners:
            points_awarded[winner] += 1
            game.set_speech(winner, say([
                "Et voila, la manche est pour moi",
                "Le palais connait son vainqueur",
                "Une manche de plus dans ma bourse",
                "La lettre arrive entre de bonnes mains",
            ]), "good")
    if spy:
        points_awarded[spy] += 1
        game.add_log(f"Bonus Espionne pour {game.names[spy]}.", "good")
        game.set_speech(spy, say([
            "Petit bonus Espionne. On prend",
            "J'avais tout vu depuis l'ombre",
            "L'Espionne travaille mieux quand personne ne l'applaudit",
            "Un point silencieux vaut bien un grand discours",
        ]), "good")
    for agent, points in points_awarded.items():
        game.match_points[agent] = game.match_points.get(agent, 0) + points
    if any(points_awarded.values()):
        text = ", ".join(f"{game.names[agent]} +{points}" for agent, points in points_awarded.items() if points)
        game.add_log(f"Points de manche: {text}.", "info")
    game.next_starter = winners[0] if winners else None
    game.match_over = any(points >= TARGET_POINTS for points in game.match_points.values())
    score = ", ".join(f"{game.names[agent]} {game.match_points[agent]}/{TARGET_POINTS}" for agent in env.possible_agents)
    game.add_log(f"Score de partie: {score}.", "info")
    record_game_event(game, "round_finished", {
        "winners": winners,
        "spy_bonus": spy,
        "reason": reason,
        "points_awarded": points_awarded,
        "match_over": bool(game.match_over),
        "state_after": replay_state_from_snapshot(game),
        "public_state_after": public_state_from_snapshot(game),
    })
    if game.match_over:
        leaders = [agent for agent, points in game.match_points.items() if points >= TARGET_POINTS]
        if game.is_tutorial and not game.match_stats_recorded:
            game.match_stats_delta = {"skipped": True, "reason": "tutorial"}
            game.match_stats_recorded = True
        elif not game.match_stats_recorded:
            game.match_stats_delta = update_player_stats_for_match(game, leaders)
            game.match_stats_recorded = True
        game.add_log(f"Partie terminee: {', '.join(game.names[agent] for agent in leaders)} gagne.", "good")
        match_dialogue = build_match_end_qadi_dialogue(game, leaders)
        game.qadi_dialogue.extend(match_dialogue)
        for line in match_dialogue:
            game.add_log(f"{line['title']}: {line['text']}", "qadi")
        record_game_event(game, "match_finished", {
            "winners": leaders,
            "winner_names": [game.names[agent] for agent in leaders],
            "human_won": HUMAN in leaders,
            "player_stats": game.match_stats_delta,
            "state_after": replay_state_from_snapshot(game),
            "public_state_after": public_state_from_snapshot(game),
        })


def apply_action(game: GameSession, action: int | None) -> None:
    env = game.env
    actor = env.agent_selection
    prev = snapshot(env)
    env.step(action)
    after = snapshot(env)
    record_discard_events(game, actor, action, prev)
    speech_text, speech_tone = build_speech(game.names, actor, action, prev)
    game.set_speech(actor, speech_text, speech_tone)
    if action is not None:
        game.turn_index += 1
        card = 6 if is_chancellor_choice(action) else action_card(action)
        target_idx = action_target(action) if not is_chancellor_choice(action) else None
        target = f"player_{target_idx}" if target_idx is not None and target_idx < env.num_players else None
        guess = action_guess(action) if card == 1 else None
        game.add_log(decode_action_label(action, game.names, actor), "player" if actor == HUMAN else "normal")
        record_game_event(game, "action_played", {
            "actor": actor,
            "actor_name": game.names.get(actor, actor),
            "actor_policy_id": "human" if actor == HUMAN else game.ai_policy_ids.get(actor),
            "actor_policy_label": "Humain" if actor == HUMAN else POLICY_LABELS.get(game.ai_policy_ids.get(actor), "?"),
            "action": int(action),
            "card": int(card),
            "card_name": CARD_NAMES.get(card, "?"),
            "target": target,
            "target_name": game.names.get(target, target) if target else None,
            "guess": int(guess) if guess is not None else None,
            "guess_name": card_name(guess, True) if guess is not None else None,
            "label": decode_action_label(action, game.names, actor),
            "is_chancellor_choice": is_chancellor_choice(action),
            "state_before": replay_state_from_snapshot(game, prev),
            "state_after": replay_state_from_snapshot(game, after),
            "public_state_before": public_state_from_snapshot(game, prev),
            "public_state_after": public_state_from_snapshot(game, after),
        })
        if actor == HUMAN and not is_chancellor_choice(action) and action_card(action) == 2:
            target_idx = action_target(action)
            target = f"player_{target_idx}" if target_idx < env.num_players else None
            seen_card = prev["hands"].get(target, [None])[0] if target else None
            if target and seen_card is not None:
                note = {
                    "ts": now_stamp(),
                    "round_index": int(game.round_index),
                    "turn_index": int(game.turn_index),
                    "text": f"Secret: {game.names[target]} a {card_name(seen_card, True)}.",
                    "target": target,
                    "card": int(seen_card),
                }
                game.private_notes.append(note)
                game.add_log(note["text"], "secret")
    for row in consequence_logs(env, game.names, actor, action, prev):
        game.add_log(row["text"], row.get("tone", "normal"))
    finalize_round_if_needed(game)


def advance_ai_once(game: GameSession) -> None:
    if game.round_over or game.match_over:
        return
    env = game.env
    agent = env.agent_selection
    obs, _reward, terminated, truncated, _info = env.last()
    if terminated or truncated:
        env.step(None)
        finalize_round_if_needed(game)
        return
    if agent == HUMAN:
        return
    policy = game.ai_policies[agent]
    action = int(policy.act(env, obs, agent))
    apply_action(game, action)


def serialize_state(game: GameSession) -> dict:
    env = game.env
    current_agent = env.agent_selection
    valid_actions = []
    if (
        current_agent == HUMAN
        and not game.round_over
        and not game.match_over
        and not env.terminations.get(HUMAN, False)
    ):
        valid_actions = valid_action_payloads(env, HUMAN, game.names)
    players = []
    for agent in env.possible_agents:
        hand = list(env._hands.get(agent, []))
        # Le moteur PettingZoo met toutes les terminations a True quand la
        # manche est resolue. Pour l'UI, "alive" doit signifier "pas elimine
        # pendant la manche"; en fin de manche, les finalistes ont encore une
        # carte en main, les elimines non.
        alive = bool(hand) if game.round_over else (
            agent in env.agents and not env.terminations.get(agent, True)
        )
        players.append(
            {
                "id": agent,
                "name": game.names[agent],
                "is_human": agent == HUMAN,
                "policy_id": "human" if agent == HUMAN else game.ai_policy_ids.get(agent),
                "policy_label": "Humain" if agent == HUMAN else POLICY_LABELS.get(game.ai_policy_ids.get(agent), "?"),
                "alive": alive,
                "protected": bool(env._protected.get(agent, False)),
                "score": int(game.match_points.get(agent, 0)),
                "hand": [int(card) for card in hand] if agent == HUMAN else [],
                "hand_count": len(hand),
                "played": [int(card) for card in env._played_cards.get(agent, [])],
                "speech": game.speeches.get(agent),
            }
        )
    discard_top = game.discard_events[-1]["card"] if game.discard_events else None
    return {
        "game_id": game.game_id,
        "is_tutorial": bool(game.is_tutorial),
        "target_points": TARGET_POINTS,
        "round_index": game.round_index,
        "deck_size": len(env._deck),
        "discard_top": int(discard_top) if discard_top is not None else None,
        "current_agent": current_agent,
        "current_name": game.names.get(current_agent, current_agent),
        "can_human_act": current_agent == HUMAN and not game.round_over and not game.match_over,
        "round_over": game.round_over,
        "match_over": game.match_over,
        "replay_available": bool(game.match_over),
        "last_speaker": game.last_speaker,
        "players": players,
        "player_profile": public_player_profile(game.player_profile),
        "entry_reason_options": ENTRY_REASON_OPTIONS,
        "qadi_dialogue": game.qadi_dialogue[-16:],
        "ai_policy_ids": dict(game.ai_policy_ids),
        "available_policies": [
            {"id": policy_id, "label": label}
            for policy_id, label in POLICY_LABELS.items()
        ],
        "discard_events": game.discard_events[-80:],
        "valid_actions": valid_actions,
        "chancellor_pool": [int(card) for card in env._chancellor_pool],
        "logs": game.logs[-120:],
        "analytics_events": game.analytics_events[-80:],
        "match_stats_delta": game.match_stats_delta,
        "rules": RULES_TEXT,
    }


def build_replay_payload(game: GameSession) -> dict:
    leaders = [agent for agent, points in game.match_points.items() if points >= TARGET_POINTS]
    return {
        "game_id": game.game_id,
        "is_tutorial": bool(game.is_tutorial),
        "target_points": TARGET_POINTS,
        "round_count": int(game.round_index),
        "player_profile": public_player_profile(game.player_profile),
        "player_stats_delta": game.match_stats_delta,
        "ai_policy_ids": dict(game.ai_policy_ids),
        "players": [
            {
                "id": agent,
                "name": game.names.get(agent, agent),
                "is_human": agent == HUMAN,
                "policy_id": "human" if agent == HUMAN else game.ai_policy_ids.get(agent),
                "policy_label": "Humain" if agent == HUMAN else POLICY_LABELS.get(game.ai_policy_ids.get(agent), "?"),
                "final_score": int(game.match_points.get(agent, 0)),
                "winner": agent in leaders,
            }
            for agent in game.env.possible_agents
        ],
        "winners": [
            {"id": agent, "name": game.names.get(agent, agent)}
            for agent in leaders
        ],
        "human_won": HUMAN in leaders,
        "qadi_dialogue": game.qadi_dialogue[-16:],
        "events": game.structured_events,
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "champion": "champion_cbp", "policies": POLICY_LABELS}


@app.get("/api/policies")
def policies() -> dict:
    return {
        "policies": [
            {"id": policy_id, "label": label}
            for policy_id, label in POLICY_LABELS.items()
        ],
        "defaults": dict(DEFAULT_AI_POLICY_IDS),
    }


@app.get("/api/rules")
def rules() -> dict:
    return {"rules": RULES_TEXT}


@app.post("/api/player-profile/dialogue")
def player_profile_dialogue(request: PlayerProfileRequest) -> dict:
    profile = build_player_profile(request)
    requires_identity_confirmation = bool(profile.get("identity_found") and not request.identity_confirmed)
    needs_entry_reason = (
        not requires_identity_confirmation
        and not profile.get("secret_profile")
        and not profile.get("identity_found")
        and not request.entry_reason
    )
    dialogue = (
        build_identity_confirmation_dialogue(profile)
        if requires_identity_confirmation
        else build_intro_qadi_dialogue(profile, needs_entry_reason=needs_entry_reason)
    )
    return {
        "player_profile": public_player_profile(profile),
        "entry_reason_options": ENTRY_REASON_OPTIONS,
        "requires_identity_confirmation": requires_identity_confirmation,
        "requires_entry_reason": needs_entry_reason,
        "qadi_dialogue": dialogue,
    }


@app.post("/api/games")
def new_game(request: NewGameRequest) -> dict:
    profile = build_player_profile(request)
    ensure_player_record(profile)
    names = dict(DEFAULT_NAMES)
    names[HUMAN] = profile["display_name"] or DEFAULT_NAMES[HUMAN]
    game = GameSession(
        game_id=uuid4().hex,
        names=names,
        player_profile=profile,
        is_tutorial=bool(request.is_tutorial),
        qadi_dialogue=build_intro_qadi_dialogue(profile, needs_entry_reason=False),
        ai_policy_ids=normalize_ai_policy_ids(request.ai_policies),
    )
    start_round(game, reset_match=True)
    record_game_event(game, "game_created", {
        "entry_reason": profile["entry_reason"],
        "entry_reason_label": profile["entry_reason_label"],
        "entry_reason_signal": profile["entry_reason_signal"],
        "identity_confirmed": bool(profile.get("identity_confirmed")),
        "player_identity_id": profile.get("player_identity_id"),
        "is_tutorial": bool(game.is_tutorial),
        "state_after": replay_state_from_snapshot(game),
        "public_state_after": public_state_from_snapshot(game),
    })
    GAMES[game.game_id] = game
    return serialize_state(game)


@app.get("/api/games/{game_id}")
def state(game_id: str) -> dict:
    return serialize_state(get_game(game_id))


@app.get("/api/games/{game_id}/replay")
def replay(game_id: str) -> dict:
    game = get_game(game_id)
    with game.lock:
        if not game.match_over:
            raise HTTPException(status_code=409, detail="Replay disponible uniquement en fin de partie")
        return build_replay_payload(game)


@app.post("/api/games/{game_id}/play")
def play(game_id: str, request: PlayActionRequest) -> dict:
    game = get_game(game_id)
    with game.lock:
        if game.match_over or game.round_over:
            return serialize_state(game)
        if game.env.agent_selection != HUMAN:
            raise HTTPException(status_code=409, detail="Ce n'est pas le tour humain")
        action = int(request.action)
        if is_chancellor_choice(action) and not game.env._chancellor_pending:
            raise HTTPException(status_code=400, detail="Choix Vizir hors sequence")
        if not is_chancellor_choice(action) and game.env._chancellor_pending:
            raise HTTPException(status_code=400, detail="Le Vizir attend un choix")
        valid = {item["action"] for item in valid_action_payloads(game.env, HUMAN, game.names)}
        if action not in valid:
            raise HTTPException(status_code=400, detail="Action invalide")
        apply_action(game, action)
        return serialize_state(game)


@app.post("/api/games/{game_id}/ai-step")
def ai_step(game_id: str) -> dict:
    game = get_game(game_id)
    with game.lock:
        advance_ai_once(game)
        return serialize_state(game)


@app.post("/api/games/{game_id}/next-round")
def next_round(game_id: str) -> dict:
    game = get_game(game_id)
    with game.lock:
        if not game.round_over or game.match_over:
            return serialize_state(game)
        start_round(game, reset_match=False)
        return serialize_state(game)
