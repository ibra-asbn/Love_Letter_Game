from __future__ import annotations

import json
import html
import io
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from love_letter.engine import LoveLetterRLEnv  # noqa: E402
from love_letter.gameplay.play_vs_agent import snapshot_state, update_kept_card_after_action  # noqa: E402
from step7_self_play_league.league_policy import (  # noqa: E402
    LeaguePolicyFactory,
    LeagueRuntimeArgs,
    load_roster,
    policy_by_id,
)


HUMAN = "player_0"
MATCH_TARGET_POINTS = 2
PARIS_TZ = ZoneInfo("Europe/Paris")
WEB_LOG_DIR = REPO_ROOT / "love_letter_web" / "logs"
CARD_NAMES = {
    0: "Espionne",
    1: "Garde",
    2: "Pretre",
    3: "Baron",
    4: "Servante",
    5: "Prince",
    6: "Chancelier",
    7: "Roi",
    8: "Comtesse",
    9: "Princesse",
}
CARD_HINTS = {
    0: "Bonus si tu es seule a l'avoir jouee",
    1: "Devine une carte adverse",
    2: "Regarde une main adverse",
    3: "Compare ta carte gardee",
    4: "Protection jusqu'au prochain tour",
    5: "Force une defausse",
    6: "Garde une carte parmi le pool",
    7: "Echange les mains",
    8: "Obligatoire avec Roi ou Prince",
    9: "La jouer t'elimine",
}
DEFAULT_NAMES = {
    "player_0": "Ta soeur",
    "player_1": "Ariane",
    "player_2": "Basile",
    "player_3": "Clara",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ll-ink: #221f1f;
            --ll-muted: #6b6260;
            --ll-paper: #fffaf1;
            --ll-line: #ded0bd;
            --ll-red: #9f2936;
            --ll-gold: #b68832;
            --ll-green: #35685b;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(255,250,241,0.98), rgba(247,239,225,0.98)),
                repeating-linear-gradient(90deg, rgba(34,31,31,0.025) 0 1px, transparent 1px 9px);
            color: var(--ll-ink);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stButton"] button {
            border-radius: 8px;
            border: 1px solid var(--ll-line);
            background: #fffdf8;
            color: var(--ll-ink);
            min-height: 42px;
            box-shadow: 0 1px 0 rgba(34,31,31,0.06);
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--ll-red);
            color: var(--ll-red);
        }
        .ll-topbar {
            border: 1px solid var(--ll-line);
            background: rgba(255,253,248,0.86);
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .ll-kicker {
            color: var(--ll-red);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }
        .ll-subtle {
            color: var(--ll-muted);
            font-size: 0.9rem;
        }
        .ll-player {
            border: 1px solid var(--ll-line);
            background: #fffdf8;
            border-radius: 8px;
            padding: 12px;
            min-height: 138px;
        }
        .ll-turn-arrow {
            height: 24px;
            text-align: center;
            color: transparent;
            font-weight: 800;
            font-size: 0.88rem;
            line-height: 22px;
            margin-bottom: 2px;
        }
        .ll-turn-arrow.active {
            color: var(--ll-red);
        }
        .ll-speech {
            border: 1px solid rgba(182,136,50,0.45);
            background: #fff3d8;
            color: var(--ll-ink);
            border-radius: 8px;
            padding: 8px 10px;
            min-height: 44px;
            margin-bottom: 7px;
            font-size: 0.86rem;
            line-height: 1.25;
            box-shadow: 0 2px 0 rgba(34,31,31,0.05);
        }
        .ll-speech.empty {
            visibility: hidden;
        }
        .ll-speech.taunt {
            border-color: rgba(159,41,54,0.55);
            background: #ffe9df;
        }
        .ll-speech.good {
            border-color: rgba(53,104,91,0.45);
            background: #eaf5ef;
        }
        .ll-speech.bad {
            border-color: rgba(159,41,54,0.45);
            background: #f7e4e2;
        }
        .ll-player.active {
            border-color: var(--ll-gold);
            box-shadow: inset 0 0 0 1px rgba(182,136,50,0.25);
        }
        .ll-player.dead {
            opacity: 0.55;
        }
        .ll-name {
            font-weight: 750;
            margin-bottom: 4px;
        }
        .ll-status {
            color: var(--ll-muted);
            font-size: 0.86rem;
            margin-bottom: 9px;
        }
        .ll-cardline {
            color: var(--ll-ink);
            font-size: 0.91rem;
            line-height: 1.35;
        }
        .ll-hand-card {
            border: 1px solid var(--ll-line);
            background: linear-gradient(180deg, #fffdf9, #f7ead7);
            border-radius: 8px;
            padding: 14px;
            min-height: 104px;
        }
        .ll-card-value {
            color: var(--ll-red);
            font-size: 1.35rem;
            font-weight: 800;
        }
        .ll-card-name {
            font-size: 1.05rem;
            font-weight: 750;
        }
        .ll-card-hint {
            color: var(--ll-muted);
            font-size: 0.84rem;
            margin-top: 6px;
        }
        .ll-journal {
            border: 1px solid var(--ll-line);
            background: #181513;
            color: #f8efe0;
            border-radius: 8px;
            padding: 14px;
            min-height: 360px;
            max-height: 480px;
            overflow-y: auto;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.91rem;
            line-height: 1.55;
            white-space: pre-wrap;
        }
        .ll-log-muted {
            color: #b8aa99;
        }
        .ll-pill {
            display: inline-block;
            border: 1px solid var(--ll-line);
            border-radius: 999px;
            padding: 2px 9px;
            color: var(--ll-muted);
            font-size: 0.78rem;
            margin-right: 6px;
        }
        .ll-score-row {
            border: 1px solid var(--ll-line);
            background: #fffdf8;
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .ll-score-name {
            font-weight: 750;
        }
        .ll-score-points {
            float: right;
            color: var(--ll-red);
            font-weight: 800;
        }
        .px-stage {
            position: relative;
            width: min(100%, 700px);
            aspect-ratio: 1 / 1;
            margin: 0 auto 16px auto;
            border: 4px solid #1f1a18;
            outline: 4px solid #70523b;
            background:
                linear-gradient(45deg, rgba(255,255,255,0.10) 25%, transparent 25%) 0 0 / 18px 18px,
                linear-gradient(-45deg, rgba(0,0,0,0.06) 25%, transparent 25%) 0 0 / 18px 18px,
                #b7c06b;
            box-shadow: 0 8px 0 #342822, 0 16px 22px rgba(34,31,31,0.22);
            image-rendering: pixelated;
            overflow: hidden;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        .px-stage::before {
            content: "";
            position: absolute;
            inset: 18% 20% 14% 20%;
            border: 4px solid #7b5b3f;
            background:
                repeating-linear-gradient(0deg, rgba(255,255,255,0.08) 0 4px, transparent 4px 12px),
                #c89158;
            box-shadow: inset 0 0 0 4px rgba(57,38,27,0.25);
        }
        .px-seat {
            position: absolute;
            width: 23%;
            min-width: 92px;
            text-align: center;
            color: #241d1b;
            filter: saturate(0.98);
        }
        .px-seat.p1 { left: 8%; top: 15%; }
        .px-seat.p2 { left: 38.5%; top: 13%; }
        .px-seat.p3 { right: 8%; top: 15%; }
        .px-seat.p0 { left: 38.5%; bottom: 5%; }
        .px-seat.dead {
            opacity: 0.48;
            filter: grayscale(0.65);
        }
        .px-cursor {
            height: 17px;
            color: transparent;
            font-weight: 900;
            text-shadow: 2px 2px 0 #fff3d8;
        }
        .px-seat.active .px-cursor {
            color: #a82b38;
            animation: px-bob 0.8s steps(2, end) infinite;
        }
        .px-bubble {
            min-height: 38px;
            border: 2px solid #241d1b;
            background: #fff4cf;
            box-shadow: 3px 3px 0 #6f4c32;
            padding: 5px 6px;
            font-size: 0.66rem;
            line-height: 1.18;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .px-bubble.empty {
            display: none;
        }
        .px-bubble.taunt {
            background: #ffd8cb;
            box-shadow: 4px 4px 0 #7d2932;
        }
        .px-bubble.good {
            background: #dff2d2;
            box-shadow: 4px 4px 0 #35685b;
        }
        .px-bubble.bad {
            background: #ead5d7;
            box-shadow: 4px 4px 0 #7d2932;
        }
        .px-nameplate {
            display: inline-block;
            border: 3px solid #241d1b;
            background: #fff8df;
            box-shadow: 3px 3px 0 #6f4c32;
            padding: 2px 6px;
            font-size: 0.68rem;
            font-weight: 900;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .px-status {
            display: flex;
            justify-content: center;
            gap: 3px;
            flex-wrap: wrap;
            margin-top: 5px;
        }
        .px-dot {
            width: 11px;
            height: 11px;
            border: 2px solid #241d1b;
            background: #cfe7bf;
            box-shadow: 1px 1px 0 #6f4c32;
        }
        .px-dot.dead { background: #efc0bd; }
        .px-dot.protected { background: #bdd7e8; }
        .px-dot.point { background: #ffe08a; }
        .px-tag {
            border: 2px solid #241d1b;
            background: #f6e6b8;
            padding: 1px 4px;
            font-size: 0.55rem;
            font-weight: 800;
        }
        .px-tag.red { background: #efc0bd; }
        .px-tag.green { background: #cfe7bf; }
        .px-tag.blue { background: #bdd7e8; }
        .px-avatar {
            position: relative;
            width: 62px;
            height: 70px;
            margin: 0 auto 4px auto;
        }
        .px-shadow {
            position: absolute;
            left: 13px;
            bottom: 2px;
            width: 36px;
            height: 8px;
            background: rgba(36,29,27,0.24);
        }
        .px-body {
            position: absolute;
            left: 19px;
            top: 35px;
            width: 24px;
            height: 27px;
            border: 3px solid #241d1b;
            background: #386b61;
        }
        .px-seat.p2 .px-body { background: #b84b46; }
        .px-seat.p3 .px-body { background: #4e5d91; }
        .px-seat.p0 .px-body { background: #c59043; }
        .px-head {
            position: absolute;
            left: 15px;
            top: 13px;
            width: 32px;
            height: 28px;
            border: 3px solid #241d1b;
            background: #f2c99e;
        }
        .px-hair {
            position: absolute;
            left: 13px;
            top: 7px;
            width: 36px;
            height: 15px;
            border: 3px solid #241d1b;
            background: #45312b;
        }
        .px-seat.p1 .px-hair { background: #573421; }
        .px-seat.p2 .px-hair { background: #253047; }
        .px-seat.p3 .px-hair { background: #6b2d3e; }
        .px-seat.p0 .px-hair { background: #2d2a25; }
        .px-eye {
            position: absolute;
            top: 25px;
            width: 4px;
            height: 4px;
            background: #241d1b;
        }
        .px-eye.l { left: 24px; }
        .px-eye.r { right: 24px; }
        .px-mouth {
            position: absolute;
            left: 28px;
            top: 33px;
            width: 7px;
            height: 3px;
            background: #8f3a38;
        }
        .px-seat.p0 .px-eye,
        .px-seat.p0 .px-mouth {
            display: none;
        }
        .px-card-row {
            display: flex;
            justify-content: center;
            gap: 3px;
            margin-top: 3px;
            min-height: 22px;
            flex-wrap: wrap;
        }
        .px-card {
            min-width: 18px;
            height: 24px;
            border: 2px solid #241d1b;
            background: #fff8df;
            box-shadow: 2px 2px 0 #70523b;
            color: #a82b38;
            font-size: 0.68rem;
            font-weight: 900;
            line-height: 20px;
            text-align: center;
        }
        .px-card.back {
            background:
                linear-gradient(45deg, #a82b38 25%, transparent 25%) 0 0 / 8px 8px,
                #d7bb74;
            color: transparent;
        }
        .px-center {
            position: absolute;
            left: 50%;
            top: 2.5%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            align-items: center;
            text-align: center;
            z-index: 5;
        }
        .px-pile {
            border: 2px solid #241d1b;
            background: #fff4cf;
            box-shadow: 3px 3px 0 #6f4c32;
            padding: 4px 6px;
            min-width: 54px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .px-pile-title {
            font-size: 0.52rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 0;
        }
        @keyframes px-bob {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }
        @media (max-width: 760px) {
            .px-seat { width: 27%; min-width: 78px; }
            .px-avatar { transform: scale(0.78); margin-bottom: -8px; }
            .px-bubble { font-size: 0.56rem; min-height: 34px; padding: 4px; }
            .px-nameplate { font-size: 0.58rem; }
            .px-center { gap: 8px; }
            .px-pile { min-width: 54px; padding: 4px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Chargement du champion CBP...")
def get_policy_factory() -> LeaguePolicyFactory:
    return LeaguePolicyFactory(LeagueRuntimeArgs(device="cpu"))


@st.cache_data(show_spinner=False)
def get_champion_spec() -> dict:
    roster = load_roster()
    policies = policy_by_id(roster)
    return policies["champion_cbp"]


def name_of(agent: str, subject: bool = False) -> str:
    names = st.session_state.get("names", DEFAULT_NAMES)
    if agent == HUMAN and subject:
        return "Tu"
    return names.get(agent, agent)


def card_name(card: int | None, with_value: bool = False) -> str:
    if card is None:
        return "?"
    base = CARD_NAMES.get(int(card), "?")
    return f"{base} ({int(card)})" if with_value else base


def action_card(action: int) -> int:
    return int(action) // 100


def action_target(action: int) -> int:
    return (int(action) % 100) // 10


def action_guess(action: int) -> int:
    return int(action) % 10


def decode_action_label(action: int, actor: str | None = None) -> str:
    if action >= 900:
        text = "choisit avec le Chancelier"
        return f"{name_of(actor, subject=True)} {text}" if actor is not None else text
    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    text = f"joue {card_name(card, True)}"
    if card in {1, 2, 3, 5, 7}:
        target = f"player_{target_idx}" if target_idx < 4 else None
        if target is not None:
            text += f" sur {name_of(target)}"
    if card == 1:
        text += f" et annonce {card_name(guess, True)}"
    if actor is not None:
        return f"{name_of(actor, subject=True)} {text}"
    return text


def valid_actions(agent: str = HUMAN) -> list[int]:
    env = st.session_state.env
    obs_dict = env.observe(agent)
    return [int(action) for action in np.where(obs_dict["action_mask"] == 1)[0]]


def playable_cards(actions: list[int]) -> list[int]:
    cards = sorted({action_card(action) for action in actions if action < 900})
    hand_order = st.session_state.env._hands.get(HUMAN, [])
    return sorted(cards, key=lambda card: (hand_order.index(card) if card in hand_order else 99, card))


def pick_line(options: list[str], *keys: object) -> str:
    if not options:
        return ""
    total = 0
    for key in keys:
        for char in str(key):
            total = (total * 33 + ord(char)) % 100_000
    return options[total % len(options)]


def now_stamp() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def now_file_stamp() -> str:
    return datetime.now(PARIS_TZ).strftime("%Y%m%d_%H%M%S")


def ensure_log_dir() -> None:
    WEB_LOG_DIR.mkdir(parents=True, exist_ok=True)


def game_id() -> str:
    if "game_id" not in st.session_state or not st.session_state.game_id:
        st.session_state.game_id = f"streamlit_love_letter_{now_file_stamp()}"
    return st.session_state.game_id


def log_paths() -> tuple[Path, Path]:
    ensure_log_dir()
    gid = game_id()
    return WEB_LOG_DIR / f"{gid}.jsonl", WEB_LOG_DIR / f"{gid}_summary.json"


def append_log_event(event: str, payload: dict) -> None:
    path, _summary_path = log_paths()
    row = {
        "ts": now_stamp(),
        "event": event,
        "game_id": game_id(),
        "round_index": int(st.session_state.get("round_index", 0)),
        "seed": st.session_state.get("seed"),
        "active_agent": getattr(st.session_state.get("env", None), "agent_selection", None),
        "match_points": dict(st.session_state.get("match_points", {})),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary() -> None:
    _path, summary_path = log_paths()
    summary = {
        "updated_at": now_stamp(),
        "game_id": game_id(),
        "target_points": MATCH_TARGET_POINTS,
        "names": dict(st.session_state.get("names", {})),
        "match_points": dict(st.session_state.get("match_points", {})),
        "round_index": int(st.session_state.get("round_index", 0)),
        "match_over": bool(st.session_state.get("match_over", False)),
        "speeches": dict(st.session_state.get("speeches", {})),
        "logs": list(st.session_state.get("logs", [])),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def init_state(force_reset: bool = False) -> None:
    if force_reset or "env" not in st.session_state:
        st.session_state.env = LoveLetterRLEnv(num_players=4)
        st.session_state.names = st.session_state.get("names", DEFAULT_NAMES.copy())
        st.session_state.logs = []
        st.session_state.rendered_log_count = 0
        st.session_state.rewards = {agent: 0.0 for agent in st.session_state.env.possible_agents}
        st.session_state.match_points = {agent: 0 for agent in st.session_state.env.possible_agents}
        st.session_state.last_kept = {agent: None for agent in st.session_state.env.possible_agents}
        st.session_state.game_over = False
        st.session_state.round_over = False
        st.session_state.match_over = False
        st.session_state.action_builder = {}
        st.session_state.current_options = []
        st.session_state.ai_policies = {}
        st.session_state.seed = None
        st.session_state.round_index = 0
        st.session_state.next_starter = None
        st.session_state.game_id = f"streamlit_love_letter_{now_file_stamp()}"
        st.session_state.speeches = {}
        st.session_state.action_counter = 0
        st.session_state.last_action_visual = None
    else:
        agents = st.session_state.env.possible_agents
        st.session_state.setdefault("names", DEFAULT_NAMES.copy())
        st.session_state.setdefault("logs", [])
        st.session_state.setdefault("rendered_log_count", 0)
        st.session_state.setdefault("rewards", {agent: 0.0 for agent in agents})
        st.session_state.setdefault("match_points", {agent: 0 for agent in agents})
        st.session_state.setdefault("last_kept", {agent: None for agent in agents})
        st.session_state.setdefault("game_over", False)
        st.session_state.setdefault("round_over", False)
        st.session_state.setdefault("match_over", False)
        st.session_state.setdefault("action_builder", {})
        st.session_state.setdefault("current_options", [])
        st.session_state.setdefault("ai_policies", {})
        st.session_state.setdefault("seed", None)
        st.session_state.setdefault("round_index", 0)
        st.session_state.setdefault("next_starter", None)
        st.session_state.setdefault("game_id", f"streamlit_love_letter_{now_file_stamp()}")
        st.session_state.setdefault("speeches", {})
        st.session_state.setdefault("action_counter", 0)
        st.session_state.setdefault("last_action_visual", None)


def make_ai_policies() -> dict[str, object]:
    factory = get_policy_factory()
    champion = get_champion_spec()
    roles = {f"player_{idx}": "model" for idx in range(4)}
    return {
        agent: factory.make(champion, agent, roles)
        for agent in ["player_1", "player_2", "player_3"]
    }


def add_log(message: str, tone: str = "normal") -> None:
    entry = {"ts": now_stamp(), "text": message, "tone": tone}
    st.session_state.logs.append(entry)
    append_log_event("log", entry)
    write_summary()


def set_speech(agent: str, text: str, tone: str = "normal") -> None:
    if not text:
        return
    st.session_state.speeches[agent] = {
        "text": text,
        "tone": tone,
        "action_counter": int(st.session_state.get("action_counter", 0)),
        "expires_at": time.time() + 18.0,
    }


def current_speech(agent: str) -> dict | None:
    speech = st.session_state.get("speeches", {}).get(agent)
    if not speech:
        return None
    if "<" in str(speech.get("text", "")) and ">" in str(speech.get("text", "")):
        return None
    too_old = int(st.session_state.get("action_counter", 0)) - int(speech.get("action_counter", 0)) > 3
    expired = time.time() > float(speech.get("expires_at", 0.0))
    if too_old or expired:
        return None
    return speech


def public_hand_line(agent: str) -> str:
    played = st.session_state.env._played_cards.get(agent, [])
    if not played:
        return "Cartes jouees: aucune"
    return "Cartes jouees: " + ", ".join(card_name(card, True) for card in played)


def build_action_speech(env, agent_who_played: str, action: int | None, prev: dict) -> tuple[str, str]:
    if action is None:
        return "", "normal"
    if action >= 900:
        return pick_line(
            [
                "Je garde celle-ci. Les autres vont attendre dans l'ombre.",
                "Petit rangement de paquet. Rien d'inquietant.",
                "Je sais exactement ce que je veux revoir plus tard.",
            ],
            agent_who_played,
            action,
            len(st.session_state.logs),
        ), "normal"

    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    target = f"player_{target_idx}" if target_idx < env.num_players else None
    target_name = name_of(target) if target else ""
    is_ai = agent_who_played != HUMAN
    targets_human = target == HUMAN

    if card == 1 and target:
        old_card = prev["hands"].get(target, [None])[0]
        if old_card == guess:
            if targets_human and is_ai:
                return pick_line(
                    [
                        f"Je t'avais lue, {target_name}.",
                        f"{target_name}, cette carte etait beaucoup trop bruyante.",
                        "Merci pour l'indice, c'etait cadeau.",
                    ],
                    agent_who_played,
                    action,
                ), "taunt"
            return pick_line(
                [
                    f"Je savais que c'etait {card_name(guess)}.",
                    f"{target_name}, dehors.",
                    "Lecture parfaite.",
                ],
                agent_who_played,
                action,
            ), "good"
        if targets_human and is_ai:
            return pick_line(
                [
                    "Je rate, mais je te garde a l'oeil.",
                    f"Bien cache, {target_name}. Pour l'instant.",
                    "Hmm. Tu marques un point, pas la guerre.",
                ],
                agent_who_played,
                action,
            ), "taunt"
        return pick_line(
            [
                "Je tente. On verra bien.",
                "Pas cette carte ? Interessant.",
                "Je note la reponse.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 2 and target:
        if targets_human and is_ai:
            return pick_line(
                [
                    f"Montre-moi ta carte, {target_name}. Promis, je ne rirai presque pas.",
                    "Voyons ce que tu caches.",
                    f"{target_name}, ton secret m'interesse.",
                ],
                agent_who_played,
                action,
            ), "taunt"
        return pick_line(
            [
                f"Montre-moi ta carte, {target_name}.",
                f"Je veux savoir ce que cache {target_name}.",
                "Un peu d'information ne fait jamais de mal.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 3 and target:
        target_card = prev["hands"].get(target, [None])[0]
        actor_hand = prev["hands"].get(agent_who_played, [])
        compared = [c for c in actor_hand if c != 3]
        actor_card = compared[0] if compared else (actor_hand[0] if actor_hand else None)
        if actor_card is not None and target_card is not None and actor_card > target_card:
            if targets_human and is_ai:
                return pick_line(
                    [
                        f"Allez dehors, {target_name}.",
                        "Duel gagne. Merci d'etre passee.",
                        f"{target_name}, c'etait courageux. Pas suffisant.",
                    ],
                    agent_who_played,
                    action,
                ), "taunt"
            return pick_line(
                [
                    f"Allez dehors, {target_name} !",
                    "Plus fort. Plus simple.",
                    "Le duel est pour moi.",
                ],
                agent_who_played,
                action,
            ), "good"
        if actor_card is not None and target_card is not None and actor_card < target_card:
            return pick_line(
                [
                    "Oups. Mauvais duel.",
                    "Bon, celui-la etait un peu ambitieux.",
                    "Je me suis crue plus forte que ca.",
                ],
                agent_who_played,
                action,
            ), "bad"
        return pick_line(
            [
                "Egalite. On se regarde dans les yeux et on continue.",
                "Personne ne tombe. Pour l'instant.",
                "Match nul, mais pas inutile.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 4:
        return pick_line(
            [
                "Personne ne me touche ce tour.",
                "Je prends une petite protection.",
                "Vous pouvez regarder, pas toucher.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 5 and target:
        old_card = prev["hands"].get(target, [None])[0]
        if targets_human and is_ai:
            return pick_line(
                [
                    f"Jette-moi ca, {target_name}.",
                    "Allez, on change cette main.",
                    f"{target_name}, ta carte me deplait.",
                ],
                agent_who_played,
                action,
            ), "taunt"
        if target == agent_who_played:
            return pick_line(
                [
                    "Je me refais une main.",
                    "Cette carte ne me convient plus.",
                    "On melange un peu mon destin.",
                ],
                agent_who_played,
                action,
            ), "normal"
        if old_card == 9:
            return pick_line(
                [
                    f"Oh, une Princesse chez {target_name}. Quelle surprise.",
                    f"{target_name}, c'est fini.",
                    "La Princesse sort, et toi avec.",
                ],
                agent_who_played,
                action,
            ), "good"
        return pick_line(
            [
                f"{target_name}, change-moi cette carte.",
                "On ne va pas te laisser garder ca.",
                "Nouvelle carte, nouveau probleme.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 6:
        return pick_line(
            [
                "Je vais organiser l'avenir.",
                "Trois options, une seule bonne main.",
                "Le fond du paquet a aussi son importance.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 7 and target:
        if targets_human and is_ai:
            return pick_line(
                [
                    f"Je prends ta main, {target_name}. Merci.",
                    "Voyons si ta carte est plus jolie que la mienne.",
                    f"{target_name}, on echange. Ne fais pas cette tete.",
                ],
                agent_who_played,
                action,
            ), "taunt"
        return pick_line(
            [
                f"On echange, {target_name}.",
                "Ta main m'interesse soudain beaucoup.",
                "Un petit Roi pour renverser la table.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 8:
        return pick_line(
            [
                "Ce n'est pas forcement ce que tu crois.",
                "La Comtesse sait garder un secret.",
                "Je pose ca calmement. Trop calmement peut-etre.",
            ],
            agent_who_played,
            action,
        ), "normal"

    if card == 9:
        return pick_line(
            [
                "Bon. Ca, c'etait dramatique.",
                "La Princesse n'aurait jamais du sortir.",
                "Je vais pretendre que c'etait volontaire.",
            ],
            agent_who_played,
            action,
        ), "bad"

    return "", "normal"


def collect_consequence_logs(env, agent_who_played: str, action: int | None, prev: dict) -> list[tuple[str, str]]:
    if action is None:
        return []
    if action >= 900:
        if agent_who_played == HUMAN:
            hand = env._hands.get(HUMAN, [])
            if hand:
                return [(f"Tu gardes {card_name(hand[0], True)}.", "info")]
        return []

    card = action_card(action)
    target_idx = action_target(action)
    guess = action_guess(action)
    target = f"player_{target_idx}" if target_idx < env.num_players else None
    msgs: list[tuple[str, str]] = []
    effect_eliminated: set[str] = set()

    if card == 1 and target:
        target_was_alive = prev["alive"].get(target, False)
        old_hand = prev["hands"].get(target, [])
        old_card = old_hand[0] if old_hand else None
        if target_was_alive and old_card == guess:
            effect_eliminated.add(target)
            msgs.append((f"Le Garde touche: {name_of(target)} avait {card_name(old_card, True)}.", "good"))
        elif target_was_alive:
            msgs.append((f"Le Garde rate: {name_of(target)} n'avait pas {card_name(guess, True)}.", "muted"))

    if card == 2 and target and agent_who_played == HUMAN and env._hands.get(target):
        seen = env._hands[target][0]
        msgs.append((f"Le Pretre revele a toi seule: {name_of(target)} a {card_name(seen, True)}.", "info"))

    if card == 3 and target:
        target_card = prev["hands"].get(target, [None])[0]
        actor_hand = prev["hands"].get(agent_who_played, [])
        compared = [c for c in actor_hand if c != 3]
        actor_card = compared[0] if compared else (actor_hand[0] if actor_hand else None)
        actor_was_alive = prev["alive"].get(agent_who_played, False)
        target_was_alive = prev["alive"].get(target, False)
        if (
            actor_card is not None
            and target_card is not None
            and actor_was_alive
            and target_was_alive
            and actor_card > target_card
        ):
            effect_eliminated.add(target)
            msgs.append(
                (
                    f"Baron gagne pour {name_of(agent_who_played)}: "
                    f"{card_name(actor_card, True)} bat {card_name(target_card, True)}.",
                    "good" if agent_who_played == HUMAN else "normal",
                )
            )
        elif (
            actor_card is not None
            and target_card is not None
            and actor_was_alive
            and target_was_alive
            and actor_card < target_card
        ):
            effect_eliminated.add(agent_who_played)
            msgs.append(
                (
                    f"Baron perdu par {name_of(agent_who_played)}: "
                    f"{card_name(actor_card, True)} tombe contre {card_name(target_card, True)}.",
                    "bad" if agent_who_played == HUMAN else "normal",
                )
            )
        elif target and actor_card is not None and target_card is not None and actor_was_alive and target_was_alive:
            msgs.append((f"Baron: egalite entre {name_of(agent_who_played)} et {name_of(target)}.", "muted"))

    if card == 4:
        msgs.append((f"{name_of(agent_who_played)} est protegee jusqu'a son prochain tour.", "info"))

    if card == 5 and target:
        old_card = prev["hands"].get(target, [None])[0]
        target_was_alive = prev["alive"].get(target, False)
        if old_card is not None:
            msgs.append((f"{name_of(target)} defausse {card_name(old_card, True)}.", "normal"))
        if target_was_alive and old_card == 9:
            effect_eliminated.add(target)
            msgs.append((f"{name_of(target)} defausse la Princesse et sort de la manche.", "bad"))
        elif target == HUMAN and env._hands.get(HUMAN):
            msgs.append((f"Tu repioches {card_name(env._hands[HUMAN][0], True)}.", "info"))

    if card == 6:
        msgs.append((f"{name_of(agent_who_played)} ouvre un choix de Chancelier.", "info"))

    if card == 7 and target:
        if target == HUMAN or agent_who_played == HUMAN:
            new_card = env._hands.get(HUMAN, [None])[0]
            msgs.append((f"Apres le Roi, ta main est {card_name(new_card, True)}.", "info"))
        else:
            msgs.append((f"{name_of(agent_who_played)} et {name_of(target)} echangent leurs mains.", "normal"))

    if card == 9:
        effect_eliminated.add(agent_who_played)
        msgs.append((f"{name_of(agent_who_played)} joue ou defausse la Princesse et sort.", "bad"))

    for agent in env.possible_agents:
        if agent in effect_eliminated and prev["alive"].get(agent, False):
            already = any(
                name_of(agent) in msg and ("sort" in msg or "bat" in msg or "touche" in msg)
                for msg, _tone in msgs
            )
            if not already:
                msgs.append((f"{name_of(agent)} est eliminee.", "bad" if agent == HUMAN else "normal"))
    return msgs


def step_agent(action: int | None) -> None:
    env = st.session_state.env
    agent = env.agent_selection
    prev = snapshot_state(env)
    env.step(action)
    st.session_state.action_counter = int(st.session_state.get("action_counter", 0)) + 1
    speech_text, speech_tone = build_action_speech(env, agent, action, prev)
    set_speech(agent, speech_text, speech_tone)
    if action is not None:
        visual_card = 6 if action >= 900 else action_card(action)
        st.session_state.last_action_visual = {
            "agent": agent,
            "agent_name": name_of(agent),
            "card": int(visual_card),
            "label": card_name(visual_card, True),
            "action_label": decode_action_label(action, agent),
            "counter": int(st.session_state.action_counter),
        }

    for reward_agent in env.possible_agents:
        st.session_state.rewards[reward_agent] += float(env.rewards.get(reward_agent, 0.0))

    if action is not None:
        add_log(decode_action_label(action, agent), "player" if agent == HUMAN else "normal")

    for message, tone in collect_consequence_logs(env, agent, action, prev):
        add_log(message, tone)

    if agent != HUMAN and action is not None and action < 900:
        card_played = action_card(action)
        kept_before = st.session_state.last_kept.get(agent)
        if kept_before is not None:
            source = "son ancienne carte" if card_played == kept_before else "sa carte piochee ce tour"
            add_log(f"{name_of(agent)} a joue {source}.", "muted")

    update_kept_card_after_action(st.session_state.last_kept, agent, action, prev)
    append_log_event(
        "action",
        {
            "actor": agent,
            "actor_name": name_of(agent),
            "action": int(action) if action is not None else None,
            "action_label": decode_action_label(action, agent) if action is not None else "pass",
            "pre_hands": {a: [int(card) for card in cards] for a, cards in prev["hands"].items()},
            "post_hands": {
                a: [int(card) for card in env._hands.get(a, [])]
                for a in env.possible_agents
            },
            "played_cards": {
                a: [int(card) for card in env._played_cards.get(a, [])]
                for a in env.possible_agents
            },
            "terminations": {a: bool(env.terminations.get(a, False)) for a in env.possible_agents},
            "deck_size": int(len(env._deck)),
            "env_rewards": {a: float(env.rewards.get(a, 0.0)) for a in env.possible_agents},
            "speech": speech_text,
            "speech_tone": speech_tone,
        },
    )
    write_summary()


def finalize_if_needed() -> bool:
    env = st.session_state.env
    if not all(env.terminations.values()):
        return False
    if st.session_state.game_over:
        return True
    st.session_state.game_over = True
    st.session_state.round_over = True
    st.session_state.action_builder = {}
    winners = getattr(env, "_round_winners", [])
    spy = getattr(env, "_round_spy_winner", None)
    reason = getattr(env, "_round_win_reason", None)
    points_awarded = {agent: 0 for agent in env.possible_agents}
    if winners:
        reason_text = "dernier survivant" if reason == "last_alive" else "plus haute carte"
        names = ", ".join(name_of(agent) for agent in winners)
        add_log(f"Fin de manche: {names} marque pour {reason_text}.", "good")
        if reason == "highest_card":
            revealed = []
            for agent in env.possible_agents:
                hand = env._hands.get(agent, [])
                if hand:
                    revealed.append(f"{name_of(agent)} {card_name(hand[0], True)}")
            if revealed:
                add_log(f"Cartes finales revelees: {', '.join(revealed)}.", "info")
        for winner in winners:
            points_awarded[winner] += 1
            set_speech(winner, "Et voila, la manche est pour moi.", "good")
    else:
        add_log("Fin de manche.", "info")
    if spy:
        add_log(f"Bonus Espionne pour {name_of(spy)}.", "good" if spy == HUMAN else "normal")
        points_awarded[spy] += 1
        if spy not in winners:
            set_speech(spy, "Petit bonus Espionne. On prend.", "good")
    match_points = st.session_state.get(
        "match_points",
        {agent: 0 for agent in env.possible_agents},
    )
    for agent, points in points_awarded.items():
        match_points[agent] = int(match_points.get(agent, 0)) + int(points)
    st.session_state.match_points = match_points
    st.session_state.next_starter = winners[0] if winners else None
    st.session_state.match_over = any(
        points >= MATCH_TARGET_POINTS for points in match_points.values()
    )
    if any(points_awarded.values()):
        award_text = ", ".join(
            f"{name_of(agent)} +{points}" for agent, points in points_awarded.items() if points
        )
        add_log(f"Points de manche: {award_text}.", "info")
    score_text = ", ".join(
        f"{name_of(agent)} {match_points.get(agent, 0)}/{MATCH_TARGET_POINTS}"
        for agent in env.possible_agents
    )
    add_log(f"Score de partie: {score_text}.", "info")
    if st.session_state.match_over:
        leaders = [
            agent
            for agent, points in match_points.items()
            if points >= MATCH_TARGET_POINTS
        ]
        add_log(f"Partie terminee: {', '.join(name_of(agent) for agent in leaders)} gagne.", "good")
    append_log_event(
        "round_end",
        {
            "round_winners": list(winners),
            "spy_winner": spy,
            "points_awarded": points_awarded,
            "round_reason": reason,
            "match_over": bool(st.session_state.match_over),
        },
    )
    write_summary()
    return True


def advance_ai_once() -> None:
    env = st.session_state.env
    if finalize_if_needed():
        return
    agent = env.agent_selection
    obs_dict, _reward, terminated, truncated, _info = env.last()
    if terminated or truncated:
        env.step(None)
        finalize_if_needed()
        return
    if agent == HUMAN:
        st.session_state.action_builder = {}
        st.session_state.current_options = valid_actions(HUMAN)
        return
    policy = st.session_state.ai_policies[agent]
    action = int(policy.act(env, obs_dict, agent))
    step_agent(action)
    finalize_if_needed()


def configure_names_from_sidebar() -> dict[str, str]:
    human_name = st.session_state.get("human_name", DEFAULT_NAMES[HUMAN]).strip() or DEFAULT_NAMES[HUMAN]
    ai_1 = st.session_state.get("ai_1", DEFAULT_NAMES["player_1"]).strip() or DEFAULT_NAMES["player_1"]
    ai_2 = st.session_state.get("ai_2", DEFAULT_NAMES["player_2"]).strip() or DEFAULT_NAMES["player_2"]
    ai_3 = st.session_state.get("ai_3", DEFAULT_NAMES["player_3"]).strip() or DEFAULT_NAMES["player_3"]
    return {
        "player_0": human_name,
        "player_1": ai_1,
        "player_2": ai_2,
        "player_3": ai_3,
    }


def start_round(reset_match: bool = False) -> None:
    names = configure_names_from_sidebar()
    match_points = (
        {agent: 0 for agent in [f"player_{idx}" for idx in range(4)]}
        if reset_match
        else dict(st.session_state.get("match_points", {}))
    )
    if not match_points:
        match_points = {agent: 0 for agent in [f"player_{idx}" for idx in range(4)]}
    next_starter = None if reset_match else st.session_state.get("next_starter")
    round_index = 1 if reset_match else int(st.session_state.get("round_index", 0)) + 1
    existing_game_id = None if reset_match else st.session_state.get("game_id")

    init_state(force_reset=True)
    if existing_game_id:
        st.session_state.game_id = existing_game_id
    st.session_state.names = names
    st.session_state.match_points = match_points
    st.session_state.round_index = round_index
    st.session_state.next_starter = next_starter
    st.session_state.seed = int(time.time()) % 2_000_000_000
    options = {"tokens": dict(st.session_state.match_points)}
    if next_starter is not None:
        options["starting_agent"] = next_starter
    st.session_state.env.reset(seed=st.session_state.seed, options=options)
    st.session_state.ai_policies = make_ai_policies()
    append_log_event(
        "round_start",
        {
            "names": dict(st.session_state.names),
            "target_points": MATCH_TARGET_POINTS,
            "starting_agent": st.session_state.env.agent_selection,
            "reset_match": bool(reset_match),
        },
    )
    add_log(f"Manche {st.session_state.round_index} contre champion_cbp.", "info")
    add_log(f"Premier joueur: {name_of(st.session_state.env.agent_selection)}.", "info")


def start_new_game() -> None:
    start_round(reset_match=True)


def start_next_round() -> None:
    start_round(reset_match=False)


def render_header() -> None:
    st.markdown(
        f"""
        <div class="ll-topbar">
          <div class="ll-kicker">Love Letter contre champion_cbp</div>
          <div style="font-size:2rem;font-weight:800;line-height:1.15;margin-top:4px;">Partie en {MATCH_TARGET_POINTS} points.</div>
          <div class="ll-subtle" style="margin-top:6px;">
            Un point pour la victoire de manche, un point bonus Espionne possible. Avec Espionne + victoire, on peut gagner directement.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_player(agent: str) -> None:
    env = st.session_state.env
    alive = agent in env.agents and not env.terminations.get(agent, True)
    active = env.agent_selection == agent and not st.session_state.game_over
    classes = "ll-player"
    if active:
        classes += " active"
    if not alive:
        classes += " dead"
    protected = "Protegee" if env._protected.get(agent, False) else "Ciblable"
    status = "En jeu" if alive else "Eliminee"
    if agent == HUMAN and env._hands.get(agent):
        hidden = "Main visible"
    elif alive:
        hidden = "Main cachee"
    else:
        hidden = "Hors manche"
    arrow_class = "ll-turn-arrow active" if active else "ll-turn-arrow"
    arrow_text = "&darr; Tour" if active else "&nbsp;"
    speech = current_speech(agent)
    if speech:
        speech_class = f"ll-speech {html.escape(str(speech.get('tone', 'normal')))}"
        speech_html = f'<div class="{speech_class}">{html.escape(str(speech.get("text", "")))}</div>'
    else:
        speech_html = '<div class="ll-speech empty">&nbsp;</div>'
    st.markdown(
        f"""
        <div class="{arrow_class}">{arrow_text}</div>
        {speech_html}
        <div class="{classes}">
          <div class="ll-name">{html.escape(name_of(agent))}</div>
          <div class="ll-status">{status} · {protected} · {hidden}</div>
          <div class="ll-cardline">{html.escape(public_hand_line(agent))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pixel_card_html(card: int | None, face_down: bool = False) -> str:
    if face_down:
        return '<span class="px-card back">?</span>'
    return f'<span class="px-card">{html.escape(str(card)) if card is not None else "?"}</span>'


def pixel_speech_html(agent: str) -> str:
    speech = current_speech(agent)
    if not speech:
        return '<div class="px-bubble empty">&nbsp;</div>'
    tone = html.escape(str(speech.get("tone", "normal")))
    text = html.escape(str(speech.get("text", "")))
    return f'<div class="px-bubble {tone}">{text}</div>'


def pixel_avatar_html() -> str:
    return """
    <div class="px-avatar">
      <div class="px-shadow"></div>
      <div class="px-body"></div>
      <div class="px-head"></div>
      <div class="px-hair"></div>
      <div class="px-eye l"></div>
      <div class="px-eye r"></div>
      <div class="px-mouth"></div>
    </div>
    """


def render_pixel_seat(agent: str, position_class: str) -> str:
    env = st.session_state.env
    alive = agent in env.agents and not env.terminations.get(agent, True)
    active = env.agent_selection == agent and not st.session_state.game_over
    last_visual = st.session_state.get("last_action_visual") or {}
    is_last = last_visual.get("agent") == agent
    classes = ["px-seat", position_class]
    if active:
        classes.append("active")
    if not alive:
        classes.append("dead")
    if is_last:
        classes.append("last")

    played_cards = env._played_cards.get(agent, [])[-5:]
    if played_cards:
        played_html = "".join(pixel_card_html(card) for card in played_cards)
    else:
        played_html = ""

    hand = env._hands.get(agent, [])
    if agent == HUMAN:
        hand_html = "".join(pixel_card_html(card) for card in hand)
    else:
        hand_html = "".join(pixel_card_html(None, face_down=True) for _ in hand)

    points = int(st.session_state.get("match_points", {}).get(agent, 0))
    protected = bool(env._protected.get(agent, False))
    life_dot = '<span class="px-dot"></span>' if alive else '<span class="px-dot dead"></span>'
    protection_dot = '<span class="px-dot protected"></span>' if protected else ""
    point_dots = "".join('<span class="px-dot point"></span>' for _ in range(points))
    cursor = "▼" if active else "&nbsp;"
    return f"""
    <div class="{' '.join(classes)}">
      <div class="px-cursor">{cursor}</div>
      {pixel_speech_html(agent)}
      {pixel_avatar_html()}
      <div class="px-nameplate">{html.escape(name_of(agent))}</div>
      <div class="px-status">
        {life_dot}
        {protection_dot}
        {point_dots}
      </div>
      <div class="px-card-row">{hand_html}</div>
      <div class="px-card-row">{played_html}</div>
    </div>
    """


def render_pixel_stage() -> None:
    env = st.session_state.env
    deck_cards = pixel_card_html(None, face_down=True) if env._deck else '<span class="px-tag red">vide</span>'
    all_played = []
    for agent in env.possible_agents:
        all_played.extend(env._played_cards.get(agent, []))
    discard_card = all_played[-1] if all_played else None
    discard_html = pixel_card_html(discard_card) if discard_card is not None else '<span class="px-tag">aucune</span>'

    stage = f"""
    <div class="px-stage">
      {render_pixel_seat("player_1", "p1")}
      {render_pixel_seat("player_2", "p2")}
      {render_pixel_seat("player_3", "p3")}
      <div class="px-center">
        <div class="px-pile">
          <div class="px-pile-title">Pioche {len(env._deck)}</div>
          {deck_cards}
        </div>
        <div class="px-pile">
          <div class="px-pile-title">Defausse</div>
          {discard_html}
        </div>
      </div>
      {render_pixel_seat(HUMAN, "p0")}
    </div>
    """
    st.markdown(stage, unsafe_allow_html=True)


def render_table() -> None:
    env = st.session_state.env
    top_cols = st.columns([1, 1, 1, 1])
    metrics = [
        ("Pioche", str(len(env._deck))),
        ("Manche", str(st.session_state.get("round_index", 1))),
        ("Tour", name_of(env.agent_selection) if not st.session_state.game_over else "Manche finie"),
        ("Objectif", f"{MATCH_TARGET_POINTS} points"),
    ]
    for col, (label, value) in zip(top_cols, metrics):
        col.metric(label, value)

    st.markdown("### Table")
    render_pixel_stage()


def render_hand() -> None:
    env = st.session_state.env
    hand = env._hands.get(HUMAN, [])
    st.markdown("### Ta main")
    st.markdown(
        f'<span class="ll-pill">Cartes restantes dans la pioche: {len(env._deck)}</span>',
        unsafe_allow_html=True,
    )
    if not hand:
        st.info("Tu n'as plus de carte en main.")
        return
    cols = st.columns(max(1, len(hand)))
    for col, card in zip(cols, hand):
        with col:
            st.markdown(
                f"""
                <div class="ll-hand-card">
                  <div class="ll-card-value">{card}</div>
                  <div class="ll-card-name">{html.escape(card_name(card))}</div>
                  <div class="ll-card-hint">{html.escape(CARD_HINTS.get(card, ""))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def action_button(label: str, key: str, data: dict | None = None) -> bool:
    clicked = st.button(label, key=key, use_container_width=True)
    if clicked and data:
        st.session_state.action_builder.update(data)
        st.rerun()
    return clicked


def submit_human_action(action: int) -> None:
    st.session_state.action_builder = {}
    step_agent(int(action))
    finalize_if_needed()
    st.rerun()


def render_chancellor_builder(actions: list[int]) -> None:
    env = st.session_state.env
    pool = list(env._chancellor_pool)
    builder = st.session_state.action_builder
    st.markdown("### Chancelier")
    st.markdown(
        " ".join(f'<span class="ll-pill">{idx + 1}: {html.escape(card_name(card, True))}</span>' for idx, card in enumerate(pool)),
        unsafe_allow_html=True,
    )
    if len(pool) <= 1:
        if st.button(f"Garder {card_name(pool[0], True)}", use_container_width=True):
            submit_human_action(actions[0])
        return
    if len(pool) == 2:
        cols = st.columns(2)
        for idx, action in enumerate(sorted(actions)):
            label = f"Garder {card_name(pool[idx], True)}"
            if cols[idx].button(label, key=f"chancellor_keep_{idx}", use_container_width=True):
                submit_human_action(action)
        return

    keep_idx = builder.get("keep_idx")
    if keep_idx is None:
        cols = st.columns(3)
        for idx, card in enumerate(pool):
            if cols[idx].button(f"Garder {card_name(card, True)}", key=f"keep_{idx}", use_container_width=True):
                st.session_state.action_builder = {"keep_idx": idx}
                st.rerun()
        return

    returned = [card for idx, card in enumerate(pool) if idx != keep_idx]
    st.markdown(f"Carte gardee: **{card_name(pool[keep_idx], True)}**")
    normal_action = 900 + keep_idx * 2
    reverse_action = normal_action + 1
    col1, col2, col3 = st.columns([1, 1, 0.65])
    if col1.button(
        f"Fond: {card_name(returned[0], True)} puis {card_name(returned[1], True)}",
        key="chancellor_order_normal",
        use_container_width=True,
    ):
        submit_human_action(normal_action)
    if col2.button(
        f"Fond: {card_name(returned[1], True)} puis {card_name(returned[0], True)}",
        key="chancellor_order_reverse",
        use_container_width=True,
    ):
        submit_human_action(reverse_action)
    if col3.button("Retour", key="chancellor_back", use_container_width=True):
        st.session_state.action_builder = {}
        st.rerun()


def target_label(target_idx: int) -> str:
    if target_idx >= 4:
        return "Sans cible"
    agent = f"player_{target_idx}"
    return "Toi" if agent == HUMAN else name_of(agent)


def render_normal_action_builder(actions: list[int]) -> None:
    builder = st.session_state.action_builder
    card = builder.get("card")
    st.markdown("### Action")

    if card is None:
        cards = playable_cards(actions)
        if not cards:
            st.warning("Aucune action disponible.")
            return
        cols = st.columns(min(4, max(1, len(cards))))
        for idx, playable in enumerate(cards):
            label = f"{card_name(playable, True)}"
            hint = CARD_HINTS.get(playable, "")
            if cols[idx % len(cols)].button(f"{label}\n{hint}", key=f"pick_card_{playable}", use_container_width=True):
                st.session_state.action_builder = {"card": playable}
                st.rerun()
        return

    card_actions = [action for action in actions if action < 900 and action_card(action) == card]
    if not card_actions:
        st.session_state.action_builder = {}
        st.rerun()
    st.markdown(f"Carte choisie: **{card_name(card, True)}**")

    if card in {1, 2, 3, 5, 7}:
        target = builder.get("target")
        targets = sorted({action_target(action) for action in card_actions})
        if target is None:
            cols = st.columns(min(4, max(1, len(targets))))
            for idx, target_idx in enumerate(targets):
                if cols[idx % len(cols)].button(
                    target_label(target_idx),
                    key=f"target_{card}_{target_idx}",
                    use_container_width=True,
                ):
                    st.session_state.action_builder.update({"target": target_idx})
                    st.rerun()
            if st.button("Changer de carte", key="back_card_from_target", use_container_width=True):
                st.session_state.action_builder = {}
                st.rerun()
            return

        target_actions = [action for action in card_actions if action_target(action) == target]
        if card == 1:
            guesses = sorted({action_guess(action) for action in target_actions})
            st.markdown(f"Cible: **{target_label(target)}**")
            cols = st.columns(3)
            for idx, guess in enumerate(guesses):
                if cols[idx % 3].button(
                    card_name(guess, True),
                    key=f"guess_{target}_{guess}",
                    use_container_width=True,
                ):
                    submit_human_action(next(action for action in target_actions if action_guess(action) == guess))
            if st.button("Retour a la cible", key="back_target_guard", use_container_width=True):
                st.session_state.action_builder = {"card": card}
                st.rerun()
            return

        if len(target_actions) == 1:
            action = target_actions[0]
            if st.button(f"Jouer sur {target_label(target)}", key=f"submit_{action}", use_container_width=True):
                submit_human_action(action)
            if st.button("Retour a la cible", key=f"back_target_{card}", use_container_width=True):
                st.session_state.action_builder = {"card": card}
                st.rerun()
            return

    if len(card_actions) == 1:
        action = card_actions[0]
        if st.button(f"Jouer {card_name(card, True)}", key=f"submit_{action}", use_container_width=True):
            submit_human_action(action)
    else:
        for action in card_actions:
            if st.button(decode_action_label(action), key=f"submit_{action}", use_container_width=True):
                submit_human_action(action)
    if st.button("Changer de carte", key="back_card_simple", use_container_width=True):
        st.session_state.action_builder = {}
        st.rerun()


def render_actions() -> None:
    env = st.session_state.env
    if st.session_state.get("match_over", False):
        match_points = st.session_state.get("match_points", {})
        leaders = [
            agent
            for agent, points in match_points.items()
            if points >= MATCH_TARGET_POINTS
        ]
        st.success(f"Partie terminee: {', '.join(name_of(agent) for agent in leaders)} gagne.")
        if st.button("Nouvelle partie", key="new_match_from_actions", use_container_width=True):
            start_new_game()
            st.rerun()
        return
    if st.session_state.game_over:
        st.info("La manche est terminee.")
        if st.button("Manche suivante", key="next_round_from_actions", use_container_width=True):
            start_next_round()
            st.rerun()
        return
    if env.agent_selection != HUMAN:
        agent = env.agent_selection
        st.info(f"Tour de {name_of(agent)}.")
        st.caption("Les IA avancent maintenant une action à la fois pour que le journal reste lisible.")
        if st.button(f"Faire jouer {name_of(agent)}", key=f"ai_step_{agent}_{st.session_state.get('action_counter', 0)}", use_container_width=True):
            advance_ai_once()
            st.rerun()
        return
    actions = valid_actions(HUMAN)
    st.session_state.current_options = actions
    if env._chancellor_pending:
        render_chancellor_builder([action for action in actions if action >= 900])
    else:
        render_normal_action_builder(actions)


def render_rewards() -> None:
    match_points = st.session_state.get(
        "match_points",
        {agent: 0 for agent in st.session_state.env.possible_agents},
    )
    st.sidebar.markdown(f"### Score partie / {MATCH_TARGET_POINTS}")
    for agent in st.session_state.env.possible_agents:
        points = int(match_points.get(agent, 0))
        st.sidebar.markdown(
            f"""
            <div class="ll-score-row">
              <span class="ll-score-name">{html.escape(name_of(agent))}</span>
              <span class="ll-score-points">{points}/{MATCH_TARGET_POINTS}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def current_logs_markdown() -> str:
    match_points = st.session_state.get("match_points", {})
    lines = [
        "# Love Letter Streamlit Log",
        "",
        f"Game id: `{game_id()}`",
        f"Date export: {now_stamp()}",
        f"Objectif: {MATCH_TARGET_POINTS} points",
        "",
        "## Joueurs",
        "",
    ]
    for agent in st.session_state.env.possible_agents:
        lines.append(f"- {agent}: {name_of(agent)}")
    lines.extend(["", "## Score", ""])
    for agent in st.session_state.env.possible_agents:
        lines.append(f"- {name_of(agent)}: {match_points.get(agent, 0)}/{MATCH_TARGET_POINTS}")
    lines.extend(["", "## Journal", ""])
    for entry in st.session_state.logs:
        lines.append(f"- [{entry.get('ts', '')}] {entry.get('text', '')}")
    return "\n".join(lines)


def render_log_downloads() -> None:
    path, summary_path = log_paths()
    all_log_files = sorted(
        WEB_LOG_DIR.glob("streamlit_love_letter_*"),
        key=lambda file_path: file_path.stat().st_mtime,
        reverse=True,
    )
    st.sidebar.markdown("### Logs")
    if path.exists():
        st.sidebar.download_button(
            "Télécharger JSONL",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/jsonl",
            use_container_width=True,
        )
    if summary_path.exists():
        st.sidebar.download_button(
            "Télécharger résumé JSON",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="application/json",
            use_container_width=True,
        )
    st.sidebar.download_button(
        "Télécharger journal Markdown",
        data=current_logs_markdown().encode("utf-8"),
        file_name=f"{game_id()}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if all_log_files:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in all_log_files:
                zip_file.write(file_path, arcname=file_path.name)
        st.sidebar.download_button(
            "Télécharger toutes les parties",
            data=archive.getvalue(),
            file_name=f"love_letter_streamlit_logs_{now_file_stamp()}.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.sidebar.caption(f"{len(all_log_files)} fichiers archivés dans `{WEB_LOG_DIR}`.")
    st.sidebar.caption(f"Sauvegarde auto: `{path}`")


def tone_class(tone: str) -> str:
    if tone == "muted":
        return "ll-log-muted"
    return ""


def journal_html(entries: list[dict], partial: str | None = None) -> str:
    lines = []
    for item in entries:
        css = tone_class(item.get("tone", "normal"))
        text = html.escape(item["text"])
        if css:
            lines.append(f'<span class="{css}">{text}</span>')
        else:
            lines.append(text)
    if partial is not None:
        lines.append(html.escape(partial))
    return '<div class="ll-journal">' + "\n".join(lines) + "</div>"


def render_journal() -> None:
    st.markdown("### Journal")
    logs = st.session_state.logs
    total = len(logs)
    seen = min(st.session_state.get("rendered_log_count", 0), total)
    start = max(0, total - 90)
    stable_until = max(start, seen)
    stable = logs[start:stable_until]
    new = logs[stable_until:]
    placeholder = st.empty()

    if not new:
        placeholder.markdown(journal_html(logs[start:]), unsafe_allow_html=True)
        st.session_state.rendered_log_count = total
        return

    current = stable[:]
    placeholder.markdown(journal_html(current), unsafe_allow_html=True)
    for item in new:
        text = item["text"]
        step = 2 if len(text) < 90 else 4
        for idx in range(0, len(text) + 1, step):
            placeholder.markdown(journal_html(current, text[:idx]), unsafe_allow_html=True)
            time.sleep(0.006)
        current.append(item)
        placeholder.markdown(journal_html(current), unsafe_allow_html=True)
    st.session_state.rendered_log_count = total


def main() -> None:
    st.set_page_config(page_title="Love Letter contre champion_cbp", layout="wide")
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown("## Partie")
        st.text_input("Nom humain", value=st.session_state.names[HUMAN], key="human_name")
        st.text_input("IA 1", value=st.session_state.names["player_1"], key="ai_1")
        st.text_input("IA 2", value=st.session_state.names["player_2"], key="ai_2")
        st.text_input("IA 3", value=st.session_state.names["player_3"], key="ai_3")
        if st.button("Nouvelle partie", use_container_width=True):
            start_new_game()
            st.rerun()
        if st.session_state.get("round_over", False) and not st.session_state.get("match_over", False):
            if st.button("Manche suivante", key="next_round_sidebar", use_container_width=True):
                start_next_round()
                st.rerun()
        render_rewards()
        st.markdown("---")
        render_log_downloads()
        st.markdown("---")
        st.caption("Mode: champion_cbp, 4 joueurs, partie courte en 2 points.")

    if not st.session_state.logs:
        start_new_game()

    render_header()

    table_col, control_col = st.columns([1.2, 0.8], gap="large")
    with table_col:
        render_table()
    with control_col:
        render_hand()
        render_actions()
        render_journal()


if __name__ == "__main__":
    main()
