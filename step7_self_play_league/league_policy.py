"""Policy factory and roster helpers for the Step7 self-play league."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interlude_heuristic_comparison.evaluate_rotating_tactical_arena import GenericAdvantageSeat, RelativeModelSeat
from love_letter.bots.heuristic import HeuristicBot
from step2_rl_finetune.common import now_stamp, resolve_checkpoint
from step3_action_value.evaluate_advantage_head_v2 import load_advantage_bundle
from step5_execution_heads.chancellor_head import load_chancellor_head
from step5_execution_heads.evaluate_combined_three_heads import Step5ThreeSeat


STEP_DIR = PROJECT_ROOT / "step7_self_play_league"
DEFAULT_ROSTER_PATH = STEP_DIR / "league_roster.json"
DEFAULT_RESULTS_PATH = STEP_DIR / "league_results.jsonl"
DEFAULT_PROMOTION_PATH = STEP_DIR / "promotion_history.jsonl"


@dataclass
class LeagueRuntimeArgs:
    device: str = "cpu"
    override_margin: float = 0.10
    max_actions: int | None = None
    verify_rollouts: int = 0
    verify_min_win_delta: float = 0.125
    verify_min_score_delta: float = 0.05
    verify_t_threshold: float = 0.75
    chancellor_margin: float = 0.10
    retarget_margin: float = 0.10
    veto_score: float = 0.05
    force_score: float = 0.32
    self_force_score: float = 0.55
    min_princess_prob: float = 0.24
    example_limit: int = 20

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


class HeuristicSeat:
    def __init__(self):
        self.bot = HeuristicBot(shuffle_targets=True)

    def act(self, env, obs_dict, agent: str) -> int:
        return int(self.bot.choose_action(env, agent))


def load_roster(path: str | Path = DEFAULT_ROSTER_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_roster(roster: dict, path: str | Path = DEFAULT_ROSTER_PATH) -> None:
    Path(path).write_text(json.dumps(roster, indent=2, ensure_ascii=False), encoding="utf-8")


def policy_by_id(roster: dict) -> dict[str, dict]:
    return {policy["policy_id"]: policy for policy in roster["policies"]}


def active_policies(roster: dict) -> list[dict]:
    return [policy for policy in roster["policies"] if policy.get("active", False)]


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LeaguePolicyFactory:
    """Creates fresh stateful policy seats from roster entries."""

    def __init__(self, runtime_args: LeagueRuntimeArgs | SimpleNamespace | None = None):
        if runtime_args is None:
            runtime_args = LeagueRuntimeArgs()
        if isinstance(runtime_args, LeagueRuntimeArgs):
            runtime_args = runtime_args.namespace()
        self.args = runtime_args
        self._advantage_cache = {}
        self._chancellor_cache = {}

    def make(self, spec: dict, agent: str, roles: dict[str, str] | None = None):
        kind = spec["kind"]
        if kind == "heuristic":
            return HeuristicSeat()
        if kind == "checkpoint":
            return RelativeModelSeat(resolve_checkpoint(spec["checkpoint"]))
        if kind == "composite":
            return self._make_composite(spec, agent, roles or {})
        raise ValueError(f"Unknown policy kind: {kind}")

    def _advantage_bundle(self, checkpoint_name: str):
        key = str(checkpoint_name)
        if key not in self._advantage_cache:
            self._advantage_cache[key] = load_advantage_bundle(checkpoint_name, None)
        return self._advantage_cache[key]

    def _chancellor_head(self, path: str | None):
        if not path:
            return None
        key = str(path)
        if key not in self._chancellor_cache:
            head_path = Path(path)
            if not head_path.exists():
                head_path = PROJECT_ROOT / path
            self._chancellor_cache[key] = load_chancellor_head(head_path, self.args.device)[0]
        return self._chancellor_cache[key]

    def _make_composite(self, spec: dict, agent: str, roles: dict[str, str]):
        checkpoint, default_base, head, ckpt = self._advantage_bundle(spec["advantage_checkpoint"])
        base_checkpoint = resolve_checkpoint(spec.get("base_checkpoint") or default_base)
        args = deepcopy(self.args)
        args.max_actions = args.max_actions or int(ckpt.get("max_actions", 14))
        base_policy = GenericAdvantageSeat(
            base_checkpoint,
            head,
            ckpt.get("categories", []),
            args,
            roles,
            agent,
        )
        if not any(spec.get(flag, False) for flag in ["use_chancellor", "use_baron", "use_prince"]):
            return base_policy
        return Step5ThreeSeat(
            base_policy,
            self._chancellor_head(spec.get("chancellor_head")),
            args,
            use_chancellor=bool(spec.get("use_chancellor", False)),
            use_baron=bool(spec.get("use_baron", False)),
            use_prince=bool(spec.get("use_prince", False)),
        )


def make_candidate_roster_entry(
    policy_id: str,
    checkpoint: str,
    parent_id: str,
    advantage_checkpoint: str = "step3_advantage_v2_dagger_attempt1_iter1.pth",
    chancellor_head: str = "step5_execution_heads/cards/chancellor/checkpoints/chancellor_head_v1.pth",
    elo: float = 1500.0,
    active: bool = False,
) -> dict:
    return {
        "policy_id": policy_id,
        "kind": "composite",
        "active": active,
        "elo": float(elo),
        "games": 0,
        "created_at": now_stamp(),
        "parent_id": parent_id,
        "base_checkpoint": checkpoint,
        "advantage_checkpoint": advantage_checkpoint,
        "chancellor_head": chancellor_head,
        "use_advantage": True,
        "use_chancellor": True,
        "use_baron": True,
        "use_prince": True,
    }


def upsert_policy(roster: dict, entry: dict) -> dict:
    policies = [policy for policy in roster["policies"] if policy["policy_id"] != entry["policy_id"]]
    policies.append(entry)
    roster["policies"] = policies
    return roster


def apply_promotion(roster: dict, candidate_id: str, max_active: int | None = None) -> tuple[dict, str | None]:
    max_active = int(max_active or roster.get("max_active", 5))
    policies = policy_by_id(roster)
    if candidate_id not in policies:
        raise ValueError(f"Candidate not in roster: {candidate_id}")
    policies[candidate_id]["active"] = True
    active = [policy for policy in roster["policies"] if policy.get("active", False)]
    dropped_id = None
    if len(active) > max_active:
        weakest = min(active, key=lambda item: (float(item.get("elo", 1500.0)), int(item.get("games", 0)), item["policy_id"]))
        weakest["active"] = False
        dropped_id = weakest["policy_id"]
    return roster, dropped_id

