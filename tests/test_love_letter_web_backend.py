from fastapi.testclient import TestClient
import pytest

from love_letter_web.backend import main as backend


client = TestClient(backend.app)


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(backend, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(backend, "GAME_EVENTS_PATH", logs_dir / "game_events.jsonl")
    monkeypatch.setattr(backend, "PLAYER_STATS_PATH", logs_dir / "player_stats.json")
    backend.GAMES.clear()


def test_amira_action_is_not_treated_as_chancellor_choice():
    names = dict(backend.DEFAULT_NAMES)

    assert backend.is_chancellor_choice(900)
    assert backend.is_chancellor_choice(905)
    assert not backend.is_chancellor_choice(906)
    assert not backend.is_chancellor_choice(990)

    label = backend.decode_action_label(990, names, backend.HUMAN)
    assert "Amira" in label
    assert "Vizir" not in label


def test_unknown_profile_requires_entry_reason():
    response = client.post(
        "/api/player-profile/dialogue",
        json={"first_name": "Camille", "last_name": "Inconnu"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["requires_entry_reason"] is True
    assert data["player_profile"]["first_name"] == "Camille"
    assert "raison" in data["qadi_dialogue"][0]["text"].lower()


def test_known_profile_skips_entry_reason_question():
    response = client.post(
        "/api/player-profile/dialogue",
        json={"first_name": "Ibra", "last_name": "ASBN"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["requires_entry_reason"] is False
    assert data["requires_identity_confirmation"] is True
    assert data["player_profile"]["is_known_profile"] is True
    assert data["player_profile"]["player_identity_id"].startswith("player_")
    assert data["player_profile"]["stats"]["matches_played"] == 0
    assert "Ibra" in data["qadi_dialogue"][0]["text"]
    assert "Est-ce bien vous" in data["qadi_dialogue"][0]["text"]


def test_confirmed_known_profile_returns_intro_dialogue():
    response = client.post(
        "/api/player-profile/dialogue",
        json={"first_name": "Ibra", "last_name": "ASBN", "identity_confirmed": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["requires_identity_confirmation"] is False
    assert data["requires_entry_reason"] is False
    assert data["player_profile"]["identity_confirmed"] is True
    assert "registre" in data["qadi_dialogue"][0]["text"].lower()


def test_recruiter_hint_reason_is_stored_on_new_game():
    response = client.post(
        "/api/games",
        json={
            "first_name": "Camille",
            "last_name": "Recruteur",
            "entry_reason": "evaluate_contender",
            "ai_policies": dict(backend.DEFAULT_AI_POLICY_IDS),
        },
    )

    assert response.status_code == 200
    data = response.json()
    profile = data["player_profile"]
    assert profile["first_name"] == "Camille"
    assert profile["last_name"] == "Recruteur"
    assert profile["display_name"] == "Camille"
    assert profile["entry_reason"] == "evaluate_contender"
    assert profile["entry_reason_signal"] == "recruiter_hint"
    assert any(event["type"] == "game_created" for event in data["analytics_events"])
    assert any("Motif d'entree" in row["text"] for row in data["logs"])
    assert profile["player_identity_id"].startswith("player_")
    assert profile["stats"]["matches_played"] == 0
    assert backend.PLAYER_STATS_PATH.exists()


def test_match_end_qadi_dialogue_is_separate_from_round_state():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(
            first_name="Camille",
            last_name="Finale",
            entry_reason="challenge_family",
        )
    )
    game = backend.GameSession(
        game_id="test-match-dialogue",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
    )

    assert game.qadi_dialogue == []
    dialogue = backend.build_match_end_qadi_dialogue(game, [backend.HUMAN])
    assert dialogue
    assert "partie" in dialogue[0]["text"].lower()


def test_structured_action_log_contains_public_and_omniscient_states():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(first_name="Log", last_name="Tester", entry_reason="challenge_family")
    )
    backend.ensure_player_record(profile)
    game = backend.GameSession(
        game_id="structured-log-test",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
        ai_policy_ids=dict(backend.DEFAULT_AI_POLICY_IDS),
    )
    backend.start_round(game, reset_match=True)
    actor = game.env.agent_selection
    action = backend.valid_action_payloads(game.env, actor, game.names)[0]["action"]

    backend.apply_action(game, action)

    action_events = [event for event in game.structured_events if event["type"] == "action_played"]
    assert action_events
    payload = action_events[-1]["payload"]
    assert payload["actor"] == actor
    assert "state_before" in payload
    assert "state_after" in payload
    assert "public_state_before" in payload
    assert "public_state_after" in payload
    assert payload["state_before"]["players"][0]["hand"]
    assert "state_before" not in game.analytics_events[-1]["payload"]


def test_replay_endpoint_is_blocked_until_match_over():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(first_name="Replay", last_name="Blocked", entry_reason="challenge_family")
    )
    backend.ensure_player_record(profile)
    game = backend.GameSession(
        game_id="replay-blocked",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
        ai_policy_ids=dict(backend.DEFAULT_AI_POLICY_IDS),
    )
    backend.start_round(game, reset_match=True)
    backend.GAMES[game.game_id] = game

    active_response = client.get(f"/api/games/{game.game_id}/replay")
    assert active_response.status_code == 409

    game.round_over = True
    round_only_response = client.get(f"/api/games/{game.game_id}/replay")
    assert round_only_response.status_code == 409

    game.match_over = True
    game.match_points[backend.HUMAN] = backend.TARGET_POINTS
    finished_response = client.get(f"/api/games/{game.game_id}/replay")
    assert finished_response.status_code == 200
    data = finished_response.json()
    assert data["game_id"] == game.game_id
    assert data["events"]


def test_match_stats_are_recorded_once_and_dialogue_uses_palmares():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(first_name="Palmares", last_name="Tester", entry_reason="challenge_family")
    )
    backend.ensure_player_record(profile)
    game = backend.GameSession(
        game_id="palmares-test",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
        ai_policy_ids=dict(backend.DEFAULT_AI_POLICY_IDS),
    )
    backend.start_round(game, reset_match=True)
    game.match_points[backend.HUMAN] = backend.TARGET_POINTS

    delta = backend.update_player_stats_for_match(game, [backend.HUMAN])
    game.match_stats_delta = delta
    game.match_stats_recorded = True
    dialogue = backend.build_match_end_qadi_dialogue(game, [backend.HUMAN])

    assert delta["before"]["matches_played"] == 0
    assert delta["after"]["matches_played"] == 1
    assert delta["after"]["wins"] == 1
    assert any("Palmarès actuel" in line["text"] for line in dialogue)


def test_private_qadi_notes_are_cleared_between_rounds():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(first_name="Secret", last_name="Tester", entry_reason="challenge_family")
    )
    game = backend.GameSession(
        game_id="private-note-clear",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
        ai_policy_ids=dict(backend.DEFAULT_AI_POLICY_IDS),
    )
    game.private_notes.append({
        "ts": backend.now_stamp(),
        "round_index": 1,
        "turn_index": 1,
        "text": "Secret: L'Amira a Émir (3).",
        "target": "player_3",
        "card": 3,
    })

    backend.start_round(game, reset_match=False)

    assert game.private_notes == []


def test_qadi_secret_is_only_exposed_in_journal_state():
    profile = backend.build_player_profile(
        backend.PlayerProfileRequest(first_name="Qadi", last_name="Journal", entry_reason="challenge_family")
    )
    game = backend.GameSession(
        game_id="qadi-journal-only",
        names={**backend.DEFAULT_NAMES, backend.HUMAN: profile["display_name"]},
        player_profile=profile,
        ai_policy_ids=dict(backend.DEFAULT_AI_POLICY_IDS),
    )
    game.add_log("Secret: L'Amira a Émir (3).", "secret")
    game.private_notes.append({
        "ts": backend.now_stamp(),
        "round_index": 1,
        "turn_index": 1,
        "text": "Secret: L'Amira a Émir (3).",
        "target": "player_3",
        "card": 3,
    })

    state = backend.serialize_state(game)

    assert "private_notes" not in state
    assert any(row["tone"] == "secret" for row in state["logs"])
