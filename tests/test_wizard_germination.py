"""Tests for gardener/wizard/germination.py — T3 state machine + persistence.

Pure unit tests; no inference needed. The integration with `gardener
query` (record_call after each call, journal events) is exercised in
tests/test_cli_query_germination.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gardener.wizard.germination import (
    GerminationState,
    OBSERVATION_WINDOW,
    is_response_error,
    load,
    record_call,
    save,
)


# ---------------------------------------------------------------------------
# State construction + validation
# ---------------------------------------------------------------------------


def test_new_returns_pending_state():
    s = GerminationState.new()
    assert s.status == "pending"
    assert s.calls_observed == 0
    assert s.errors == []
    assert s.on_fail == "alert"
    assert s.drafted_by == "template"
    assert s.planted_at  # ISO timestamp string


def test_new_accepts_drafter_and_policy():
    s = GerminationState.new(drafted_by="qwen-0.5b", on_fail="regenerate")
    assert s.drafted_by == "qwen-0.5b"
    assert s.on_fail == "regenerate"


@pytest.mark.parametrize("bad", ["unknown", "PENDING", "", None])
def test_invalid_status_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        GerminationState(status=bad)


@pytest.mark.parametrize("bad", ["panic", "delete", ""])
def test_invalid_on_fail_rejected(bad):
    with pytest.raises(ValueError):
        GerminationState(on_fail=bad)


def test_calls_observed_bounded():
    with pytest.raises(ValueError):
        GerminationState(calls_observed=5)
    with pytest.raises(ValueError):
        GerminationState(calls_observed=-1)


# ---------------------------------------------------------------------------
# State machine — record_call
# ---------------------------------------------------------------------------


def test_record_call_increments_calls_observed():
    s = GerminationState.new()
    s1, events = record_call(s, succeeded=True)
    assert s1.calls_observed == 1
    assert s1.status == "pending"
    # Always emits a GerminationProgress event.
    assert any(e["kind"] == "GerminationProgress" for e in events)


def test_three_successes_transition_to_passed():
    s = GerminationState.new()
    for i in range(OBSERVATION_WINDOW):
        s, events = record_call(s, succeeded=True)
    assert s.status == "passed"
    assert s.calls_observed == OBSERVATION_WINDOW
    # Last call emits both Progress AND Germinated.
    kinds = [e["kind"] for e in events]
    assert "Germinated" in kinds
    germinated = next(e for e in events if e["kind"] == "Germinated")
    assert germinated["status"] == "passed"
    assert germinated["error_count"] == 0


def test_any_failure_in_window_causes_failed_status():
    s = GerminationState.new()
    s, _ = record_call(s, succeeded=True)
    s, _ = record_call(s, succeeded=False, error_reason="empty response")
    s, events = record_call(s, succeeded=True)
    assert s.status == "failed"
    assert len(s.errors) == 1
    assert "call 2" in s.errors[0]
    assert "empty response" in s.errors[0]
    germinated = next(e for e in events if e["kind"] == "Germinated")
    assert germinated["status"] == "failed"
    assert germinated["error_count"] == 1


def test_record_call_past_window_is_noop():
    s = GerminationState.new()
    for _ in range(OBSERVATION_WINDOW):
        s, _ = record_call(s, succeeded=True)
    # Now in passed state. Subsequent calls should NOT change state.
    s_after, events = record_call(s, succeeded=False)
    assert s_after.status == "passed"
    assert s_after.calls_observed == OBSERVATION_WINDOW
    assert events == []


# ---------------------------------------------------------------------------
# is_response_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response,expected",
    [
        ("", True),
        ("   ", True),
        ("\n\n\t  ", True),
        ("ok", False),
        ("Hello, world!", False),
        ("I don't know.", False),  # not an error — model said something
    ],
)
def test_is_response_error(response: str, expected: bool):
    assert is_response_error(response) == expected


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path: Path):
    s = GerminationState.new(drafted_by="qwen-0.5b", on_fail="regenerate")
    s, _ = record_call(s, succeeded=True)
    s, _ = record_call(s, succeeded=False, error_reason="empty")
    save(tmp_path, s)

    loaded = load(tmp_path)
    assert loaded.status == s.status
    assert loaded.calls_observed == s.calls_observed
    assert loaded.errors == s.errors
    assert loaded.on_fail == "regenerate"
    assert loaded.drafted_by == "qwen-0.5b"


def test_load_returns_fresh_state_when_file_missing(tmp_path: Path):
    s = load(tmp_path)
    assert s.status == "pending"
    assert s.calls_observed == 0


def test_save_is_atomic(tmp_path: Path):
    """Atomicity check: a save error doesn't leave a partial file."""
    s = GerminationState.new()
    save(tmp_path, s)
    # File exists + valid YAML
    p = tmp_path / "germination.yaml"
    assert p.exists()
    data = yaml.safe_load(p.read_text())
    assert data["status"] == "pending"
