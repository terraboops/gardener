"""Tests for gardener.wizard.germination.react_to_germination — F4 wiring."""
from __future__ import annotations

import pytest

from gardener.wizard.germination import (
    GerminationAction,
    GerminationState,
    OBSERVATION_WINDOW,
    react_to_germination,
    record_call,
)


def _walk_to_failed(on_fail: str = "alert", drafted_by: str = "qwen-0.5b") -> GerminationState:
    """Advance a state through 3 calls with ≥1 failure, ending in `failed`."""
    s = GerminationState.new(on_fail=on_fail, drafted_by=drafted_by)
    s, _ = record_call(s, succeeded=False, error_reason="empty response")
    s, _ = record_call(s, succeeded=True)
    s, _ = record_call(s, succeeded=True)
    assert s.status == "failed"
    return s


# ---------------------------------------------------------------------------
# Non-failure cases return None
# ---------------------------------------------------------------------------


def test_pending_state_returns_none():
    s = GerminationState.new()
    assert react_to_germination("alpha", s) is None


def test_passed_state_returns_none():
    s = GerminationState.new()
    for _ in range(OBSERVATION_WINDOW):
        s, _ = record_call(s, succeeded=True)
    assert s.status == "passed"
    assert react_to_germination("alpha", s) is None


# ---------------------------------------------------------------------------
# alert policy
# ---------------------------------------------------------------------------


def test_alert_policy_surfaces_summary_and_event():
    s = _walk_to_failed(on_fail="alert", drafted_by="qwen-0.5b")
    action = react_to_germination("alpha", s)
    assert action is not None
    assert isinstance(action, GerminationAction)
    assert action.policy == "alert"
    # Summary names the agent + drafter + observe command
    assert "alpha" in action.summary
    assert "qwen-0.5b" in action.summary
    assert "gardener observe" in action.summary
    # Journal event has structured fields
    assert action.journal_event["kind"] == "GerminationFailureReaction"
    assert action.journal_event["policy"] == "alert"
    assert action.journal_event["agent"] == "alpha"
    assert action.journal_event["error_count"] == 1


# ---------------------------------------------------------------------------
# regenerate policy
# ---------------------------------------------------------------------------


def test_regenerate_policy_recommends_command():
    s = _walk_to_failed(on_fail="regenerate", drafted_by="qwen-0.5b")
    action = react_to_germination("beta", s)
    assert action is not None
    assert action.policy == "regenerate"
    assert "gardener wizard --regenerate beta" in action.summary
    # Journal event flags that this is a recommendation, not autonomous action
    assert action.journal_event["policy"] == "regenerate"
    assert "TBD" in action.journal_event["note"] or "manual" in action.journal_event["note"]


# ---------------------------------------------------------------------------
# ignore policy
# ---------------------------------------------------------------------------


def test_ignore_policy_emits_event_but_no_surface():
    s = _walk_to_failed(on_fail="ignore", drafted_by="qwen-7b")
    action = react_to_germination("gamma", s)
    assert action is not None
    assert action.policy == "ignore"
    # Summary marks itself as silent
    assert "no surface" in action.summary.lower()
    # Journal event still recorded
    assert action.journal_event["kind"] == "GerminationFailureReaction"
    assert action.journal_event["policy"] == "ignore"


# ---------------------------------------------------------------------------
# Idempotency over states past the window
# ---------------------------------------------------------------------------


def test_already_failed_state_still_returns_action_for_caller_to_handle():
    # Important: react_to_germination is called by the query path right
    # after the state flips. If something downstream loads a previously-
    # failed state, calling react again should still return an action so
    # the caller can re-issue alerts if needed (e.g. retry tooling).
    s = _walk_to_failed(on_fail="alert")
    action1 = react_to_germination("delta", s)
    action2 = react_to_germination("delta", s)
    assert action1 == action2 or (
        action1.journal_event["kind"] == action2.journal_event["kind"]
    )
