"""Tests for gardener/wizard/drafter.py — Drafter Protocol + MockDrafter.

The Drafter Protocol is the seam between CI-runnable code and local-only
MLX inference. MockDrafter replays fixture-recorded responses so the rest
of Q2.1 (parser, adversarial classifier, threshold logic, calibrate
command) can be tested without loading any model.

The real MLX-backed drafter (slice F) implements the same Protocol.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gardener.wizard.drafter import (
    Drafter,
    DrafterReplayMiss,
    MockDrafter,
)


# ---------------------------------------------------------------------------
# Protocol structural-typing
# ---------------------------------------------------------------------------


def test_mock_satisfies_drafter_protocol():
    mock = MockDrafter(fixtures={})
    # runtime_checkable Protocol — isinstance should work.
    assert isinstance(mock, Drafter)


def test_mock_exposes_name():
    mock = MockDrafter(fixtures={}, name="mock-test-drafter")
    assert mock.name == "mock-test-drafter"


# ---------------------------------------------------------------------------
# Replay behavior
# ---------------------------------------------------------------------------


def test_replay_returns_recorded_output():
    fixtures = [
        {
            "input": "a Python coding agent",
            "stratum": "narrow",
            "output": "### System Prompt\nYou are Python expert.\n",
        }
    ]
    mock = MockDrafter(fixtures=fixtures)
    out = mock.draft("a Python coding agent")
    assert "Python expert" in out


def test_replay_miss_raises_helpful_error():
    mock = MockDrafter(fixtures=[])
    with pytest.raises(DrafterReplayMiss) as excinfo:
        mock.draft("an unrecorded purpose")
    # Error message should point the developer at the fixture file.
    assert "fixture" in str(excinfo.value).lower()
    assert "unrecorded purpose" in str(excinfo.value)


def test_replay_distinguishes_different_inputs():
    fixtures = [
        {
            "input": "purpose A",
            "stratum": "narrow",
            "output": "### System Prompt\nAlpha agent.",
        },
        {
            "input": "purpose B",
            "stratum": "narrow",
            "output": "### System Prompt\nBravo agent.",
        },
    ]
    mock = MockDrafter(fixtures=fixtures)
    assert "Alpha" in mock.draft("purpose A")
    assert "Bravo" in mock.draft("purpose B")


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_load_from_yaml_file(tmp_path: Path):
    fixture_file = tmp_path / "mock_drafter_outputs.yaml"
    fixture_file.write_text(
        textwrap.dedent(
            """
            - input: "a SQL agent"
              stratum: narrow
              output: |
                ### System Prompt
                You are SQL expert.
            - input: ""
              stratum: adversarial
              sub_type: vacuous
              output: |
                ### Decline
                Empty input.
            """
        )
    )
    mock = MockDrafter.from_yaml(fixture_file)
    assert "SQL expert" in mock.draft("a SQL agent")
    assert "Decline" in mock.draft("")


def test_load_from_yaml_with_duplicate_inputs_raises():
    fixtures = [
        {"input": "X", "stratum": "narrow", "output": "first"},
        {"input": "X", "stratum": "narrow", "output": "second"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        MockDrafter(fixtures=fixtures)


# ---------------------------------------------------------------------------
# Replicas / nondeterminism simulation
# ---------------------------------------------------------------------------


def test_fixture_can_list_multiple_outputs_for_one_input():
    # Real drafters are nondeterministic — fixtures support replicas so
    # calibration tests can exercise variance handling.
    fixtures = [
        {
            "input": "ambiguous",
            "stratum": "broad",
            "outputs": [
                "### System Prompt\nVariant 1.",
                "### System Prompt\nVariant 2.",
                "### Decline\nCouldn't decide.",
            ],
        },
    ]
    mock = MockDrafter(fixtures=fixtures)
    # First three calls cycle through; fourth wraps.
    a = mock.draft("ambiguous")
    b = mock.draft("ambiguous")
    c = mock.draft("ambiguous")
    d = mock.draft("ambiguous")
    assert "Variant 1" in a
    assert "Variant 2" in b
    assert "Decline" in c
    assert a == d  # wrap-around


def test_single_output_and_multi_output_cannot_coexist_in_one_entry():
    fixtures = [
        {
            "input": "X",
            "output": "single",
            "outputs": ["multi-1", "multi-2"],
        }
    ]
    with pytest.raises(ValueError, match="both 'output' and 'outputs'"):
        MockDrafter(fixtures=fixtures)
