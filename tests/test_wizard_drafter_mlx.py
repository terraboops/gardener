"""Tests for gardener/wizard/drafter_mlx.py.

Pure-Python unit tests (output trimming) run in CI.
Real-MLX integration tests are marked @pytest.mark.model and skipped
in environments without an MLX backend (i.e. GitHub Actions).
"""
from __future__ import annotations

import pytest

# CI-runnable: the trim helper is pure Python.
from gardener.wizard.drafter_mlx import _trim_to_recognized_sections


# ---------------------------------------------------------------------------
# Output trimming — pure-Python, no MLX needed
# ---------------------------------------------------------------------------


def test_trim_drops_extra_examples_after_last_section():
    raw = (
        "### System Prompt\nYou are a helper.\n\n"
        "### Knowledge\n- one fact\n\n"
        "---\nEXAMPLE INPUT: something\nEXAMPLE OUTPUT:\n"
        "### System Prompt\nAnother thing.\n"
    )
    out = _trim_to_recognized_sections(raw)
    assert "EXAMPLE INPUT" not in out
    assert "Another thing" not in out
    assert "one fact" in out


def test_trim_drops_unknown_header_continuations():
    raw = (
        "### System Prompt\nHelper.\n\n"
        "### Knowledge\n- fact\n\n"
        "### Some Other Section\nIrrelevant content.\n"
    )
    out = _trim_to_recognized_sections(raw)
    assert "Some Other Section" not in out
    assert "Irrelevant" not in out
    assert "- fact" in out


def test_trim_preserves_clean_output():
    raw = "### System Prompt\nClean.\n\n### Knowledge\n- fact\n"
    out = _trim_to_recognized_sections(raw)
    # End-of-string differences are fine; just check core preserved.
    assert "Clean" in out
    assert "- fact" in out


def test_trim_preserves_decline_only():
    raw = "### Decline\nEmpty input.\n"
    out = _trim_to_recognized_sections(raw)
    assert "Decline" in out
    assert "Empty input" in out


def test_trim_passes_through_when_no_recognized_section():
    # If the drafter produced garbage with no recognizable section, we
    # want T1 failure to surface — NOT silent rescue.
    raw = "completely unstructured prose with no sections at all"
    out = _trim_to_recognized_sections(raw)
    assert out == raw


def test_trim_handles_h2_and_h3_levels():
    raw = (
        "## System Prompt\nHelper.\n\n"
        "## Knowledge\n- fact\n\n"
        "EXAMPLE INPUT: garbage\n"
    )
    out = _trim_to_recognized_sections(raw)
    assert "EXAMPLE INPUT" not in out
    assert "fact" in out


# ---------------------------------------------------------------------------
# Real MLX drafter — gated behind `model` marker (CI skips)
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_mlx_drafter_satisfies_protocol():
    from gardener.wizard.drafter import Drafter
    from gardener.wizard.drafter_mlx import MlxDrafter

    drafter = MlxDrafter("mlx-community/Qwen2.5-0.5B-Instruct-4bit", max_tokens=64)
    assert isinstance(drafter, Drafter)
    assert drafter.name == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.mark.model
def test_mlx_drafter_returns_non_empty_for_narrow_input():
    """Sanity: drafter produces *some* output. Quality is calibrate-wizard's job.

    A consistent failure here means the 0.5B model can't even run with
    this prompt template — that's a deeper problem than calibration can
    fix and should surface loudly.
    """
    from gardener.wizard.drafter_mlx import MlxDrafter

    drafter = MlxDrafter("mlx-community/Qwen2.5-0.5B-Instruct-4bit", max_tokens=128)
    out = drafter.draft("a Python coding agent")
    assert isinstance(out, str)
    assert out.strip(), f"drafter returned empty: {out!r}"
