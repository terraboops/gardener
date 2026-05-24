"""Integration: gardener query updates germination state + reacts on failure.

Mocks the MLX path so this runs in CI without inference. Verifies the
F4 wiring end-to-end: query → record_call → save → react_to_germination
→ journal events + (when applicable) stderr surface.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from gardener.cli.commands.query import cmd_query
from gardener.cli.registry import register


# unittest.mock.patch("mlx_lm.X") imports mlx_lm to resolve the attribute,
# so this test file needs MLX in the environment even though it mocks all
# generation. On Linux CI (no MLX) the conftest skips collection of any
# file with a module-level pytestmark = pytest.mark.model.
pytestmark = pytest.mark.model


def _scaffold_agent(tmp_path: Path, name: str, *, on_fail: str = "alert") -> Path:
    """Plant a minimal agent with germination already at the brink."""
    p = tmp_path / name
    p.mkdir()
    (p / "agent.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "description": "",
                "prompt_path": "prompt.md",
                "tools": [],
                "max_tokens": 32,
                "temperature": 0.0,
                "draft_model": None,
                "num_draft_tokens": 8,
                "created_at": "2026-05-23T12:00:00",
            },
            sort_keys=False,
        )
    )
    (p / "prompt.md").write_text("You are a test agent.")
    return p


def _seed_germination_at_brink(agent_path: Path, on_fail: str) -> None:
    """Write germination.yaml with 2 calls observed + 1 prior error.

    Next call (whether success or failure) completes the OBSERVATION_WINDOW;
    if the next response is empty (error), state goes to `failed`.
    """
    (agent_path / "germination.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "pending",
                "calls_observed": 2,
                "errors": ["call 1: empty response"],
                "on_fail": on_fail,
                "drafted_by": "qwen-0.5b",
                "planted_at": "2026-05-23T12:00:00",
            },
            sort_keys=True,
        )
    )


def _make_args(agent_name: str, registry_path: Path):
    return SimpleNamespace(
        agent=agent_name,
        question="hello?",
        max_tokens=None,
        temperature=None,
    )


def _read_journal_germination(agent_path: Path) -> list[dict]:
    """Read all events under topic 'germination' from the agent's journal."""
    j = agent_path / "journal" / "germination.jsonl"
    if not j.exists():
        return []
    return [json.loads(line) for line in j.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Successful 3rd call → state becomes "failed" (1 prior error) → alert fires
# ---------------------------------------------------------------------------


def test_query_with_pending_germination_alert_on_failure(tmp_path: Path, monkeypatch, capsys):
    agent_path = _scaffold_agent(tmp_path, "alpha", on_fail="alert")
    _seed_germination_at_brink(agent_path, on_fail="alert")

    registry = tmp_path / "registry.json"
    monkeypatch.setenv("GARDENER_REGISTRY", str(registry))
    register("alpha", agent_path)

    # Mock the MLX path: pretend the model returned a NON-EMPTY response,
    # so this 3rd call is a success — combined with the 1 prior error,
    # the state finalizes as `failed` (1 error, 3 calls observed).
    fake_model = object()
    fake_tok = SimpleNamespace(
        apply_chat_template=lambda msgs, tokenize, add_generation_prompt: "PROMPT"
    )
    with patch("mlx_lm.load", return_value=(fake_model, fake_tok)), \
         patch("mlx_lm.sample_utils.make_sampler", return_value=lambda x: x), \
         patch("mlx_lm.generate", return_value="hello back"):
        rc = cmd_query(_make_args("alpha", registry))

    assert rc == 0
    # State persisted as "failed"
    germ = yaml.safe_load((agent_path / "germination.yaml").read_text())
    assert germ["status"] == "failed"
    assert germ["calls_observed"] == 3

    # Journal has: GerminationProgress (3) + Germinated + GerminationFailureReaction
    events = _read_journal_germination(agent_path)
    kinds = [e["kind"] for e in events]
    assert "GerminationProgress" in kinds
    assert "Germinated" in kinds
    assert "GerminationFailureReaction" in kinds
    reaction = next(e for e in events if e["kind"] == "GerminationFailureReaction")
    assert reaction["policy"] == "alert"
    assert reaction["agent"] == "alpha"

    # Alert surface on stderr
    captured = capsys.readouterr()
    assert "alpha" in captured.err
    assert "germination failed" in captured.err.lower()


# ---------------------------------------------------------------------------
# ignore policy: event recorded, no stderr surface
# ---------------------------------------------------------------------------


def test_query_with_ignore_policy_silent(tmp_path: Path, monkeypatch, capsys):
    agent_path = _scaffold_agent(tmp_path, "delta", on_fail="ignore")
    _seed_germination_at_brink(agent_path, on_fail="ignore")

    registry = tmp_path / "registry.json"
    monkeypatch.setenv("GARDENER_REGISTRY", str(registry))
    register("delta", agent_path)

    fake_tok = SimpleNamespace(
        apply_chat_template=lambda msgs, tokenize, add_generation_prompt: "PROMPT"
    )
    with patch("mlx_lm.load", return_value=(object(), fake_tok)), \
         patch("mlx_lm.sample_utils.make_sampler", return_value=lambda x: x), \
         patch("mlx_lm.generate", return_value="another response"):
        rc = cmd_query(_make_args("delta", registry))

    assert rc == 0
    # State is failed, event recorded, stderr SILENT
    events = _read_journal_germination(agent_path)
    assert any(e["kind"] == "GerminationFailureReaction" for e in events)
    captured = capsys.readouterr()
    # The query response ("another response") IS printed to stdout via print(),
    # but stderr should not carry the germination alert.
    assert "germination failed" not in captured.err.lower()


# ---------------------------------------------------------------------------
# No-op when germination already decided (past window)
# ---------------------------------------------------------------------------


def test_query_with_decided_germination_no_state_change(tmp_path: Path, monkeypatch):
    """Once germination has decided (passed or failed), subsequent queries
    don't touch the state machine — the window has closed."""
    agent_path = _scaffold_agent(tmp_path, "epsilon", on_fail="alert")
    (agent_path / "germination.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "passed",
                "calls_observed": 3,
                "errors": [],
                "on_fail": "alert",
                "drafted_by": "qwen-7b",
                "planted_at": "2026-05-23T12:00:00",
            },
            sort_keys=True,
        )
    )

    registry = tmp_path / "registry.json"
    monkeypatch.setenv("GARDENER_REGISTRY", str(registry))
    register("epsilon", agent_path)

    fake_tok = SimpleNamespace(
        apply_chat_template=lambda msgs, tokenize, add_generation_prompt: "PROMPT"
    )
    with patch("mlx_lm.load", return_value=(object(), fake_tok)), \
         patch("mlx_lm.sample_utils.make_sampler", return_value=lambda x: x), \
         patch("mlx_lm.generate", return_value=""):  # even an empty response should not flip
        rc = cmd_query(_make_args("epsilon", registry))

    assert rc == 0
    germ = yaml.safe_load((agent_path / "germination.yaml").read_text())
    # Unchanged
    assert germ["status"] == "passed"
    assert germ["calls_observed"] == 3
    # No new germination events
    events = _read_journal_germination(agent_path)
    assert events == [] or all(e.get("kind") != "GerminationFailureReaction" for e in events)
