"""Tests for gardener swarm (Q3)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args, env_extra=None, expect_zero=True, timeout=60):
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        [sys.executable, "-m", "gardener.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if expect_zero:
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def test_swarm_runs_echo_only_pipeline(tmp_path):
    """A single echo node pipeline succeeds without loading any model."""
    registry = tmp_path / "r.json"
    pipe_file = tmp_path / "echo.prose"
    pipe_file.write_text(
        """\
pipeline echo-test:
  deadline 10.0

  agent echo:
    prompt: "ignored"
    max_tokens: 4
"""
    )
    journal_root = tmp_path / "swarm-journal"
    r = _run_cli(
        "swarm",
        str(pipe_file),
        "--journal-root",
        str(journal_root),
        "--timeout",
        "20",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "state=succeeded" in r.stdout
    # Journal file must exist after a successful run.
    assert (journal_root / "pipeline.jsonl").exists()


def test_swarm_outputs_section_present(tmp_path):
    """Outputs section is always printed even if empty-valued."""
    registry = tmp_path / "r.json"
    pipe_file = tmp_path / "echo.prose"
    pipe_file.write_text(
        """\
pipeline out-check:
  deadline 10.0

  agent echo:
    prompt: "hello"
    max_tokens: 4
"""
    )
    r = _run_cli(
        "swarm",
        str(pipe_file),
        "--timeout",
        "20",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "=== outputs ===" in r.stdout
    assert "=== dead letters" in r.stdout


def test_swarm_reports_failure_on_unknown_agent(tmp_path):
    """A pipeline that references an unknown agent exits non-zero."""
    registry = tmp_path / "r.json"
    pipe_file = tmp_path / "bad.prose"
    pipe_file.write_text(
        """\
pipeline bad-agent:
  deadline 5.0

  agent definitely-not-registered:
    prompt: "x"
    max_tokens: 4
"""
    )
    journal_root = tmp_path / "j"
    r = _run_cli(
        "swarm",
        str(pipe_file),
        "--journal-root",
        str(journal_root),
        "--timeout",
        "10",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    # Either preload-stage failure (returncode 1) or node failure (DLQ, returncode 1).
    assert r.returncode != 0


def test_swarm_help_lists_options(tmp_path):
    """--help shows all relevant flags."""
    r = _run_cli("swarm", "--help")
    assert "--agent" in r.stdout
    assert "--inputs" in r.stdout
    assert "--journal-root" in r.stdout
    assert "--timeout" in r.stdout
    assert "--priority" in r.stdout
    assert "--max-concurrent" in r.stdout


def test_swarm_inputs_passed_to_pipeline(tmp_path):
    """--inputs JSON appears in the outputs section as initial state."""
    registry = tmp_path / "r.json"
    pipe_file = tmp_path / "with_inputs.prose"
    # One echo node; the seed key from --inputs ends up in outputs.
    pipe_file.write_text(
        """\
pipeline inputs-check:
  deadline 10.0

  agent echo:
    prompt: "ignored"
    max_tokens: 4
"""
    )
    r = _run_cli(
        "swarm",
        str(pipe_file),
        "--inputs",
        '{"seed": "hello from inputs"}',
        "--timeout",
        "20",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "state=succeeded" in r.stdout
    # The seed key is part of initial_inputs and therefore in outputs.
    assert "seed" in r.stdout or "hello from inputs" in r.stdout


def test_swarm_journal_root_respected(tmp_path):
    """--journal-root places the journal file in the specified directory."""
    registry = tmp_path / "r.json"
    pipe_file = tmp_path / "jr.prose"
    pipe_file.write_text(
        """\
pipeline jr-check:
  deadline 10.0

  agent echo:
    prompt: "ok"
    max_tokens: 4
"""
    )
    journal_root = tmp_path / "my-journal"
    r = _run_cli(
        "swarm",
        str(pipe_file),
        "--journal-root",
        str(journal_root),
        "--timeout",
        "20",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "state=succeeded" in r.stdout
    assert (journal_root / "pipeline.jsonl").exists()
    assert str(journal_root) in r.stdout


def test_swarm_missing_pipeline_file(tmp_path):
    """A missing pipeline file exits with code 1 and a clear error."""
    registry = tmp_path / "r.json"
    r = _run_cli(
        "swarm",
        str(tmp_path / "nonexistent.prose"),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    assert r.returncode != 0
