"""Tests for gardener observe (Q3)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(*args, env_extra=None, expect_zero=True, timeout=10):
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


def _init_agent(tmp_path, name="alpha"):
    registry = tmp_path / "r.json"
    agent = tmp_path / name
    _run_cli(
        "init",
        name,
        "--path",
        str(agent),
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    return registry, agent


def _seed_journal(agent_path, topic="pipeline", events=None):
    j_path = agent_path / "journal" / f"{topic}.jsonl"
    j_path.parent.mkdir(parents=True, exist_ok=True)
    with j_path.open("a") as f:
        for ev in events or []:
            f.write(json.dumps(ev, sort_keys=True) + "\n")


def test_observe_prints_recent_events(tmp_path):
    registry, agent_path = _init_agent(tmp_path)
    _seed_journal(
        agent_path,
        "pipeline",
        [
            {"event": "start", "name": "x"},
            {"event": "node_ok", "node": "a"},
            {"event": "end", "name": "x"},
        ],
    )
    r = _run_cli(
        "observe",
        "alpha",
        "-n",
        "10",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "start" in r.stdout
    assert "node_ok" in r.stdout
    assert "end" in r.stdout
    assert "[dlq depth:" in r.stdout


def test_observe_respects_n_tail(tmp_path):
    registry, agent_path = _init_agent(tmp_path, "beta")
    _seed_journal(
        agent_path, "pipeline", [{"event": f"e{i}"} for i in range(50)]
    )
    r = _run_cli(
        "observe",
        "beta",
        "-n",
        "3",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    # Should only show the last 3 events (e47, e48, e49).
    assert '"e49"' in r.stdout
    assert '"e47"' in r.stdout
    assert '"e0"' not in r.stdout


def test_observe_topic_filter(tmp_path):
    registry, agent_path = _init_agent(tmp_path, "gamma")
    _seed_journal(agent_path, "query", [{"q": "hi"}])
    _seed_journal(agent_path, "pipeline", [{"event": "start"}])
    r = _run_cli(
        "observe",
        "gamma",
        "--topic",
        "query",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert '"hi"' in r.stdout
    assert "start" not in r.stdout


def test_observe_no_journal_yet(tmp_path):
    """Agent exists but has never been run — should print banner and exit cleanly."""
    registry, agent_path = _init_agent(tmp_path, "delta")
    # Remove the auto-created journal dir so it truly doesn't exist.
    import shutil
    shutil.rmtree(agent_path / "journal", ignore_errors=True)
    r = _run_cli(
        "observe",
        "delta",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    # Graceful output — no crash.
    assert r.returncode == 0


def test_observe_dlq_depth_visible_when_dlq_has_entries(tmp_path):
    registry, agent_path = _init_agent(tmp_path, "epsilon")
    _seed_journal(
        agent_path,
        "dead_letter",
        [
            {"node": "x", "reason": "boom"},
            {"node": "y", "reason": "bang"},
        ],
    )
    r = _run_cli(
        "observe",
        "epsilon",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "[dlq depth: 2]" in r.stdout


def test_observe_empty_topic_shows_banner(tmp_path):
    """Topic file missing (but journal dir exists) → banner + no events, no crash."""
    registry, agent_path = _init_agent(tmp_path, "zeta")
    # Journal dir created by init; leave topic file absent.
    r = _run_cli(
        "observe",
        "zeta",
        "--topic",
        "pipeline",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "[dlq depth:" in r.stdout
    assert r.returncode == 0


def test_observe_help(tmp_path):
    """--help shows expected flags."""
    r = _run_cli("observe", "--help")
    assert "--topic" in r.stdout
    assert "--follow" in r.stdout
    assert "-n" in r.stdout
