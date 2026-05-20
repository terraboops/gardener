"""CLI agent-lifecycle tests (Q1).

Pure-logic: directory scaffolding, registry round-trip, name validation,
subprocess invocation of the CLI. Model integration for query."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args, env_extra=None, expect_zero=True):
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        [sys.executable, "-m", "gardener.cli.main", *args],
        capture_output=True, text=True, env=env,
    )
    if expect_zero:
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}: stderr={result.stderr!r} "
            f"stdout={result.stdout!r}"
        )
    return result


# --- registry ---------------------------------------------------------------

def test_init_creates_directory_structure(tmp_path):
    registry = tmp_path / "registry.json"
    agent_dir = tmp_path / "alpha"
    _run_cli("init", "alpha", "--path", str(agent_dir),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    assert (agent_dir / "agent.yaml").exists()
    assert (agent_dir / "prompt.md").exists()
    assert (agent_dir / "knowledge").is_dir()
    assert (agent_dir / "journal").is_dir()
    assert (agent_dir / "cache").is_dir()
    assert (agent_dir / "pipelines").is_dir()


def test_init_registers_agent_by_name(tmp_path):
    registry = tmp_path / "registry.json"
    _run_cli("init", "beta", "--path", str(tmp_path / "beta"),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    data = json.loads(registry.read_text())
    assert "beta" in data
    assert Path(data["beta"]).resolve() == (tmp_path / "beta").resolve()


def test_init_rejects_bad_slug(tmp_path):
    # Names like "My Agent" or "" must be rejected.
    registry = tmp_path / "r.json"
    r = _run_cli("init", "My Agent", "--path", str(tmp_path / "ma"),
                 env_extra={"GARDENER_REGISTRY": str(registry)},
                 expect_zero=False)
    assert r.returncode != 0


def test_init_refuses_to_overwrite_existing(tmp_path):
    registry = tmp_path / "r.json"
    agent = tmp_path / "gamma"
    agent.mkdir()
    (agent / "agent.yaml").write_text("preexisting")
    r = _run_cli("init", "gamma", "--path", str(agent),
                 env_extra={"GARDENER_REGISTRY": str(registry)},
                 expect_zero=False)
    assert r.returncode != 0


def test_init_force_overwrites(tmp_path):
    registry = tmp_path / "r.json"
    agent = tmp_path / "delta"
    agent.mkdir()
    (agent / "stale.txt").write_text("x")
    _run_cli("init", "delta", "--path", str(agent), "--force",
             env_extra={"GARDENER_REGISTRY": str(registry)})
    assert (agent / "agent.yaml").exists()


def test_list_shows_registered_agents(tmp_path):
    registry = tmp_path / "r.json"
    _run_cli("init", "one", "--path", str(tmp_path / "one"),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    _run_cli("init", "two", "--path", str(tmp_path / "two"),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    r = _run_cli("list", env_extra={"GARDENER_REGISTRY": str(registry)})
    assert "one" in r.stdout and "two" in r.stdout


def test_remove_unregisters_but_keeps_files(tmp_path):
    registry = tmp_path / "r.json"
    agent = tmp_path / "epsilon"
    _run_cli("init", "epsilon", "--path", str(agent),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    _run_cli("remove", "epsilon",
             env_extra={"GARDENER_REGISTRY": str(registry)})
    data = json.loads(registry.read_text())
    assert "epsilon" not in data
    # Files should NOT be deleted.
    assert agent.exists() and (agent / "agent.yaml").exists()


def test_update_writes_knowledge_object(tmp_path):
    registry = tmp_path / "r.json"
    agent = tmp_path / "zeta"
    _run_cli("init", "zeta", "--path", str(agent),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    _run_cli("update", "zeta", "the sky appears blue because of Rayleigh scattering",
             env_extra={"GARDENER_REGISTRY": str(registry)})
    # KnowledgeStore lives at agent/knowledge/<agent-name>/.
    knowledge_dir = agent / "knowledge" / "zeta"
    assert knowledge_dir.is_dir()
    yamls = list(knowledge_dir.glob("*.yaml"))
    assert len(yamls) == 1
    body = yamls[0].read_text()
    assert "Rayleigh" in body


# --- model integration ------------------------------------------------------

@pytest.mark.model
def test_query_end_to_end(tmp_path):
    registry = tmp_path / "r.json"
    agent_dir = tmp_path / "answerer"
    _run_cli("init", "answerer", "--path", str(agent_dir),
             env_extra={"GARDENER_REGISTRY": str(registry)})
    # Overwrite prompt.md with something opinionated so output is deterministic-ish.
    (agent_dir / "prompt.md").write_text(
        "You are a careful, terse answerer. Answer in one short sentence."
    )
    r = _run_cli("query", "answerer", "What color is the sky on a clear day?",
                 "--max-tokens", "30",
                 env_extra={"GARDENER_REGISTRY": str(registry)})
    out = r.stdout.strip()
    assert len(out) > 0
    # Journal recorded the query.
    journal_files = list((agent_dir / "journal").glob("*.jsonl"))
    assert len(journal_files) >= 1
