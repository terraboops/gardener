"""Tests for the gardener wizard (Q2)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _run_cli(*args, env_extra=None, expect_zero=True, stdin: str = ""):
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        [sys.executable, "-m", "gardener.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        input=stdin,
    )
    if expect_zero:
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}: stderr={result.stderr}"
        )
    return result


# --- inline (scripted) path -------------------------------------------------


def test_wizard_inline_scaffolds_agent(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "alpha"
    _run_cli(
        "wizard",
        "--inline",
        "name=alpha",
        "description=test agent",
        "purpose=help with tests",
        "model=mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "temperature=0.0",
        "max_tokens=256",
        f"path={agent_path}",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert (agent_path / "agent.yaml").exists()
    cfg = yaml.safe_load((agent_path / "agent.yaml").read_text())
    assert cfg["name"] == "alpha"
    assert cfg["description"] == "test agent"
    assert cfg["temperature"] == 0.0
    assert cfg["max_tokens"] == 256
    reg = json.loads(registry.read_text())
    assert "alpha" in reg


def test_wizard_drafts_system_prompt_from_purpose(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "beta"
    _run_cli(
        "wizard",
        "--inline",
        "name=beta",
        "purpose=draft careful Python code",
        f"path={agent_path}",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    prompt = (agent_path / "prompt.md").read_text()
    assert "beta" in prompt
    assert "draft careful Python code" in prompt
    assert "if you don't know something" in prompt.lower()


def test_wizard_seeds_knowledge_when_provided(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "gamma"
    _run_cli(
        "wizard",
        "--inline",
        "name=gamma",
        "purpose=knowledge test",
        f"path={agent_path}",
        "seed_knowledge=the sky appears blue because of Rayleigh scattering",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    kdir = agent_path / "knowledge" / "gamma"
    yamls = list(kdir.glob("*.yaml"))
    assert len(yamls) == 1
    body = yaml.safe_load(yamls[0].read_text())
    assert "Rayleigh" in body["insight"]
    # Human-seeded → tested=True.
    assert body["empirical"]["tested"] is True
    assert body["empirical"]["helped"] >= 1


def test_wizard_no_seed_skips_knowledge(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "delta"
    _run_cli(
        "wizard",
        "--inline",
        "name=delta",
        "purpose=no seed",
        f"path={agent_path}",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    kdir = agent_path / "knowledge" / "delta"
    # Either dir doesn't exist or it's empty.
    if kdir.exists():
        assert list(kdir.glob("*.yaml")) == []


def test_wizard_inline_rejects_bad_slug(tmp_path):
    registry = tmp_path / "r.json"
    r = _run_cli(
        "wizard",
        "--inline",
        "name=My Bad Name",
        f"path={tmp_path / 'bad'}",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    assert r.returncode != 0


def test_wizard_inline_force_overwrites(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "epsilon"
    agent_path.mkdir()
    (agent_path / "stale.txt").write_text("x")
    _run_cli(
        "wizard",
        "--inline",
        "name=epsilon",
        "purpose=stale",
        f"path={agent_path}",
        "force=1",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert (agent_path / "agent.yaml").exists()


# --- interactive path (piped stdin) ------------------------------------------


def test_wizard_interactive_with_piped_input(tmp_path):
    registry = tmp_path / "r.json"
    agent_path = tmp_path / "zeta"
    # Pipe the answers in order; -y skips final confirmation.
    stdin_lines = "\n".join([
        "zeta",                          # name
        "wizard-test agent",             # description
        "answer questions concisely",    # purpose
        "",                              # model (accept default)
        "",                              # temperature (default)
        "",                              # max_tokens (default)
        str(agent_path),                 # path
        "",                              # seed_knowledge (skip)
    ]) + "\n"
    _run_cli(
        "wizard",
        "-y",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=stdin_lines,
    )
    assert (agent_path / "agent.yaml").exists()
    cfg = yaml.safe_load((agent_path / "agent.yaml").read_text())
    assert cfg["name"] == "zeta"
    assert cfg["description"] == "wizard-test agent"


def test_wizard_help_lists_options():
    r = _run_cli("wizard", "--help")
    assert "--inline" in r.stdout
    assert "--yes" in r.stdout or "-y" in r.stdout
