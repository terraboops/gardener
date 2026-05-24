"""Test: wizard plants initial germination state alongside agent.yaml."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml


def _run_cli(*args, env_extra=None, stdin: str = ""):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "gardener.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
        input=stdin,
    )


def test_inline_wizard_plants_pending_germination(tmp_path: Path):
    agent_path = tmp_path / "alpha"
    registry = tmp_path / "r.json"
    r = _run_cli(
        "wizard",
        "--inline",
        "name=alpha",
        "description=test agent",
        f"path={agent_path}",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert r.returncode == 0, r.stderr

    germ_path = agent_path / "germination.yaml"
    assert germ_path.exists(), (
        f"wizard should plant germination.yaml; not found at {germ_path}"
    )
    data = yaml.safe_load(germ_path.read_text())
    assert data["status"] == "pending"
    assert data["calls_observed"] == 0
    assert data["errors"] == []
    assert data["on_fail"] == "alert"
    assert data["drafted_by"] == "template"  # inline path skips drafter
    assert data["planted_at"]


def test_drafter_wizard_plants_germination_with_drafter_id(tmp_path: Path):
    purpose = "a coding agent"
    fixtures = tmp_path / "mock.yaml"
    fixtures.write_text(
        yaml.safe_dump(
            [
                {
                    "input": purpose,
                    "stratum": "narrow",
                    "output": textwrap.dedent("""\
                        ### System Prompt
                        Drafted seed.

                        ### Knowledge
                        - one fact
                    """),
                }
            ]
        )
    )
    agent_path = tmp_path / "beta"
    registry = tmp_path / "r.json"

    # Interactive with drafter + accept
    lines = [
        "beta",          # name
        "",              # description
        purpose,
        "a",             # accept drafter output
        "",              # model default
        "",              # temperature default
        "",              # max_tokens default
        str(agent_path),
        "y",             # confirm
    ]
    r = _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin="\n".join(lines) + "\n",
    )
    assert r.returncode == 0, r.stderr

    germ = yaml.safe_load((agent_path / "germination.yaml").read_text())
    assert germ["status"] == "pending"
    # drafted_by records the fixture path (the drafter we used)
    assert "mock.yaml" in germ["drafted_by"] or germ["drafted_by"] != "template"
