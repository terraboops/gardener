"""Tests for the Q2.1 drafter flow integrated into `gardener wizard`.

Uses --mock-drafter-fixtures so CI doesn't need MLX. Pipes stdin for the
interactive prompts. The mock fixture supplies the drafter response for
a chosen purpose; the test scripts the user's accept/edit/regen choice
via the prompt stream.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
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
            f"CLI exited {result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )
    return result


def _write_mock_fixtures(path: Path, purpose: str, output: str) -> Path:
    path.write_text(
        yaml.safe_dump(
            [{"input": purpose, "stratum": "narrow", "output": output}]
        )
    )
    return path


def _multi_output_fixtures(path: Path, purpose: str, outputs: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump(
            [{"input": purpose, "stratum": "narrow", "outputs": outputs}]
        )
    )
    return path


# Interactive prompt sequence (with drafter producing knowledge facts):
#   name, description, purpose, drafter-flow-choice,
#   model, temperature, max_tokens, path, confirm
def _stdin_with_drafter(
    *,
    name: str,
    description: str,
    purpose: str,
    draft_choices: list[str],
    model: str = "",
    temperature: str = "",
    max_tokens: str = "",
    path: str,
    confirm: str = "y",
) -> str:
    lines = [name, description, purpose, *draft_choices,
             model, temperature, max_tokens, path, confirm]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Accept flow
# ---------------------------------------------------------------------------


def test_wizard_drafter_accept_writes_drafted_prompt_and_knowledge(tmp_path: Path):
    purpose = "a coding agent"
    fixtures = _write_mock_fixtures(
        tmp_path / "mock.yaml",
        purpose,
        textwrap.dedent("""\
            ### System Prompt
            You are a coding agent. You write clean, idiomatic code.

            ### Knowledge
            - test-first beats debug-later
            - small diffs are reviewable diffs
        """),
    )
    agent_path = tmp_path / "alpha"
    registry = tmp_path / "r.json"

    result = _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=_stdin_with_drafter(
            name="alpha",
            description="",
            purpose=purpose,
            draft_choices=["a"],   # accept the draft
            path=str(agent_path),
        ),
    )

    # Drafted prompt landed
    prompt_md = (agent_path / "prompt.md").read_text()
    assert "coding agent" in prompt_md
    assert "clean, idiomatic" in prompt_md
    assert "### System Prompt" not in prompt_md  # section header stripped

    # Knowledge facts seeded as separate KnowledgeObjects
    # (KnowledgeStore layout: <root>/<agent>/<oid>.yaml)
    kfiles = list((agent_path / "knowledge" / "alpha").glob("*.yaml"))
    assert len(kfiles) >= 2, f"expected ≥2 knowledge objects; got {kfiles}"
    facts = []
    for kf in kfiles:
        kdata = yaml.safe_load(kf.read_text())
        facts.append(kdata.get("insight", ""))
    assert any("test-first" in f for f in facts)
    assert any("small diffs" in f for f in facts)


# ---------------------------------------------------------------------------
# Skip flow → template fallback
# ---------------------------------------------------------------------------


def test_wizard_drafter_skip_falls_back_to_template(tmp_path: Path):
    purpose = "a coding agent"
    fixtures = _write_mock_fixtures(
        tmp_path / "mock.yaml",
        purpose,
        "### System Prompt\nA seed the user will skip.",
    )
    agent_path = tmp_path / "skipper"
    registry = tmp_path / "r.json"

    # When user skips drafter, the seed_knowledge prompt DOES fire.
    # Sequence: name, description, purpose, draft-choice=skip, model,
    # temperature, max_tokens, path, seed_knowledge, confirm
    lines = [
        "skipper",     # name
        "",            # description
        purpose,
        "s",           # skip drafter
        "",            # model (default)
        "",            # temperature
        "",            # max_tokens
        str(agent_path),
        "",            # seed_knowledge (empty)
        "y",           # confirm
    ]
    stdin = "\n".join(lines) + "\n"

    _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=stdin,
    )

    prompt_md = (agent_path / "prompt.md").read_text()
    # Template includes the agent name and purpose verbatim.
    assert "You are skipper" in prompt_md
    assert "coding agent" in prompt_md
    # NOT the drafted content
    assert "user will skip" not in prompt_md


# ---------------------------------------------------------------------------
# Regenerate → accept
# ---------------------------------------------------------------------------


def test_wizard_drafter_regenerate_then_accept(tmp_path: Path):
    purpose = "an agent that helps me write"
    # Two outputs cycle through; user regenerates once then accepts.
    fixtures = _multi_output_fixtures(
        tmp_path / "mock.yaml",
        purpose,
        [
            "### System Prompt\nFirst draft, user rejects.",
            "### System Prompt\nSecond draft, user accepts.",
        ],
    )
    agent_path = tmp_path / "regen"
    registry = tmp_path / "r.json"

    _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=_stdin_with_drafter(
            name="regen",
            description="",
            purpose=purpose,
            draft_choices=["r", "a"],   # regenerate, then accept
            path=str(agent_path),
        ),
    )

    prompt_md = (agent_path / "prompt.md").read_text()
    assert "Second draft" in prompt_md
    assert "First draft" not in prompt_md


# ---------------------------------------------------------------------------
# T1 auto-retry on first parse failure
# ---------------------------------------------------------------------------


def test_wizard_drafter_auto_retries_on_t1_failure(tmp_path: Path):
    """First output doesn't parse; drafter auto-retries with second; user accepts."""
    purpose = "a parser tester"
    fixtures = _multi_output_fixtures(
        tmp_path / "mock.yaml",
        purpose,
        [
            "garbage no sections here at all",   # T1 fail → auto-retry
            "### System Prompt\nRecovered on retry.",
        ],
    )
    agent_path = tmp_path / "retry"
    registry = tmp_path / "r.json"

    result = _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=_stdin_with_drafter(
            name="retry",
            description="",
            purpose=purpose,
            draft_choices=["a"],
            path=str(agent_path),
        ),
    )

    prompt_md = (agent_path / "prompt.md").read_text()
    assert "Recovered on retry" in prompt_md
    # Retry message surfaced in stdout
    assert "retry" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Quit
# ---------------------------------------------------------------------------


def test_wizard_drafter_quit_exits_nonzero(tmp_path: Path):
    purpose = "a coding agent"
    fixtures = _write_mock_fixtures(
        tmp_path / "mock.yaml",
        purpose,
        "### System Prompt\nSomething the user quits on.",
    )
    agent_path = tmp_path / "quitter"
    registry = tmp_path / "r.json"

    result = _run_cli(
        "wizard",
        "--mock-drafter-fixtures", str(fixtures),
        env_extra={"GARDENER_REGISTRY": str(registry)},
        stdin=_stdin_with_drafter(
            name="quitter",
            description="",
            purpose=purpose,
            draft_choices=["q"],
            path=str(agent_path),
        ),
        expect_zero=False,
    )
    assert result.returncode == 1
    assert not agent_path.exists(), "agent dir should not be created on quit"
