"""Tests for gardener cache (Q4).

Pure-logic tests (no model load) run by default.
Model integration test requires: pytest -m model
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            f"CLI exited {result.returncode}\n"
            f"stdout={result.stdout!r}\n"
            f"stderr={result.stderr!r}"
        )
    return result


def _init_agent(tmp_path: Path, name: str = "alpha"):
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


# ---------------------------------------------------------------------------
# Pure-logic tests (no model load)
# ---------------------------------------------------------------------------


def test_cache_list_on_empty_agent(tmp_path):
    registry, agent = _init_agent(tmp_path, "empty-cache")
    r = _run_cli(
        "cache", "list", "empty-cache",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "(no warm caches yet)" in r.stdout


def test_cache_list_no_cache_dir(tmp_path):
    """list on an agent whose cache/ dir was manually removed prints a message."""
    registry, agent = _init_agent(tmp_path, "no-dir")
    # Remove the cache dir that scaffold created.
    import shutil
    shutil.rmtree(agent / "cache", ignore_errors=True)
    r = _run_cli(
        "cache", "list", "no-dir",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    # Either "(no cache directory …)" or "(no warm caches yet)" is acceptable;
    # the key thing is that it exits 0.
    assert r.returncode == 0


def test_cache_rm_unknown_returns_error(tmp_path):
    registry, agent = _init_agent(tmp_path, "ghost-cache")
    r = _run_cli(
        "cache", "rm", "ghost-cache", "does-not-exist",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    assert r.returncode != 0
    assert "no such cache" in r.stderr


def test_cache_warm_mutual_exclusion_errors(tmp_path):
    """Specifying both --text and --file should error before any model load."""
    registry, agent = _init_agent(tmp_path, "mutex-test")
    r = _run_cli(
        "cache", "warm", "mutex-test",
        "--text", "hello",
        "--file", "/tmp/does-not-exist.txt",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    combined = (r.stderr + r.stdout).lower()
    assert "at most one" in combined


def test_cache_help_lists_subcommands():
    r = _run_cli("cache", "--help")
    for sub in ("warm", "list", "rm", "use"):
        assert sub in r.stdout


def test_derived_cache_name_is_deterministic():
    """Same text always produces the same cache name."""
    from gardener.cli.commands.cache import _derived_cache_name

    text = "hello world"
    assert _derived_cache_name(text) == _derived_cache_name(text)
    assert _derived_cache_name(text).startswith("warm-")
    assert len(_derived_cache_name(text)) == len("warm-") + 12


def test_derived_cache_name_differs_for_different_text():
    from gardener.cli.commands.cache import _derived_cache_name

    assert _derived_cache_name("abc") != _derived_cache_name("xyz")


def test_cache_use_missing_cache_returns_error(tmp_path):
    registry, agent = _init_agent(tmp_path, "use-missing")
    r = _run_cli(
        "cache", "use", "use-missing", "nonexistent-cache", "hello?",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        expect_zero=False,
    )
    assert r.returncode != 0
    assert "no such cache" in r.stderr


# ---------------------------------------------------------------------------
# Model integration test
# ---------------------------------------------------------------------------


@pytest.mark.model
def test_cache_warm_then_list_then_use_then_rm(tmp_path):
    """End-to-end: warm a cache, see it in list, use it, remove it."""
    registry, agent = _init_agent(tmp_path, "warmer")

    # Write a concise prompt for deterministic warmup.
    (agent / "prompt.md").write_text(
        "You are a terse assistant. Answer in one short sentence.\n"
    )

    # warm
    r = _run_cli(
        "cache", "warm", "warmer",
        "--use-prompt-md",
        "--name", "default",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        timeout=300,
    )
    assert "[cache] warmed: default" in r.stdout

    cache_file = agent / "cache" / "default.safetensors"
    assert cache_file.exists()
    assert cache_file.stat().st_size > 0

    # list shows it
    r = _run_cli(
        "cache", "list", "warmer",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "default" in r.stdout

    # use it
    r = _run_cli(
        "cache", "use", "warmer", "default", "What color is the sky?",
        env_extra={"GARDENER_REGISTRY": str(registry)},
        timeout=300,
    )
    assert len(r.stdout.strip()) > 0

    # remove it
    r = _run_cli(
        "cache", "rm", "warmer", "default",
        env_extra={"GARDENER_REGISTRY": str(registry)},
    )
    assert "[cache] removed: default" in r.stdout
    assert not cache_file.exists()
    # meta sidecar should be gone too
    assert not (agent / "cache" / "default.meta").exists()
