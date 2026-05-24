"""Tests for `gardener germination-status` CLI + its integration with
`gardener calibrate-wizard --t3-from-germination-status`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml


def _run_cli(*args, env_extra=None, expect_zero=True):
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        [sys.executable, "-m", "gardener.cli.main", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    if expect_zero:
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}\nstderr={result.stderr}\nstdout={result.stdout}"
        )
    return result


def _seed_registry(tmp_path: Path, agents: dict) -> Path:
    """Write a registry.json under tmp_path with the given {name: rel_path}."""
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps({n: str((tmp_path / p).resolve()) for n, p in agents.items()})
    )
    return reg_path


def _write_germination(
    agent_dir: Path,
    *,
    status: str = "pending",
    calls_observed: int = 0,
    errors=None,
    on_fail: str = "alert",
    drafted_by: str = "template",
    planted_at: str = "2026-05-23T12:00:00",
) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "germination.yaml").write_text(
        yaml.safe_dump(
            {
                "status": status,
                "calls_observed": calls_observed,
                "errors": errors or [],
                "on_fail": on_fail,
                "drafted_by": drafted_by,
                "planted_at": planted_at,
            },
            sort_keys=True,
        )
    )


# ---------------------------------------------------------------------------
# Table output (default)
# ---------------------------------------------------------------------------


def test_empty_registry_prints_hint(tmp_path: Path):
    reg = _seed_registry(tmp_path, {})
    r = _run_cli("germination-status", env_extra={"GARDENER_REGISTRY": str(reg)})
    assert "no agents registered" in r.stdout.lower()


def test_table_output_with_mixed_statuses(tmp_path: Path):
    _write_germination(tmp_path / "alpha", status="passed", calls_observed=3,
                       drafted_by="qwen-7b")
    _write_germination(tmp_path / "beta", status="failed", calls_observed=3,
                       errors=["call 1: empty response", "call 2: timeout"],
                       drafted_by="qwen-0.5b", on_fail="regenerate")
    _write_germination(tmp_path / "gamma", status="pending", calls_observed=1)
    reg = _seed_registry(tmp_path, {"alpha": "alpha", "beta": "beta", "gamma": "gamma"})

    r = _run_cli("germination-status", env_extra={"GARDENER_REGISTRY": str(reg)})

    out = r.stdout
    # All three statuses surfaced
    assert "passed" in out and "1" in out
    assert "failed" in out
    assert "pending" in out
    # Per-drafter breakdown
    assert "qwen-7b" in out
    assert "qwen-0.5b" in out
    # Failure detail
    assert "beta" in out
    assert "empty response" in out
    # T3 rate: 1 failed / 2 decided = 50%
    assert "50.0%" in out or "50%" in out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def test_json_output_to_stdout(tmp_path: Path):
    _write_germination(tmp_path / "x", status="passed", calls_observed=3)
    _write_germination(tmp_path / "y", status="failed", calls_observed=3,
                       errors=["call 1: timeout"])
    reg = _seed_registry(tmp_path, {"x": "x", "y": "y"})

    r = _run_cli("germination-status", "--format", "json",
                 env_extra={"GARDENER_REGISTRY": str(reg)})
    data = json.loads(r.stdout)
    assert data["total_agents"] == 2
    assert data["decided_count"] == 2
    assert data["t3_germination_fail_rate"] == pytest.approx(0.5)
    assert data["per_status"]["passed"] == 1
    assert data["per_status"]["failed"] == 1


def test_json_output_to_file(tmp_path: Path):
    _write_germination(tmp_path / "x", status="passed", calls_observed=3)
    reg = _seed_registry(tmp_path, {"x": "x"})
    out_file = tmp_path / "germ.json"

    _run_cli("germination-status", "--format", "json",
             "--output", str(out_file),
             env_extra={"GARDENER_REGISTRY": str(reg)})
    data = json.loads(out_file.read_text())
    assert data["total_agents"] == 1


# ---------------------------------------------------------------------------
# --fail-on-t3-exceed CI gating
# ---------------------------------------------------------------------------


def test_fail_on_t3_exceed_returns_nonzero(tmp_path: Path):
    # 2 of 3 decided agents failed → T3 = 66.7% >> 5%
    _write_germination(tmp_path / "a", status="passed", calls_observed=3)
    _write_germination(tmp_path / "b", status="failed", calls_observed=3,
                       errors=["call 1: empty"])
    _write_germination(tmp_path / "c", status="failed", calls_observed=3,
                       errors=["call 1: empty"])
    reg = _seed_registry(tmp_path, {n: n for n in ("a", "b", "c")})

    r = _run_cli("germination-status", "--fail-on-t3-exceed", "0.05",
                 env_extra={"GARDENER_REGISTRY": str(reg)},
                 expect_zero=False)
    assert r.returncode == 1
    assert "exceeds threshold" in r.stderr.lower()


def test_fail_on_t3_exceed_passes_when_under_threshold(tmp_path: Path):
    # 1 of 20 decided agents failed → T3 = 5.0% <= 5% threshold (boundary)
    # Use 19 passed + 1 failed to land at exactly 5%.
    for i in range(19):
        _write_germination(tmp_path / f"p{i}", status="passed", calls_observed=3)
    _write_germination(tmp_path / "f0", status="failed", calls_observed=3,
                       errors=["call 1: empty"])
    reg = _seed_registry(
        tmp_path, {**{f"p{i}": f"p{i}" for i in range(19)}, "f0": "f0"}
    )

    r = _run_cli("germination-status", "--fail-on-t3-exceed", "0.05",
                 env_extra={"GARDENER_REGISTRY": str(reg)})
    assert r.returncode == 0


def test_fail_on_t3_exceed_no_decided_agents_returns_zero(tmp_path: Path):
    _write_germination(tmp_path / "p", status="pending", calls_observed=1)
    reg = _seed_registry(tmp_path, {"p": "p"})

    r = _run_cli("germination-status", "--fail-on-t3-exceed", "0.05",
                 env_extra={"GARDENER_REGISTRY": str(reg)})
    assert r.returncode == 0  # Can't fail something undefined.
    assert "no decided agents" in r.stderr.lower()


# ---------------------------------------------------------------------------
# Integration: calibrate-wizard reads germination-status output
# ---------------------------------------------------------------------------


def test_calibrate_wizard_folds_t3_from_germination_status(tmp_path: Path):
    # Set up: produce a germination-status JSON with bad T3.
    _write_germination(tmp_path / "a", status="failed", calls_observed=3,
                       errors=["call 1: empty"])
    _write_germination(tmp_path / "b", status="failed", calls_observed=3,
                       errors=["call 1: empty"])
    _write_germination(tmp_path / "c", status="passed", calls_observed=3)
    reg = _seed_registry(tmp_path, {n: n for n in ("a", "b", "c")})

    germ_json = tmp_path / "germ.json"
    _run_cli("germination-status", "--format", "json", "--output", str(germ_json),
             env_extra={"GARDENER_REGISTRY": str(reg)})

    # Sanity: T3 = 2/3 = 66.7%
    germ_data = json.loads(germ_json.read_text())
    assert germ_data["t3_germination_fail_rate"] == pytest.approx(2 / 3)

    # Now calibrate-wizard with clean mock fixtures → T1 would PASS.
    # But folding the bad T3 must flip the verdict to FAIL.
    out_path = tmp_path / "baseline.json"
    r = _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", "tests/fixtures/wizard_smoke_corpus.yaml",
        "--mock-fixtures", "tests/fixtures/mock_drafter_outputs.yaml",
        "--output", str(out_path),
        "--t3-from-germination-status", str(germ_json),
        "--exit-nonzero-on-flip",
        expect_zero=False,
    )
    assert r.returncode == 1, (
        f"expected exit 1 (T3 flips verdict); got {r.returncode}\nstdout={r.stdout}"
    )
    data = json.loads(out_path.read_text())
    assert data["verdict"]["passes"] is False
    assert data["verdict"]["flip_default"] is True
    assert any("germination" in reason.lower() for reason in data["verdict"]["reasons"])
