"""Tests for `gardener calibrate-wizard` CLI command.

Uses --mock so CI doesn't need MLX. The real-MLX path is exercised by
slice F's @pytest.mark.model tests + by developers running calibration
locally to produce the committed baseline.json artifacts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args, expect_zero: bool = True):
    env = {**os.environ}
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


def test_calibrate_wizard_emits_baseline_json(tmp_path: Path):
    out_path = tmp_path / "baseline.json"
    result = _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", "tests/fixtures/wizard_smoke_corpus.yaml",
        "--mock-fixtures", "tests/fixtures/mock_drafter_outputs.yaml",
        "--output", str(out_path),
    )
    assert out_path.exists(), result.stdout + result.stderr
    data = json.loads(out_path.read_text())

    # Top-level shape
    assert "drafter" in data
    assert "total_trials" in data
    assert data["total_trials"] == 22  # 22 corpus entries × 1 replica
    assert "thresholds" in data
    # F5: T1 default tightened from 0.10 to 0.05.
    assert data["thresholds"]["t1_parse_fail_max"] == 0.05
    assert data["thresholds"]["t3_germination_fail_max"] == 0.05
    assert "adversarial" in data["exclude_strata_from_threshold"]

    # Per-stratum stats
    assert "t1_fail_rate_by_stratum" in data
    assert "narrow" in data["t1_fail_rate_by_stratum"]
    assert "broad" in data["t1_fail_rate_by_stratum"]
    assert "multi-domain" in data["t1_fail_rate_by_stratum"]

    # Adversarial breakdown by sub_type
    assert "adversarial_breakdown" in data
    assert "vacuous" in data["adversarial_breakdown"]
    assert "prompt_injection" in data["adversarial_breakdown"]

    # Verdict
    assert "verdict" in data
    assert "passes" in data["verdict"]
    assert "flip_default" in data["verdict"]
    assert "recommended_drafter" in data["verdict"]


def test_calibrate_wizard_passes_with_clean_mock_fixtures(tmp_path: Path):
    out_path = tmp_path / "baseline.json"
    _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", "tests/fixtures/wizard_smoke_corpus.yaml",
        "--mock-fixtures", "tests/fixtures/mock_drafter_outputs.yaml",
        "--output", str(out_path),
    )
    data = json.loads(out_path.read_text())
    # Mock fixtures are all clean → PASS, no flip.
    assert data["verdict"]["passes"] is True
    assert data["verdict"]["flip_default"] is False


def test_calibrate_wizard_flips_default_when_threshold_zero(tmp_path: Path):
    """With T1 threshold set to 0%, even a single parse failure flips.

    The mock fixtures are clean (parse rate is 100%), but the resolution
    warning at 0% threshold means SOMETHING in the verdict will register —
    use --t1-fail-max 0 to force the gate semantics, then check that
    verdict.passes still depends on actual failures.

    Actually with perfect fixtures, 0 fail rate <= 0 threshold so it
    still passes. To force a fail we need a poisoned fixture.
    """
    # Construct a small failing corpus + fixture pair.
    bad_corpus = tmp_path / "corpus.yaml"
    bad_corpus.write_text(
        "- input: 'will-fail'\n  stratum: narrow\n"
    )
    bad_fixtures = tmp_path / "fixtures.yaml"
    bad_fixtures.write_text(
        "- input: 'will-fail'\n  stratum: narrow\n  output: 'no sections here at all'\n"
    )
    out_path = tmp_path / "baseline.json"
    result = _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", str(bad_corpus),
        "--mock-fixtures", str(bad_fixtures),
        "--output", str(out_path),
        "--exit-nonzero-on-flip",
        expect_zero=False,
    )
    assert result.returncode == 1, (
        f"expected exit 1 (flip_default); got {result.returncode}\n{result.stdout}"
    )
    data = json.loads(out_path.read_text())
    assert data["verdict"]["passes"] is False
    assert data["verdict"]["flip_default"] is True
    assert data["t1_fail_rate_by_stratum"]["narrow"] == 1.0


def test_calibrate_wizard_with_replicas(tmp_path: Path):
    out_path = tmp_path / "baseline.json"
    _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", "tests/fixtures/wizard_smoke_corpus.yaml",
        "--mock-fixtures", "tests/fixtures/mock_drafter_outputs.yaml",
        "--output", str(out_path),
        "--replicas", "2",
    )
    data = json.loads(out_path.read_text())
    assert data["total_trials"] == 44  # 22 corpus × 2 replicas


def test_calibrate_wizard_missing_corpus_returns_error(tmp_path: Path):
    out_path = tmp_path / "baseline.json"
    result = _run_cli(
        "calibrate-wizard",
        "--mock",
        "--corpus", str(tmp_path / "nope.yaml"),
        "--output", str(out_path),
        expect_zero=False,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()
