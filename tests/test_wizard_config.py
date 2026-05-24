"""Tests for gardener/wizard/config.py — thresholds, drafter, corpus policy.

Pure unit tests, no inference. Drives the Q2.1 config surface locked on
2026-05-23: T1 parse fail max 10%, T2 acceptance fail max 20% (logged-only),
T3 germination fail max 5%. Adversarial stratum excluded from threshold
calculation by default.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from gardener.wizard.config import WizardConfig, WizardThresholds


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_thresholds_match_terra_locked_values():
    cfg = WizardConfig()
    assert cfg.thresholds.t1_parse_fail_max == pytest.approx(0.10)
    assert cfg.thresholds.t2_acceptance_fail_max == pytest.approx(0.20)
    assert cfg.thresholds.t3_germination_fail_max == pytest.approx(0.05)


def test_default_excludes_adversarial_from_thresholds():
    cfg = WizardConfig()
    assert cfg.exclude_strata_from_threshold == ("adversarial",)


def test_default_drafter_is_smallest():
    # The "planting fee" debate ended with: ship 0.5B as default; the
    # threshold logic flips to 7B if calibration shows 0.5B can't clear.
    cfg = WizardConfig()
    assert "0.5B" in cfg.drafter or "0.5b" in cfg.drafter.lower()


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_load_from_yaml_overrides_defaults(tmp_path: Path):
    cfg_path = tmp_path / "gardener.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            wizard:
              drafter: mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
              thresholds:
                t1_parse_fail_max: 0.05
                t3_germination_fail_max: 0.02
              corpus:
                path: tests/fixtures/wizard_smoke_corpus.yaml
                exclude_strata_from_threshold:
                  - adversarial
                  - cross-language
            """
        )
    )
    cfg = WizardConfig.from_yaml(cfg_path)
    assert "7B" in cfg.drafter
    assert cfg.thresholds.t1_parse_fail_max == pytest.approx(0.05)
    assert cfg.thresholds.t3_germination_fail_max == pytest.approx(0.02)
    # Unspecified threshold stays at default.
    assert cfg.thresholds.t2_acceptance_fail_max == pytest.approx(0.20)
    assert cfg.exclude_strata_from_threshold == ("adversarial", "cross-language")
    assert cfg.corpus_path == "tests/fixtures/wizard_smoke_corpus.yaml"


def test_load_from_yaml_missing_file_returns_defaults(tmp_path: Path):
    cfg = WizardConfig.from_yaml(tmp_path / "nonexistent.yaml")
    assert cfg.thresholds.t1_parse_fail_max == pytest.approx(0.10)


def test_load_from_yaml_missing_wizard_key_returns_defaults(tmp_path: Path):
    cfg_path = tmp_path / "gardener.yaml"
    cfg_path.write_text("# empty config\n")
    cfg = WizardConfig.from_yaml(cfg_path)
    assert cfg.thresholds.t3_germination_fail_max == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides_apply_atop_file():
    cfg = WizardConfig().with_overrides(
        drafter="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        t1_fail_max=0.03,
        t3_fail_max=0.01,
    )
    assert "7B" in cfg.drafter
    assert cfg.thresholds.t1_parse_fail_max == pytest.approx(0.03)
    assert cfg.thresholds.t3_germination_fail_max == pytest.approx(0.01)
    # T2 default preserved.
    assert cfg.thresholds.t2_acceptance_fail_max == pytest.approx(0.20)


def test_cli_overrides_with_none_values_are_noops():
    base = WizardConfig().with_overrides(t1_fail_max=0.03)
    after = base.with_overrides(t1_fail_max=None, drafter=None)
    assert after.thresholds.t1_parse_fail_max == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_threshold_must_be_in_unit_interval(bad: float):
    with pytest.raises(ValueError, match="must be in"):
        WizardThresholds(t1_parse_fail_max=bad)


def test_thresholds_are_immutable():
    # Dataclass(frozen=True) — guards against accidental mutation downstream.
    cfg = WizardConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        cfg.thresholds.t1_parse_fail_max = 0.5  # type: ignore[misc]
