"""Wizard configuration: thresholds, drafter selection, corpus policy.

Loaded from ~/.config/gardener/config.yaml (or any explicit path) under the
`wizard:` key. CLI flags override loaded values via `with_overrides()`.

The threshold values themselves are *global wizard policy*. Per-agent
germination state (status, calls_observed, on_fail) lives in agent.yaml
because it's measured per-agent — but the *threshold* that decides whether
a given germination rate is acceptable is global and lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml


# Default drafter is the smallest cached candidate. The threshold logic
# automatically flips to a larger drafter if calibration shows this one
# can't clear the bar — that's the whole point of the threshold machinery.
DEFAULT_DRAFTER = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Thresholds — locked 2026-05-23 (terra); T1 tightened 2026-05-23 (V) after
# F5 confirmed that gardener.wizard.cli_flow.run_draft_flow auto-retries
# once on T1 parse failure (max_t1_retries=1). With auto-retry, a single
# parse failure is invisible to the user — only a *systematic* failure
# pattern can drive the user-visible rate above 5%. Tighten the default
# to match.
#
# Operators who want the looser bar (e.g. T1=10% as in the (IV) lock) can
# still override via global config: wizard.thresholds.t1_parse_fail_max
# or CLI flag --t1-fail-max.
DEFAULT_T1_PARSE_FAIL_MAX = 0.05
DEFAULT_T2_ACCEPTANCE_FAIL_MAX = 0.20
DEFAULT_T3_GERMINATION_FAIL_MAX = 0.05


def _validate_unit(name: str, value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be in [0.0, 1.0]; got {value!r}"
        )
    return float(value)


@dataclass(frozen=True)
class WizardThresholds:
    """Failure-rate ceilings. Above these, default drafter flips."""

    t1_parse_fail_max: float = DEFAULT_T1_PARSE_FAIL_MAX
    t2_acceptance_fail_max: float = DEFAULT_T2_ACCEPTANCE_FAIL_MAX
    t3_germination_fail_max: float = DEFAULT_T3_GERMINATION_FAIL_MAX

    def __post_init__(self) -> None:
        _validate_unit("t1_parse_fail_max", self.t1_parse_fail_max)
        _validate_unit("t2_acceptance_fail_max", self.t2_acceptance_fail_max)
        _validate_unit("t3_germination_fail_max", self.t3_germination_fail_max)


@dataclass(frozen=True)
class WizardConfig:
    drafter: str = DEFAULT_DRAFTER
    thresholds: WizardThresholds = field(default_factory=WizardThresholds)
    corpus_path: str = "tests/fixtures/wizard_smoke_corpus.yaml"
    # Strata observed but never folded into the threshold pass/fail decision.
    # Adversarial entries are designed to fail — including them would make
    # any drafter fail the threshold by construction.
    exclude_strata_from_threshold: tuple[str, ...] = ("adversarial",)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "WizardConfig":
        """Load wizard config from a YAML file. Missing file → defaults."""
        p = Path(path)
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        wiz = raw.get("wizard", {}) or {}
        return cls._from_dict(wiz)

    @classmethod
    def _from_dict(cls, d: dict) -> "WizardConfig":
        defaults = cls()
        th_raw = d.get("thresholds", {}) or {}
        thresholds = WizardThresholds(
            t1_parse_fail_max=float(
                th_raw.get("t1_parse_fail_max", defaults.thresholds.t1_parse_fail_max)
            ),
            t2_acceptance_fail_max=float(
                th_raw.get(
                    "t2_acceptance_fail_max", defaults.thresholds.t2_acceptance_fail_max
                )
            ),
            t3_germination_fail_max=float(
                th_raw.get(
                    "t3_germination_fail_max",
                    defaults.thresholds.t3_germination_fail_max,
                )
            ),
        )
        corpus_raw = d.get("corpus", {}) or {}
        return cls(
            drafter=str(d.get("drafter", defaults.drafter)),
            thresholds=thresholds,
            corpus_path=str(corpus_raw.get("path", defaults.corpus_path)),
            exclude_strata_from_threshold=tuple(
                corpus_raw.get(
                    "exclude_strata_from_threshold",
                    defaults.exclude_strata_from_threshold,
                )
            ),
        )

    def with_overrides(
        self,
        *,
        drafter: Optional[str] = None,
        t1_fail_max: Optional[float] = None,
        t2_fail_max: Optional[float] = None,
        t3_fail_max: Optional[float] = None,
    ) -> "WizardConfig":
        """Return a new config with CLI overrides applied. None = keep."""
        th = self.thresholds
        new_th = WizardThresholds(
            t1_parse_fail_max=(
                th.t1_parse_fail_max if t1_fail_max is None else t1_fail_max
            ),
            t2_acceptance_fail_max=(
                th.t2_acceptance_fail_max if t2_fail_max is None else t2_fail_max
            ),
            t3_germination_fail_max=(
                th.t3_germination_fail_max if t3_fail_max is None else t3_fail_max
            ),
        )
        return replace(
            self,
            drafter=self.drafter if drafter is None else drafter,
            thresholds=new_th,
        )
