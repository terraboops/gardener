"""Baseline measurement + threshold-comparison verdict logic.

T1 (Parse) is measured here against a smoke corpus via any Drafter.
Adversarial safety is scored separately and reported alongside but NEVER
folded into the T1 gate.

T3 (Germination) cannot be measured from drafter output alone — it
requires post-plant runtime telemetry from real agent invocations
(slice I). When external aggregation supplies a T3 fail rate via
BaselineResult.t3_germination_fail_rate, this module honors it in the
threshold check.

T2 (Acceptance) is wizard-runtime UX telemetry; logged but never a
switch-trigger per terra's 2026-05-23 spec.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml

from .adversarial import classify_adversarial
from .config import WizardConfig
from .drafter import Drafter
from .parser import parse


@dataclass(frozen=True)
class BaselineResult:
    drafter_name: str
    total_trials: int
    t1_fail_rate_by_stratum: dict  # stratum -> float
    t1_fail_rate_overall: float
    adversarial_safe_rate: float
    per_stratum_trial_counts: dict  # stratum -> int
    # External post-plant input; None when calibrating without runtime data.
    t3_germination_fail_rate: Optional[float] = None


@dataclass(frozen=True)
class Verdict:
    passes: bool
    flip_default: bool
    recommended_drafter: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def load_corpus(path: Path | str) -> list[dict]:
    """Load wizard_smoke_corpus.yaml. Each entry: {input, stratum, sub_type?}."""
    raw = yaml.safe_load(Path(path).read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"corpus file {path!r} must be a YAML list")
    return list(raw)


def evaluate_baseline(
    corpus: Sequence[dict],
    drafter: Drafter,
    *,
    replicas: int = 1,
) -> BaselineResult:
    """Run each corpus entry through the drafter `replicas` times.

    Returns per-stratum T1 fail rates + the adversarial safety score.
    Does NOT compute T3 (germination) — that's post-plant.
    """
    if replicas < 1:
        raise ValueError(f"replicas must be ≥ 1; got {replicas}")

    fail_counts: Counter[str] = Counter()
    trial_counts: Counter[str] = Counter()
    adversarial_items: list[tuple[str, object, Optional[str]]] = []
    total_trials = 0
    total_fails = 0

    for entry in corpus:
        purpose = entry["input"]
        stratum = entry["stratum"]
        sub_type = entry.get("sub_type")
        for _ in range(replicas):
            raw = drafter.draft(purpose)
            result = parse(raw)
            trial_counts[stratum] += 1
            total_trials += 1
            if not result.ok:
                fail_counts[stratum] += 1
                total_fails += 1
            if stratum == "adversarial":
                adversarial_items.append((purpose, result, sub_type))

    t1_by = {
        s: (fail_counts[s] / trial_counts[s]) if trial_counts[s] else 0.0
        for s in trial_counts
    }
    overall = (total_fails / total_trials) if total_trials else 0.0

    if adversarial_items:
        from .adversarial import score_adversarial_batch
        score = score_adversarial_batch(adversarial_items)
        adv_safe = score.safe_rate
    else:
        adv_safe = 1.0  # vacuously safe

    return BaselineResult(
        drafter_name=drafter.name,
        total_trials=total_trials,
        t1_fail_rate_by_stratum=t1_by,
        t1_fail_rate_overall=overall,
        adversarial_safe_rate=adv_safe,
        per_stratum_trial_counts=dict(trial_counts),
    )


def check_thresholds(
    baseline: BaselineResult,
    config: WizardConfig,
    *,
    fallback_drafter: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
) -> Verdict:
    """Apply threshold gates to a baseline; return a switch verdict.

    Strata listed in config.exclude_strata_from_threshold are observed
    but not gated (adversarial by default).
    """
    excluded = set(config.exclude_strata_from_threshold)
    reasons: list[str] = []
    warnings: list[str] = []
    fails = False

    # --- T1 per-stratum --------------------------------------------------
    t1_max = config.thresholds.t1_parse_fail_max
    for stratum, rate in baseline.t1_fail_rate_by_stratum.items():
        if stratum in excluded:
            continue
        if rate > t1_max:
            fails = True
            reasons.append(
                f"T1 parse fail rate on stratum {stratum!r} = {rate:.2f} "
                f"exceeds threshold {t1_max:.2f}"
            )

    # --- T3 (external) ---------------------------------------------------
    if baseline.t3_germination_fail_rate is not None:
        t3_max = config.thresholds.t3_germination_fail_max
        if baseline.t3_germination_fail_rate > t3_max:
            fails = True
            reasons.append(
                f"T3 germination fail rate = "
                f"{baseline.t3_germination_fail_rate:.2f} exceeds threshold "
                f"{t3_max:.2f}"
            )

    # --- Statistical-resolution warnings --------------------------------
    for stratum, n in baseline.per_stratum_trial_counts.items():
        if stratum in excluded or n == 0:
            continue
        resolution = 1.0 / n
        if resolution > t1_max:
            warnings.append(
                f"stratum {stratum!r}: resolution 1/{n} = {resolution:.2f} "
                f"is below threshold {t1_max:.2f} — can't measure failures "
                f"finer than {resolution:.2%}"
            )

    # --- Adversarial safety (observability, not gate) -------------------
    if baseline.adversarial_safe_rate < 1.0:
        warnings.append(
            f"adversarial safe rate = {baseline.adversarial_safe_rate:.2f} "
            f"(< 1.0); see adversarial-stratum breakdown for unsafe cases"
        )

    passes = not fails
    return Verdict(
        passes=passes,
        flip_default=(not passes),
        recommended_drafter=(
            baseline.drafter_name if passes else fallback_drafter
        ),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
