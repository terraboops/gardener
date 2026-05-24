"""Tests for gardener/wizard/thresholds.py — baseline + threshold verdict.

This module measures T1 (parse) failure rate per stratum against a smoke
corpus, scores the adversarial stratum separately, and decides whether
the current drafter should remain the default or be flipped.

T3 (germination) CANNOT be measured from a drafter + corpus — it requires
post-plant runtime telemetry (slice I). The threshold module accepts an
optional T3 fail-rate input from external aggregation; if absent, the
verdict is T1-only.
"""
from __future__ import annotations

import pytest

from gardener.wizard.config import WizardConfig, WizardThresholds
from gardener.wizard.drafter import MockDrafter
from gardener.wizard.thresholds import (
    BaselineResult,
    Verdict,
    check_thresholds,
    evaluate_baseline,
)


# ---------------------------------------------------------------------------
# Small in-test corpus + fixture pairs
# ---------------------------------------------------------------------------


def _corpus_pass():
    """All entries produce valid sectioned-markdown output."""
    return [
        {"input": "narrow-1", "stratum": "narrow"},
        {"input": "narrow-2", "stratum": "narrow"},
        {"input": "broad-1", "stratum": "broad"},
        {"input": "adv-1", "stratum": "adversarial", "sub_type": "vacuous"},
    ]


def _fixtures_pass():
    return [
        {
            "input": "narrow-1",
            "stratum": "narrow",
            "output": "### System Prompt\nA helper.",
        },
        {
            "input": "narrow-2",
            "stratum": "narrow",
            "output": "### System Prompt\nAnother helper.",
        },
        {
            "input": "broad-1",
            "stratum": "broad",
            "output": "### System Prompt\nGeneralist.",
        },
        {
            "input": "adv-1",
            "stratum": "adversarial",
            "sub_type": "vacuous",
            "output": "### Decline\nCan't seed from this.",
        },
    ]


# ---------------------------------------------------------------------------
# evaluate_baseline
# ---------------------------------------------------------------------------


def test_evaluate_baseline_perfect_run():
    corpus = _corpus_pass()
    drafter = MockDrafter(fixtures=_fixtures_pass())
    baseline = evaluate_baseline(corpus, drafter, replicas=1)

    assert baseline.total_trials == 4
    assert baseline.drafter_name == drafter.name

    # Per-stratum T1 fail rates: all zero on this fixture.
    assert baseline.t1_fail_rate_by_stratum["narrow"] == pytest.approx(0.0)
    assert baseline.t1_fail_rate_by_stratum["broad"] == pytest.approx(0.0)
    # Adversarial T1 also recorded (Decline parses OK).
    assert baseline.t1_fail_rate_by_stratum["adversarial"] == pytest.approx(0.0)

    # Adversarial safety score is reported separately.
    assert baseline.adversarial_safe_rate == pytest.approx(1.0)


def test_evaluate_baseline_with_parse_failures():
    corpus = [
        {"input": "good", "stratum": "narrow"},
        {"input": "bad", "stratum": "narrow"},
        {"input": "broad-bad", "stratum": "broad"},
    ]
    fixtures = [
        {"input": "good", "stratum": "narrow", "output": "### System Prompt\nOK."},
        {"input": "bad", "stratum": "narrow", "output": "no sections at all"},
        {"input": "broad-bad", "stratum": "broad", "output": "also no sections"},
    ]
    baseline = evaluate_baseline(corpus, MockDrafter(fixtures=fixtures))
    # narrow: 1/2 fail = 0.5
    assert baseline.t1_fail_rate_by_stratum["narrow"] == pytest.approx(0.5)
    # broad: 1/1 fail = 1.0
    assert baseline.t1_fail_rate_by_stratum["broad"] == pytest.approx(1.0)


def test_evaluate_baseline_with_replicas():
    # Multi-output fixture cycles: output 1 parses, output 2 doesn't.
    corpus = [{"input": "flaky", "stratum": "broad"}]
    fixtures = [
        {
            "input": "flaky",
            "stratum": "broad",
            "outputs": [
                "### System Prompt\nOK.",
                "garbage no sections",
                "### System Prompt\nOK again.",
                "more garbage",
            ],
        }
    ]
    baseline = evaluate_baseline(corpus, MockDrafter(fixtures=fixtures), replicas=4)
    # 2 of 4 parse → 0.5 fail rate
    assert baseline.t1_fail_rate_by_stratum["broad"] == pytest.approx(0.5)
    assert baseline.total_trials == 4


def test_evaluate_baseline_adversarial_safety_separate_from_t1():
    # Adversarial: drafter parses OK (returns a System Prompt section) but
    # was UNSAFE (confidently seeded from vacuous input).
    corpus = [{"input": "", "stratum": "adversarial", "sub_type": "vacuous"}]
    fixtures = [
        {
            "input": "",
            "stratum": "adversarial",
            "output": "### System Prompt\nHelper.",
        }
    ]
    baseline = evaluate_baseline(corpus, MockDrafter(fixtures=fixtures))
    # T1 fail = 0 (parse succeeded)
    assert baseline.t1_fail_rate_by_stratum["adversarial"] == pytest.approx(0.0)
    # Adversarial safety = 0 (was UNSAFE_CONFIDENT_WRONG)
    assert baseline.adversarial_safe_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_thresholds — Verdict
# ---------------------------------------------------------------------------


def test_check_thresholds_all_pass():
    cfg = WizardConfig()  # defaults: T1 max 10%
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=4,
        t1_fail_rate_by_stratum={"narrow": 0.0, "broad": 0.05},
        t1_fail_rate_overall=0.02,
        adversarial_safe_rate=1.0,
        per_stratum_trial_counts={"narrow": 2, "broad": 2},
    )
    v = check_thresholds(baseline, cfg)
    assert v.passes
    assert v.recommended_drafter == "qwen-0.5b"
    assert v.flip_default is False


def test_check_thresholds_t1_overshoot_flips_default():
    cfg = WizardConfig()  # T1 max 10%
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=20,
        t1_fail_rate_by_stratum={"narrow": 0.20, "broad": 0.05},  # narrow overshoots
        t1_fail_rate_overall=0.13,
        adversarial_safe_rate=1.0,
        per_stratum_trial_counts={"narrow": 10, "broad": 10},
    )
    v = check_thresholds(baseline, cfg)
    assert not v.passes
    assert v.flip_default is True
    # Verdict should name WHICH stratum failed for debuggability.
    assert any("narrow" in r and "0.20" in r for r in v.reasons)


def test_check_thresholds_excludes_adversarial_by_default():
    # Adversarial T1 is huge but should NOT trigger a flip since the
    # stratum is excluded from threshold gates by default.
    cfg = WizardConfig()
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=8,
        t1_fail_rate_by_stratum={"narrow": 0.0, "adversarial": 0.80},
        t1_fail_rate_overall=0.40,
        adversarial_safe_rate=0.90,
        per_stratum_trial_counts={"narrow": 4, "adversarial": 4},
    )
    v = check_thresholds(baseline, cfg)
    assert v.passes
    assert v.flip_default is False


def test_check_thresholds_warns_when_resolution_below_threshold():
    # T1 max = 5%; with only 10 trials per stratum, resolution is 10%.
    cfg = WizardConfig().with_overrides(t1_fail_max=0.05)
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=20,
        t1_fail_rate_by_stratum={"narrow": 0.0, "broad": 0.0},
        t1_fail_rate_overall=0.0,
        adversarial_safe_rate=1.0,
        per_stratum_trial_counts={"narrow": 10, "broad": 10},
    )
    v = check_thresholds(baseline, cfg)
    assert v.passes
    assert any(
        "resolution" in w.lower() and "below threshold" in w.lower()
        for w in v.warnings
    )


def test_check_thresholds_reports_adversarial_safety_as_warning_not_gate():
    cfg = WizardConfig()
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=4,
        t1_fail_rate_by_stratum={"narrow": 0.0, "adversarial": 0.0},
        t1_fail_rate_overall=0.0,
        adversarial_safe_rate=0.40,  # 60% of adversarial inputs got UNSAFE
        per_stratum_trial_counts={"narrow": 2, "adversarial": 2},
    )
    v = check_thresholds(baseline, cfg)
    # T1 is fine → passes overall, even though adversarial safety is poor.
    assert v.passes
    # But it MUST surface as a warning so a human notices.
    assert any("adversarial" in w.lower() and "safe" in w.lower() for w in v.warnings)


def test_check_thresholds_t3_overshoot_flips_default():
    # T3 cannot be computed from drafter + corpus; it's fed in from
    # post-plant telemetry. Verify the threshold check honors it when
    # supplied.
    cfg = WizardConfig()  # T3 max 5%
    baseline = BaselineResult(
        drafter_name="qwen-0.5b",
        total_trials=20,
        t1_fail_rate_by_stratum={"narrow": 0.0},
        t1_fail_rate_overall=0.0,
        adversarial_safe_rate=1.0,
        per_stratum_trial_counts={"narrow": 20},
        t3_germination_fail_rate=0.15,  # 15% — way over 5%
    )
    v = check_thresholds(baseline, cfg)
    assert not v.passes
    assert v.flip_default is True
    assert any("germination" in r.lower() and "0.15" in r for r in v.reasons)


# ---------------------------------------------------------------------------
# Smoke corpus loading
# ---------------------------------------------------------------------------


def test_load_smoke_corpus_from_yaml():
    from gardener.wizard.thresholds import load_corpus
    from pathlib import Path

    corpus = load_corpus(Path("tests/fixtures/wizard_smoke_corpus.yaml"))
    # Spec mandates ≥20 entries (or 10×4 replicas; for the file itself, ≥20).
    assert len(corpus) >= 20, f"smoke corpus has {len(corpus)} entries; need ≥20"
    # Every entry has stratum.
    for entry in corpus:
        assert "input" in entry
        assert "stratum" in entry
        if entry["stratum"] == "adversarial":
            assert "sub_type" in entry, (
                f"adversarial entry {entry['input']!r} missing sub_type"
            )
    # All four strata represented.
    strata = {e["stratum"] for e in corpus}
    assert strata == {"narrow", "broad", "multi-domain", "adversarial"}


def test_smoke_corpus_inputs_have_mock_fixtures():
    """Every smoke-corpus entry must have a corresponding mock-drafter fixture.

    Otherwise CI runs against the corpus will hit DrafterReplayMiss.
    """
    from gardener.wizard.thresholds import load_corpus
    from pathlib import Path

    corpus = load_corpus(Path("tests/fixtures/wizard_smoke_corpus.yaml"))
    drafter = MockDrafter.from_yaml("tests/fixtures/mock_drafter_outputs.yaml")
    corpus_inputs = {e["input"] for e in corpus}
    fixture_inputs = set(drafter.inputs())
    missing = corpus_inputs - fixture_inputs
    assert not missing, (
        f"smoke corpus has inputs without mock fixtures:\n  "
        + "\n  ".join(sorted(repr(m) for m in missing))
    )
