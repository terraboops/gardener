"""gardener calibrate-wizard — measure drafter quality against the smoke corpus.

Runs the configured drafter through the smoke corpus, computes per-stratum
T1 (parse) fail rates + adversarial safety score, applies the threshold
gate, and emits a baseline.json artifact for commit to the repo.

This is a *developer-invoked* command — CI does not run it (CI can't run
MLX inference). The committed baseline.json is what CI consumes to
verify the threshold-comparison LOGIC.

Usage:
    gardener calibrate-wizard \\
        --drafter mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
        --replicas 4 \\
        --output tests/fixtures/wizard_baseline_qwen-0.5b.json

CI-style usage (mock drafter, fast):
    gardener calibrate-wizard --mock --output /tmp/baseline.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from gardener.wizard.adversarial import score_adversarial_batch
from gardener.wizard.config import WizardConfig
from gardener.wizard.drafter import Drafter, MockDrafter
from gardener.wizard.parser import parse
from gardener.wizard.thresholds import (
    BaselineResult,
    Verdict,
    check_thresholds,
    evaluate_baseline,
    load_corpus,
)


def cmd_calibrate_wizard(args) -> int:
    config = _load_config(args.config)
    config = config.with_overrides(
        drafter=getattr(args, "drafter", None),
        t1_fail_max=getattr(args, "t1_fail_max", None),
        t2_fail_max=getattr(args, "t2_fail_max", None),
        t3_fail_max=getattr(args, "t3_fail_max", None),
    )

    corpus_path = Path(args.corpus or config.corpus_path)
    if not corpus_path.exists():
        print(
            f"gardener calibrate-wizard: corpus file not found: {corpus_path}",
            file=sys.stderr,
        )
        return 1
    corpus = load_corpus(corpus_path)

    drafter = _build_drafter(args, config)
    print(
        f"Calibrating drafter={drafter.name!r} against "
        f"{len(corpus)} corpus entries × {args.replicas} replicas "
        f"= {len(corpus) * args.replicas} trials...",
        flush=True,
    )

    baseline = evaluate_baseline(corpus, drafter, replicas=args.replicas)

    # Pull per-stratum adversarial breakdown for the JSON artifact.
    adversarial_breakdown = _adversarial_breakdown(corpus, drafter, args.replicas)

    # Optionally fold post-plant T3 germination data into the verdict.
    # The companion command `gardener germination-status --format json`
    # produces the file this flag consumes — closes the calibration loop.
    t3_external = _maybe_load_t3(
        getattr(args, "t3_from_germination_status", None)
    )
    if t3_external is not None:
        baseline = _with_t3(baseline, t3_external)
        print(
            f"Folded T3 germination fail rate from {args.t3_from_germination_status}: "
            f"{t3_external:.1%}",
            flush=True,
        )

    verdict = check_thresholds(baseline, config)
    _print_verdict(baseline, verdict)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            _serialize(baseline, verdict, adversarial_breakdown, config),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nWrote {out_path}", flush=True)

    if args.exit_nonzero_on_flip and verdict.flip_default:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Optional[str]) -> WizardConfig:
    if path:
        return WizardConfig.from_yaml(Path(path))
    # Default search: ~/.config/gardener/config.yaml then ./gardener.yaml
    candidates = [
        Path.home() / ".config" / "gardener" / "config.yaml",
        Path("gardener.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return WizardConfig.from_yaml(c)
    return WizardConfig()


def _build_drafter(args, config: WizardConfig) -> Drafter:
    if args.mock:
        fixtures_path = Path(
            args.mock_fixtures or "tests/fixtures/mock_drafter_outputs.yaml"
        )
        if not fixtures_path.exists():
            raise FileNotFoundError(
                f"--mock requires fixtures file at {fixtures_path}"
            )
        return MockDrafter.from_yaml(fixtures_path, name=f"mock:{fixtures_path.name}")

    # Real MLX drafter — lazy import keeps this module CI-loadable.
    from gardener.wizard.drafter_mlx import MlxDrafter

    return MlxDrafter(config.drafter)


def _maybe_load_t3(path_str: Optional[str]) -> Optional[float]:
    """Read T3 germination fail rate from a germination-status JSON dump."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"--t3-from-germination-status: file not found: {p}"
        )
    data = json.loads(p.read_text())
    rate = data.get("t3_germination_fail_rate")
    if rate is None:
        # No decided agents — undefined, treat as "no T3 data".
        return None
    return float(rate)


def _with_t3(baseline: BaselineResult, t3: float) -> BaselineResult:
    """Return a new BaselineResult with the supplied T3 fail rate."""
    from dataclasses import replace
    return replace(baseline, t3_germination_fail_rate=t3)


def _adversarial_breakdown(corpus, drafter: Drafter, replicas: int) -> dict:
    """Re-run adversarial entries to record per-sub_type safety stats.

    We re-evaluate (rather than threading through evaluate_baseline) so
    the artifact contains the rich breakdown without complicating the
    BaselineResult dataclass.
    """
    items_by_subtype: dict[str, list] = {}
    for entry in corpus:
        if entry["stratum"] != "adversarial":
            continue
        sub = entry.get("sub_type", "unspecified")
        items_by_subtype.setdefault(sub, [])
        for _ in range(replicas):
            raw = drafter.draft(entry["input"])
            result = parse(raw)
            items_by_subtype[sub].append((entry["input"], result, sub))

    out: dict[str, dict] = {}
    for sub, items in items_by_subtype.items():
        score = score_adversarial_batch(items)
        out[sub] = {
            "total": score.total,
            "safe": score.safe,
            "unsafe": score.unsafe,
            "safe_rate": round(score.safe_rate, 4),
            "by_category": {
                cat.value: count for cat, count in score.by_category.items()
            },
        }
    return out


def _serialize(
    baseline: BaselineResult,
    verdict: Verdict,
    adversarial_breakdown: dict,
    config: WizardConfig,
) -> dict:
    return {
        "drafter": baseline.drafter_name,
        "total_trials": baseline.total_trials,
        "thresholds": {
            "t1_parse_fail_max": config.thresholds.t1_parse_fail_max,
            "t2_acceptance_fail_max": config.thresholds.t2_acceptance_fail_max,
            "t3_germination_fail_max": config.thresholds.t3_germination_fail_max,
        },
        "exclude_strata_from_threshold": list(config.exclude_strata_from_threshold),
        "t1_fail_rate_by_stratum": {
            s: round(r, 4) for s, r in baseline.t1_fail_rate_by_stratum.items()
        },
        "t1_fail_rate_overall": round(baseline.t1_fail_rate_overall, 4),
        "adversarial_safe_rate": round(baseline.adversarial_safe_rate, 4),
        "adversarial_breakdown": adversarial_breakdown,
        "per_stratum_trial_counts": baseline.per_stratum_trial_counts,
        "verdict": {
            "passes": verdict.passes,
            "flip_default": verdict.flip_default,
            "recommended_drafter": verdict.recommended_drafter,
            "reasons": list(verdict.reasons),
            "warnings": list(verdict.warnings),
        },
    }


def _print_verdict(baseline: BaselineResult, verdict: Verdict) -> None:
    print(
        f"\nT1 fail rates by stratum:",
        flush=True,
    )
    for s, r in sorted(baseline.t1_fail_rate_by_stratum.items()):
        n = baseline.per_stratum_trial_counts.get(s, 0)
        print(f"  {s:15s} {r:6.1%}  (n={n})", flush=True)
    print(f"Adversarial safe rate: {baseline.adversarial_safe_rate:.1%}", flush=True)
    print(
        f"\nVerdict: {'PASS' if verdict.passes else 'FAIL'} "
        f"(flip_default={verdict.flip_default})",
        flush=True,
    )
    if verdict.reasons:
        print("Reasons:", flush=True)
        for r in verdict.reasons:
            print(f"  - {r}", flush=True)
    if verdict.warnings:
        print("Warnings:", flush=True)
        for w in verdict.warnings:
            print(f"  ⚠ {w}", flush=True)
