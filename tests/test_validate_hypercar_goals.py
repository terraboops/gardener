"""Tests for the hypercar-goal validation report assembler.

Pure-logic only — exercises `assemble_report` against synthetic JSON shaped
like what `gardener.bench.cli` produces. No model load, no real bench run.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The script is at scripts/, not a package — import via path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_hypercar_goals import assemble_report, HYPERCAR_TARGETS  # noqa: E402


def _fake_run(*, swap_mode="fast", swap_delta_gb=2.0, metal_peak_gb=20.0,
              phases=None):
    """Build one fake run record matching gardener.bench.cli output."""
    return {
        "model": "fake",
        "config": {},
        "phases": phases or [],
        "swap_mode": swap_mode,
        "swap_delta_gb": swap_delta_gb,
        "metal_peak_gb": metal_peak_gb,
        "started_at": "2026-05-19T00:00:00",
        "elapsed_s": 1.0,
    }


def _phase(name, status="passed", metrics=None):
    return {"name": name, "status": status,
            "metrics": metrics or {}, "elapsed_s": 0.1, "error": None}


# ---------------------------------------------------------------------------

def test_report_lists_all_six_goals():
    pass1 = {"runs": [_fake_run(phases=[_phase("smoke")])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    for goal in ("**G1**", "**G2**", "**G3**", "**G4**", "**G5**", "**G6**"):
        assert goal in out, f"missing {goal} in report"


def test_g1_marked_skipped_in_quick_mode():
    pass1 = {"runs": [_fake_run()]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    assert "skipped (--quick)" in out
    assert "⏭️" in out


def test_g3_passes_when_decode_above_threshold():
    pass1 = {"runs": [_fake_run(phases=[
        _phase("decode_speed", metrics={"tok_per_sec": 55.0}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    # Find the G3 row.
    g3_line = next(ln for ln in out.splitlines() if "**G3**" in ln)
    assert "✅" in g3_line
    assert "p50=55.0" in g3_line


def test_g3_fails_when_decode_below_threshold():
    pass1 = {"runs": [_fake_run(phases=[
        _phase("decode_speed", metrics={"tok_per_sec": 20.0}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    g3_line = next(ln for ln in out.splitlines() if "**G3**" in ln)
    assert "❌" in g3_line


def test_g6_fails_when_metal_peak_over_48():
    pass1 = {"runs": [_fake_run(metal_peak_gb=50.0)]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    g6_line = next(ln for ln in out.splitlines() if "**G6**" in ln)
    assert "❌" in g6_line
    assert "50.0 GB" in g6_line


def test_g5_passes_when_eighty_pct_fast_mode():
    # 4 of 5 in fast bucket → 80% → passes.
    runs = [_fake_run(swap_mode="fast") for _ in range(4)]
    runs.append(_fake_run(swap_mode="slow", swap_delta_gb=10.0))
    pass1 = {"runs": runs}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    g5_line = next(ln for ln in out.splitlines() if "**G5**" in ln)
    assert "✅" in g5_line
    assert "4/5 fast" in g5_line


def test_g2_humaneval_passes_at_or_above_threshold():
    pass1 = {"runs": [_fake_run(phases=[
        _phase("humaneval_lite", metrics={"pass_rate": 0.40}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    he_line = next(ln for ln in out.splitlines() if "HumanEval-Lite" in ln)
    assert "✅" in he_line
    assert "40.0%" in he_line


def test_g2_ruler_reports_both_thresholds():
    pass1 = {"runs": [_fake_run(phases=[
        _phase("ruler", metrics={"mk_16k_accuracy": 0.85, "vt_4k_accuracy": 0.75}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    assert "RULER multi-key" in out
    assert "RULER variable-tracking" in out


def test_g1_passes_when_niah_succeeds_in_pass2():
    pass1 = {"runs": [_fake_run()]}
    pass2 = {"runs": [_fake_run(phases=[
        _phase("niah", metrics={"found": True, "context_tokens": 524288}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=pass2, quick=False)
    g1_line = next(ln for ln in out.splitlines() if "**G1**" in ln)
    assert "✅" in g1_line
    assert "524,288" in g1_line


def test_g1_fails_when_niah_status_failed_in_pass2():
    pass1 = {"runs": [_fake_run()]}
    pass2 = {"runs": [_fake_run(phases=[
        _phase("niah", status="failed",
                metrics={"found": False, "context_tokens": 524288}),
    ])]}
    out = assemble_report(model="fake", pass1=pass1, pass2=pass2, quick=False)
    g1_line = next(ln for ln in out.splitlines() if "**G1**" in ln)
    assert "❌" in g1_line


def test_swap_stratification_groups_runs_by_mode():
    runs = [
        _fake_run(swap_mode="fast", metal_peak_gb=20.0),
        _fake_run(swap_mode="fast", metal_peak_gb=21.0),
        _fake_run(swap_mode="slow", metal_peak_gb=30.0),
    ]
    pass1 = {"runs": runs}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    assert "## Swap-mode stratification" in out
    # Both buckets appear in the table.
    assert "| fast |" in out or " fast " in out
    assert "| slow |" in out or " slow " in out


def test_report_includes_honest_caveats_section():
    pass1 = {"runs": [_fake_run()]}
    out = assemble_report(model="fake", pass1=pass1, pass2=None, quick=True)
    assert "## Honest caveats" in out
    assert "MInference" in out
    assert "speculative decoding" in out.lower()
    assert "TQ3" in out
