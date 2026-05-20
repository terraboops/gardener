"""Tests for the slim bench harness (HX10).

Pure-logic tests for swap mode classification, config defaults, and gate
threshold logic. Model-bearing smoke test verifies the harness runs end-to-
end on the small test model.
"""
from __future__ import annotations

import json
import pytest

from gardener.bench.harness import (
    BenchConfig,
    BenchReport,
    PhaseResult,
    run_bench,
    _classify_swap_mode,
)


def test_swap_mode_thresholds_match_hypercar_discriminator():
    """Hypercar's CLAUDE.md:46: swap < 5 GB → fast; > 8 GB → slow; else neutral."""
    assert _classify_swap_mode(0.0) == "fast"
    assert _classify_swap_mode(2.5) == "fast"
    assert _classify_swap_mode(4.99) == "fast"
    assert _classify_swap_mode(5.0) == "neutral"
    assert _classify_swap_mode(7.5) == "neutral"
    assert _classify_swap_mode(8.0) == "neutral"
    assert _classify_swap_mode(8.01) == "slow"
    assert _classify_swap_mode(20.0) == "slow"


def test_phase_result_defaults():
    pr = PhaseResult(name="smoke", status="passed", metrics={"x": 1}, elapsed_s=0.1)
    assert pr.error is None


def test_bench_config_defaults():
    cfg = BenchConfig(model_id="x")
    assert "smoke" in cfg.phases
    assert cfg.niah_context_tokens == 4096
    assert cfg.gate_smoke_min_tokens == 4


def test_bench_report_write_json_round_trips(tmp_path):
    r = BenchReport(
        model="x",
        config={"kv_mode": "duo"},
        phases=[
            PhaseResult(
                name="smoke", status="passed",
                metrics={"tokens": 10}, elapsed_s=0.1,
            )
        ],
        swap_mode="fast",
        swap_delta_gb=1.0,
        metal_peak_gb=10.5,
        started_at="2026-05-19T12:00:00",
        elapsed_s=2.0,
    )
    p = tmp_path / "r.json"
    r.write_json(p)
    data = json.loads(p.read_text())
    assert data["model"] == "x"
    assert data["swap_mode"] == "fast"
    assert data["phases"][0]["name"] == "smoke"


def test_bench_report_passed_when_all_phases_passed():
    r = BenchReport(
        model="x", config={},
        phases=[
            PhaseResult(name="smoke", status="passed", metrics={}, elapsed_s=0.1),
            PhaseResult(name="coherence", status="passed", metrics={}, elapsed_s=0.2),
        ],
        swap_mode="fast", swap_delta_gb=0, metal_peak_gb=0,
        started_at="x", elapsed_s=0.3,
    )
    assert r.passed() is True
    assert r.gates_met == ["smoke", "coherence"]
    assert r.gates_failed == []


def test_bench_report_failed_when_any_phase_failed():
    r = BenchReport(
        model="x", config={},
        phases=[
            PhaseResult(name="smoke", status="passed", metrics={}, elapsed_s=0.1),
            PhaseResult(
                name="coherence", status="failed",
                metrics={"correct": 0}, elapsed_s=0.2,
                error="expected '4'",
            ),
        ],
        swap_mode="fast", swap_delta_gb=0, metal_peak_gb=0,
        started_at="x", elapsed_s=0.3,
    )
    assert r.passed() is False
    assert r.gates_failed == ["coherence"]


def test_bench_config_phases_subset():
    cfg = BenchConfig(model_id="x", phases=["smoke", "coherence"])
    assert cfg.phases == ["smoke", "coherence"]
    assert "niah" not in cfg.phases


def test_classify_swap_mode_boundary_exactness():
    """Boundary values: 5.0 is neutral (not fast), 8.0 is neutral (not slow)."""
    assert _classify_swap_mode(5.0) == "neutral"
    assert _classify_swap_mode(8.0) == "neutral"
    # Just over the upper boundary → slow
    assert _classify_swap_mode(8.000001) == "slow"
    # Just under the lower boundary → fast
    assert _classify_swap_mode(4.999999) == "fast"


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------

@pytest.mark.model
def test_smoke_and_coherence_phases_pass_on_test_model():
    """The smoke + coherence + memory_profile phases pass on the standing test
    model. NIAH is skipped here (4K context bench takes >30s)."""
    cfg = BenchConfig(
        model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        phases=["smoke", "coherence", "memory_profile"],
    )
    r = run_bench(cfg)
    assert r.passed(), f"phases: {[(p.name, p.status, p.error) for p in r.phases]}"
    assert "smoke" in r.gates_met
    assert "coherence" in r.gates_met
    assert r.swap_mode in {"fast", "slow", "neutral"}
