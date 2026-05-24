"""Tests for gardener.wizard.germination.aggregate — cross-agent T3 rollup.

The aggregator walks a {name: path} registry mapping, reads each
agent's germination.yaml, and summarizes:
  - per-status counts (pending/passed/failed)
  - per-drafter breakdown (which drafter seeded each)
  - T3 fail rate over decided agents (passed + failed)
  - failure detail (errors per failed agent)
  - unknown list (corrupt/missing germination.yaml — surfaced, never silently dropped)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gardener.wizard.germination import (
    GerminationState,
    OBSERVATION_WINDOW,
    aggregate,
    record_call,
    save,
)


def _plant_agent(
    root: Path,
    name: str,
    *,
    drafted_by: str = "template",
    on_fail: str = "alert",
    call_results: list[bool] = (),
) -> Path:
    """Helper: scaffold an agent dir + write its germination.yaml after N calls."""
    agent_dir = root / name
    agent_dir.mkdir()
    state = GerminationState.new(drafted_by=drafted_by, on_fail=on_fail)
    for ok in call_results:
        state, _ = record_call(state, succeeded=ok, error_reason=None if ok else "empty response")
    save(agent_dir, state)
    return agent_dir


# ---------------------------------------------------------------------------
# Basic aggregation
# ---------------------------------------------------------------------------


def test_aggregate_empty_registry():
    agg = aggregate({})
    assert agg.total_agents == 0
    assert agg.decided_count == 0
    assert agg.t3_germination_fail_rate is None  # no agents → undefined
    assert agg.per_status == {"pending": 0, "passed": 0, "failed": 0}


def test_aggregate_all_pending(tmp_path: Path):
    a = _plant_agent(tmp_path, "a", call_results=[])
    b = _plant_agent(tmp_path, "b", call_results=[True])  # 1 call, still pending
    agg = aggregate({"a": str(a), "b": str(b)})
    assert agg.total_agents == 2
    assert agg.per_status["pending"] == 2
    assert agg.decided_count == 0
    assert agg.t3_germination_fail_rate is None  # no decided → undefined


def test_aggregate_mixed_statuses(tmp_path: Path):
    # 2 passed, 1 failed, 1 pending = 4 total, 3 decided, fail rate 1/3
    _plant_agent(tmp_path, "alpha", call_results=[True, True, True])
    _plant_agent(tmp_path, "beta", call_results=[True, True, True])
    _plant_agent(tmp_path, "gamma", call_results=[True, False, True])  # 1 fail → failed
    _plant_agent(tmp_path, "delta", call_results=[True])  # pending

    registry = {n: str(tmp_path / n) for n in ("alpha", "beta", "gamma", "delta")}
    agg = aggregate(registry)

    assert agg.total_agents == 4
    assert agg.per_status == {"pending": 1, "passed": 2, "failed": 1}
    assert agg.decided_count == 3
    assert agg.t3_germination_fail_rate == pytest.approx(1 / 3)


def test_aggregate_per_drafter_breakdown(tmp_path: Path):
    _plant_agent(tmp_path, "a", drafted_by="qwen-0.5b", call_results=[True, True, True])
    _plant_agent(tmp_path, "b", drafted_by="qwen-0.5b", call_results=[True, False, True])
    _plant_agent(tmp_path, "c", drafted_by="qwen-7b", call_results=[True, True, True])

    registry = {n: str(tmp_path / n) for n in ("a", "b", "c")}
    agg = aggregate(registry)

    assert agg.per_drafter["qwen-0.5b"] == {"pending": 0, "passed": 1, "failed": 1}
    assert agg.per_drafter["qwen-7b"] == {"pending": 0, "passed": 1, "failed": 0}


def test_aggregate_failures_list_includes_errors(tmp_path: Path):
    _plant_agent(
        tmp_path, "broken",
        drafted_by="qwen-0.5b",
        on_fail="regenerate",
        call_results=[False, True, False],
    )
    agg = aggregate({"broken": str(tmp_path / "broken")})
    assert len(agg.failures) == 1
    f = agg.failures[0]
    assert f["agent"] == "broken"
    assert f["drafted_by"] == "qwen-0.5b"
    assert f["on_fail"] == "regenerate"
    assert len(f["errors"]) == 2  # 2 failed calls recorded


def test_aggregate_unknown_for_missing_or_corrupt(tmp_path: Path):
    # Agent dir exists but no germination.yaml → load() returns fresh pending.
    (tmp_path / "fresh").mkdir()
    # Agent dir with a CORRUPT germination.yaml (invalid status).
    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "germination.yaml").write_text("status: invalid-status\n")
    # Agent path doesn't exist at all — load() reads None and returns fresh.
    # Surfaces as "pending" since load is lenient.

    agg = aggregate({"fresh": str(tmp_path / "fresh"), "corrupt": str(corrupt_dir)})
    # The corrupt one is in `unknown`; the fresh one is "pending".
    assert len(agg.unknown) == 1
    assert "corrupt" in agg.unknown[0]
    assert agg.per_status["pending"] == 1  # "fresh"
    assert agg.total_agents == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_aggregate_to_dict_round_trip_shape(tmp_path: Path):
    _plant_agent(tmp_path, "x", call_results=[True, True, True])
    agg = aggregate({"x": str(tmp_path / "x")})
    d = agg.to_dict()
    # All expected keys present.
    for k in (
        "total_agents", "decided_count", "t3_germination_fail_rate",
        "per_status", "per_drafter", "failures", "unknown", "by_agent",
    ):
        assert k in d
    # Round-trippable through JSON.
    import json
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["total_agents"] == 1
