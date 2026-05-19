from gardener.mlxsuper.observability import timer, counter, registry, median_of_n

def test_timer_and_counter_record_into_registry():
    registry.reset()
    with timer("unit.work"):
        sum(range(1000))
    counter("unit.events").add(3)
    counter("unit.events").add(2)
    snap = registry.snapshot()
    names = {t["name"] for t in snap["timers"]}
    assert "unit.work" in names
    work = next(t for t in snap["timers"] if t["name"] == "unit.work")
    assert work["count"] == 1
    assert {"p50_ms", "p95_ms", "p99_ms", "mean_ms"} <= set(work)
    ev = next(c for c in snap["counters"] if c["name"] == "unit.events")
    assert ev["value"] == 5

def test_median_of_n_returns_float_and_records():
    registry.reset()
    ms = median_of_n(lambda: sum(range(100)), name="unit.median", warmup=2, n=5)
    assert isinstance(ms, float) and ms >= 0.0
    snap = registry.snapshot()
    assert any(t["name"] == "unit.median" for t in snap["timers"])
