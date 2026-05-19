"""Minimal observability: timer/counter/registry/median_of_n.

Reference map shape: omlx/observability (registry.py summary() / snapshot()).
Re-derived standalone; no omlx import. OMLX_OBSERVABILITY=0 → near-no-op.
"""
from __future__ import annotations

import contextlib
import os
import statistics
import time
from typing import Callable

_ENABLED = os.environ.get("OMLX_OBSERVABILITY", "1") != "0"


class _TimerStats:
    def __init__(self, name: str):
        self.name = name
        self._samples: list[float] = []

    def add(self, seconds: float) -> None:
        self._samples.append(seconds)

    def _pct(self, q: float) -> float:
        if not self._samples:
            return 0.0
        s = sorted(self._samples)
        idx = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
        return s[idx]

    def summary(self) -> dict:
        n = len(self._samples)
        total = sum(self._samples)
        mean = total / n if n else 0.0
        return {
            "name": self.name,
            "count": n,
            "total_s": round(total, 6),
            "mean_ms": round(mean * 1000, 4),
            "min_ms": round(min(self._samples) * 1000, 4) if n else 0.0,
            "p50_ms": round(self._pct(0.50) * 1000, 4),
            "p95_ms": round(self._pct(0.95) * 1000, 4),
            "p99_ms": round(self._pct(0.99) * 1000, 4),
            "max_ms": round(max(self._samples) * 1000, 4) if n else 0.0,
        }


class _Counter:
    def __init__(self, name: str):
        self.name = name
        self.value = 0

    def add(self, n: int = 1) -> None:
        self.value += n

    def summary(self) -> dict:
        return {"name": self.name, "value": self.value}


class _Registry:
    def __init__(self):
        self._timers: dict[str, _TimerStats] = {}
        self._counters: dict[str, _Counter] = {}
        self._started_at = time.perf_counter()

    def timer(self, name: str) -> _TimerStats:
        return self._timers.setdefault(name, _TimerStats(name))

    def counter(self, name: str) -> _Counter:
        return self._counters.setdefault(name, _Counter(name))

    def reset(self) -> None:
        self._timers.clear()
        self._counters.clear()
        self._started_at = time.perf_counter()

    def snapshot(self) -> dict:
        timers = sorted(
            (t.summary() for t in self._timers.values()),
            key=lambda t: t["total_s"], reverse=True,
        )
        counters = sorted(
            (c.summary() for c in self._counters.values()),
            key=lambda c: c["name"],
        )
        return {
            "elapsed_s": round(time.perf_counter() - self._started_at, 4),
            "timers": timers,
            "counters": counters,
        }


registry = _Registry()


@contextlib.contextmanager
def timer(name: str):
    if not _ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        registry.timer(name).add(time.perf_counter() - t0)


def counter(name: str) -> _Counter:
    return registry.counter(name)


def median_of_n(fn: Callable[[], object], name: str, warmup: int = 30,
                 n: int = 60) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        samples.append(dt)
        if _ENABLED:
            registry.timer(name).add(dt)
    return statistics.median(samples) * 1000.0
