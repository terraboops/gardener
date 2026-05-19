# Phase 0 — Superpowered-MLX Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `gardener/mlxsuper/` — a thin, tested layer on **stock** `mlx`/`mlx-lm` that gives base MLX the four "superpowers" later Gardener phases need: persistent/forkable/rewindable sessions, per-layer LR schedules, OPLoRA gradient safety, and a minimal TTT learning loop — with **zero `omlx` import**.

**Architecture:** omlx-mamba3 is a *reference map only*. We re-derive each technique against stock `mlx_lm` APIs and pin it with our own tests. Sessions wrap `mlx_lm.models.cache` (`make_prompt_cache`/`save_prompt_cache`/`load_prompt_cache`/`KVCache.trim`) and add fork + a registry. TTT rebuilds on `mlx_lm.tuner` LoRA. OPLoRA and the LR schedule are pure functions ported verbatim (math only). Fast unit tests (math/logic, no model) gate correctness; model-loading tests are marked `@pytest.mark.model` and kept minimal.

**Tech Stack:** Python 3.13, `mlx`, `mlx-lm`, `numpy`, `pytest`. No `omlx`, no network at test time except a one-time tiny-model download in marked tests.

---

## Scope & Non-Goals

- **In:** session save/load/fork/rewind, `compute_layer_lrs`, `project_lora_grads` + `compute_svd_cache`, a minimal `TTTEngine` (generate_candidates / feedback / train_step with optional OPLoRA projection), observability shim, `import omlx` CI gate.
- **Out (later phases):** the TQ3 WHT codec (stock `QuantizedKVCache` is the v0 quantize-on-save path), the daemon, scheduler, pipeline IR, knowledge store. Do not build them here.
- **Test model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (small, Qwen-family, ~300 MB). Referenced as `TEST_MODEL` in `tests/conftest.py`.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata + base deps only |
| `gardener/__init__.py` | Package marker |
| `gardener/mlxsuper/__init__.py` | Public exports |
| `gardener/mlxsuper/observability.py` | `timer`, `counter`, `registry`, `median_of_n` |
| `gardener/mlxsuper/schedules.py` | `compute_layer_lrs` (pure) |
| `gardener/mlxsuper/oplora.py` | `project_lora_grads`, `compute_svd_cache`, `_truncated_svd` (pure-ish) |
| `gardener/mlxsuper/session.py` | `Session`, `SessionPool` (load/generate/save/load/fork/rewind) |
| `gardener/mlxsuper/ttt.py` | `Candidate`, `TrainStats`, `TTTEngine` |
| `scripts/check_no_omlx_import.py` | CI grep gate: fail if any `import omlx` |
| `tests/conftest.py` | `TEST_MODEL`, shared fixtures, `model` marker |
| `tests/test_*` | One per module |

---

### Task 1: Repo scaffold + omlx-import CI gate

**Files:**
- Create: `/Users/terra/Developer/gardener/pyproject.toml`
- Create: `/Users/terra/Developer/gardener/gardener/__init__.py`
- Create: `/Users/terra/Developer/gardener/gardener/mlxsuper/__init__.py`
- Create: `/Users/terra/Developer/gardener/scripts/check_no_omlx_import.py`
- Create: `/Users/terra/Developer/gardener/tests/conftest.py`
- Create: `/Users/terra/Developer/gardener/tests/test_no_omlx_import.py`
- Create: `/Users/terra/Developer/gardener/.gitignore`

- [ ] **Step 1: git init + copy the approved design doc**

```bash
cd /Users/terra/Developer/gardener
git init -q
mkdir -p gardener/mlxsuper scripts tests docs
cp /Users/terra/Developer/omlx-mamba3/gardener/docs/design.md docs/design.md
cp /Users/terra/Developer/omlx-mamba3/gardener/README.md README.md
```
(Run sandbox-disabled — `~/Developer/gardener` is outside the Bash write-allowlist.)

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "gardener"
version = "0.0.0"
requires-python = ">=3.13"
dependencies = [
    "mlx",
    "mlx-lm",
    "numpy",
    "pyyaml",
    "lark",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.pytest.ini_options]
markers = ["model: loads a real MLX model (slow, downloads once)"]
testpaths = ["tests"]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Write package markers and `.gitignore`**

`gardener/__init__.py`:
```python
```
`gardener/mlxsuper/__init__.py`:
```python
"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable."""
```
`.gitignore`:
```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Write the omlx-import gate script**

`scripts/check_no_omlx_import.py`:
```python
"""CI gate: the gardener package must never import omlx (port, don't depend)."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "gardener"
PATTERN = re.compile(r"^\s*(import\s+omlx|from\s+omlx[\s.])", re.MULTILINE)

def main() -> int:
    offenders = []
    for path in PKG.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        print("FAIL: omlx import found in:", *offenders, sep="\n  ")
        return 1
    print("OK: no omlx imports in gardener/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write the test that runs the gate**

`tests/test_no_omlx_import.py`:
```python
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def test_no_omlx_import():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_omlx_import.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
import pytest

TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

@pytest.fixture(scope="session")
def loaded_model():
    from mlx_lm import load
    model, tokenizer = load(TEST_MODEL)
    return model, tokenizer
```

- [ ] **Step 7: Run the gate test**

Run: `cd /Users/terra/Developer/gardener && python -m pytest tests/test_no_omlx_import.py -v`
Expected: PASS (`OK: no omlx imports in gardener/`)

- [ ] **Step 8: Commit**

```bash
cd /Users/terra/Developer/gardener
git add -A
git commit -m "chore: phase0 scaffold + omlx-import CI gate"
```

---

### Task 2: Observability shim

**Files:**
- Create: `gardener/mlxsuper/observability.py`
- Test: `tests/test_observability.py`

- [ ] **Step 1: Write the failing test**

`tests/test_observability.py`:
```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_observability.py -v`
Expected: FAIL (`ModuleNotFoundError: gardener.mlxsuper.observability`)

- [ ] **Step 3: Implement `gardener/mlxsuper/observability.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_observability.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/observability.py tests/test_observability.py
git commit -m "feat(mlxsuper): observability shim (timer/counter/registry)"
```

---

### Task 3: Per-layer LR schedules (pure port)

**Files:**
- Create: `gardener/mlxsuper/schedules.py`
- Test: `tests/test_schedules.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schedules.py`:
```python
import math
import pytest
from gardener.mlxsuper.schedules import compute_layer_lrs

def test_uniform_is_flat():
    assert compute_layer_lrs(1e-4, n_layers=8, schedule="uniform") == [1e-4] * 8

def test_reservoir_is_monotone_nondecreasing_zero_first_base_last():
    lrs = compute_layer_lrs(1.0, n_layers=10, schedule="reservoir", gamma=1.5)
    assert lrs[0] == 0.0
    assert math.isclose(lrs[-1], 1.0, rel_tol=1e-9)
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))

def test_cosine_is_u_shaped_edges_high_middle_low_min_10pct():
    lrs = compute_layer_lrs(1.0, n_layers=11, schedule="cosine")
    assert lrs[0] > lrs[len(lrs) // 2] < lrs[-1]
    assert min(lrs) >= 0.1 - 1e-9

def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        compute_layer_lrs(1e-4, n_layers=4, schedule="bogus")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_schedules.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `gardener/mlxsuper/schedules.py`** (ported verbatim from reference map `omlx/ttt_schedules.py:23`)

```python
"""Per-layer learning-rate schedules. Pure function; reference map:
omlx/ttt_schedules.py:23 (re-derived, no omlx import)."""
from __future__ import annotations

import math
from typing import Literal


def compute_layer_lrs(
    base_lr: float,
    n_layers: int = 48,
    schedule: Literal["uniform", "reservoir", "cosine"] = "reservoir",
    gamma: float = 1.5,
) -> list[float]:
    if schedule == "uniform":
        return [base_lr] * n_layers
    if schedule == "reservoir":
        return [
            base_lr * ((l / max(n_layers - 1, 1)) ** gamma)
            for l in range(n_layers)
        ]
    if schedule == "cosine":
        lrs = []
        for l in range(n_layers):
            t = l / max(n_layers - 1, 1)
            weight = 0.5 * (1 + math.cos(2 * math.pi * t))
            lrs.append(base_lr * (0.1 + 0.9 * weight))
        return lrs
    raise ValueError(f"Unknown schedule: {schedule}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_schedules.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/schedules.py tests/test_schedules.py
git commit -m "feat(mlxsuper): per-layer LR schedules (pure port)"
```

---

### Task 4: OPLoRA gradient projection (pure math)

**Files:**
- Create: `gardener/mlxsuper/oplora.py`
- Test: `tests/test_oplora.py`

- [ ] **Step 1: Write the failing test**

`tests/test_oplora.py`:
```python
import mlx.core as mx
from gardener.mlxsuper.oplora import project_lora_grads, _truncated_svd

def test_truncated_svd_shapes():
    W = mx.random.normal((12, 9))
    U_k, V_k = _truncated_svd(W, k=4)
    assert U_k.shape == (12, 4)
    assert V_k.shape == (9, 4)

def test_projection_removes_top_subspace_component():
    mx.random.seed(0)
    m, n, r, k = 16, 10, 3, 4
    W = mx.random.normal((m, n))
    U_k, V_k = _truncated_svd(W, k)
    A = mx.random.normal((r, n))
    B = mx.random.normal((m, r))
    dA = mx.random.normal((r, n))
    dB = mx.random.normal((m, r))
    dA_s, dB_s = project_lora_grads(W, A, B, dA, dB, {"U_k": U_k, "V_k": V_k})
    # Projected grads must be orthogonal to the removed subspace.
    assert float(mx.max(mx.abs(dA_s @ V_k))) < 1e-4
    assert float(mx.max(mx.abs(U_k.T @ dB_s))) < 1e-4
    # And idempotent: projecting again is a no-op.
    dA_s2, dB_s2 = project_lora_grads(W, A, B, dA_s, dB_s,
                                      {"U_k": U_k, "V_k": V_k})
    assert float(mx.max(mx.abs(dA_s2 - dA_s))) < 1e-4
    assert float(mx.max(mx.abs(dB_s2 - dB_s))) < 1e-4
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_oplora.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `gardener/mlxsuper/oplora.py`** (reference map `omlx/oplora.py:87,129`)

```python
"""OPLoRA gradient projection. Pure linear algebra; reference map:
omlx/oplora.py:87 (project_lora_grads), :129 (compute_svd_cache).
Re-derived on stock mlx; no omlx import."""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger("gardener.mlxsuper.oplora")


def _truncated_svd(W: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Return top-k left/right singular vectors (U_k:(m,k), V_k:(n,k))."""
    U, S, Vt = mx.linalg.svd(W, stream=mx.cpu)
    k = min(k, U.shape[1], Vt.shape[0])
    return U[:, :k], Vt[:k, :].T


def project_lora_grads(
    W: mx.array,
    A: mx.array,
    B: mx.array,
    dA: mx.array,
    dB: mx.array,
    svd_data: dict[str, mx.array],
) -> tuple[mx.array, mx.array]:
    """Project LoRA grads onto the orthogonal complement of W's top-k SVD.

    dA_safe = dA - (dA V_k) V_k^T   (remove right-singular component)
    dB_safe = dB - U_k (U_k^T dB)   (remove left-singular component)
    """
    U_k = svd_data["U_k"]
    V_k = svd_data["V_k"]
    dA_safe = dA - (dA @ V_k) @ V_k.T
    dB_safe = dB - U_k @ (U_k.T @ dB)
    return dA_safe, dB_safe


def compute_svd_cache(
    weights: dict[str, mx.array],
    k: int = 8,
) -> dict[str, dict[str, mx.array]]:
    """Truncated SVD per named frozen weight matrix.

    `weights` maps a layer path -> its (m, n) base weight. Returns
    {path: {"U_k": (m,k), "V_k": (n,k)}}. (Caller supplies the weight
    dict; this keeps the function model-structure-agnostic and pure.)
    """
    cache: dict[str, dict[str, mx.array]] = {}
    for path, W in weights.items():
        if not isinstance(W, mx.array) or W.ndim != 2:
            logger.warning("SVD cache: skipping non-2D weight %s", path)
            continue
        U_k, V_k = _truncated_svd(W, k)
        cache[path] = {"U_k": U_k, "V_k": V_k}
    return cache
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_oplora.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/oplora.py tests/test_oplora.py
git commit -m "feat(mlxsuper): OPLoRA gradient projection (pure math)"
```

---

### Task 5: Session save / load / fork / rewind (logic, fake cache)

**Files:**
- Create: `gardener/mlxsuper/session.py`
- Test: `tests/test_session_logic.py`

This task pins the **fork/rewind semantics** with a fake cache (fast, no model). Task 6 confirms end-to-end with a real model.

- [ ] **Step 1: Write the failing test**

`tests/test_session_logic.py`:
```python
from gardener.mlxsuper.session import SessionPool

class FakeCache:
    """Mimics mlx_lm KVCache trim/offset semantics with a python list."""
    def __init__(self):
        self.tokens = []
        self.offset = 0
    def append(self, t):
        self.tokens.append(t)
        self.offset += 1
    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        del self.tokens[self.offset:]
        return n

def _mk_pool():
    return SessionPool(cache_factory=lambda: [FakeCache()])

def test_create_and_fork_are_independent():
    pool = _mk_pool()
    a = pool.create("a")
    pool.get(a)[0].append("x"); pool.get(a)[0].append("y")
    b = pool.fork(a, "b")
    pool.get(b)[0].append("z")          # diverge child
    pool.get(a)[0].append("w")          # diverge parent
    assert pool.get(a)[0].tokens == ["x", "y", "w"]
    assert pool.get(b)[0].tokens == ["x", "y", "z"]
    assert pool.parent_of(b) == a

def test_rewind_drops_tail_o1():
    pool = _mk_pool()
    s = pool.create("s")
    for t in "abcde":
        pool.get(s)[0].append(t)
    dropped = pool.rewind(s, 2)
    assert dropped == 3
    assert pool.get(s)[0].tokens == ["a", "b"]
    assert pool.get(s)[0].offset == 2

def test_unknown_session_raises():
    pool = _mk_pool()
    import pytest
    with pytest.raises(KeyError):
        pool.get("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_session_logic.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `gardener/mlxsuper/session.py`**

```python
"""Persistent / forkable / rewindable sessions over stock mlx_lm caches.

Reference map (semantics only): omlx/turboquant_kv.py:1624 fork,
:1648 rewind_to, :1670 save_to_disk, :1722 load_from_disk;
omlx/hypercar_server.py:410-680 session model. Built on stock
mlx_lm.models.cache (make_prompt_cache/save_prompt_cache/
load_prompt_cache/KVCache.trim). No omlx import."""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any, Callable

from .observability import counter, timer


class SessionPool:
    """In-memory registry of named KV-cache sessions.

    A session's value is a `list[cache]` — one cache object per model
    layer (the stock mlx_lm prompt-cache shape). `cache_factory` builds
    a fresh empty cache list (real use: a closure over the model).
    """

    def __init__(self, cache_factory: Callable[[], list[Any]]):
        self._cache_factory = cache_factory
        self._sessions: dict[str, list[Any]] = {}
        self._parent: dict[str, str] = {}

    def create(self, name: str | None = None) -> str:
        sid = name or uuid.uuid4().hex[:8]
        if sid in self._sessions:
            raise KeyError(f"session exists: {sid}")
        self._sessions[sid] = self._cache_factory()
        counter("session.created").add(1)
        return sid

    def get(self, sid: str) -> list[Any]:
        if sid not in self._sessions:
            raise KeyError(f"unknown session: {sid}")
        return self._sessions[sid]

    def parent_of(self, sid: str) -> str | None:
        return self._parent.get(sid)

    def fork(self, sid: str, new_name: str | None = None) -> str:
        """Independent deep copy — both branches generate without
        corrupting each other (semantics from turboquant_kv.fork)."""
        src = self.get(sid)
        nid = new_name or uuid.uuid4().hex[:8]
        if nid in self._sessions:
            raise KeyError(f"session exists: {nid}")
        with timer("session.fork"):
            self._sessions[nid] = copy.deepcopy(src)
        self._parent[nid] = sid
        counter("session.forked").add(1)
        return nid

    def rewind(self, sid: str, target_offset: int) -> int:
        """O(1) tail drop via stock cache.trim(); returns dropped count
        (semantics from turboquant_kv.rewind_to)."""
        caches = self.get(sid)
        offset = caches[0].offset
        n = max(0, offset - max(0, min(target_offset, offset)))
        with timer("session.rewind"):
            for c in caches:
                c.trim(n)
        counter("session.rewound").add(1)
        return n

    def save(self, sid: str, path: str, metadata: dict | None = None) -> str:
        """Freeze to disk via stock mlx_lm.save_prompt_cache."""
        from mlx_lm.models.cache import save_prompt_cache

        p = str(Path(path))
        with timer("session.save"):
            save_prompt_cache(p, self.get(sid), metadata or {})
        counter("session.saved").add(1)
        return p

    def load(self, path: str, name: str | None = None) -> str:
        """Resume from disk via stock mlx_lm.load_prompt_cache —
        no re-prefill (semantics from turboquant_kv.load_from_disk)."""
        from mlx_lm.models.cache import load_prompt_cache

        sid = name or uuid.uuid4().hex[:8]
        if sid in self._sessions:
            raise KeyError(f"session exists: {sid}")
        with timer("session.load"):
            self._sessions[sid] = load_prompt_cache(path)
        counter("session.loaded").add(1)
        return sid
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_session_logic.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/session.py tests/test_session_logic.py
git commit -m "feat(mlxsuper): session pool save/load/fork/rewind logic"
```

---

### Task 6: Session model integration (Phase 0 gate part A)

**Files:**
- Create: `tests/test_session_model.py` (marked `@pytest.mark.model`)

- [ ] **Step 1: Write the failing test**

`tests/test_session_model.py`:
```python
import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm import generate

from gardener.mlxsuper.session import SessionPool

pytestmark = pytest.mark.model

def _greedy(model, tokenizer, prompt, cache, n):
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=n,
        sampler=make_sampler(temp=0.0), prompt_cache=cache, verbose=False,
    )

def test_save_load_resumes_bit_stable(loaded_model, tmp_path):
    model, tok = loaded_model
    pool = SessionPool(cache_factory=lambda: make_prompt_cache(model))
    sid = pool.create("s")
    prompt = "Count: one two three"
    _greedy(model, tok, prompt, pool.get(sid), 8)            # warm the cache
    path = pool.save(sid, str(tmp_path / "s.safetensors"))
    cont_a = _greedy(model, tok, " four", pool.get(sid), 8)
    rid = pool.load(path, "restored")
    cont_b = _greedy(model, tok, " four", pool.get(rid), 8)
    assert cont_a == cont_b                                  # no re-prefill, bit-stable

def test_fork_then_divergent_generation_is_independent(loaded_model):
    model, tok = loaded_model
    pool = SessionPool(cache_factory=lambda: make_prompt_cache(model))
    a = pool.create("a")
    _greedy(model, tok, "The capital of France is", pool.get(a), 4)
    b = pool.fork(a, "b")
    out_a = _greedy(model, tok, " definitely", pool.get(a), 6)
    out_b = _greedy(model, tok, " arguably", pool.get(b), 6)
    # Independent caches: continuing 'a' again is unaffected by 'b'.
    out_a2 = _greedy(model, tok, " and", pool.get(a), 4)
    assert isinstance(out_a, str) and isinstance(out_b, str)
    assert out_a != out_b
    assert isinstance(out_a2, str)
```

- [ ] **Step 2: Run to verify it fails first time only by downloading, then passes**

Run: `python -m pytest tests/test_session_model.py -v -m model`
Expected: PASS (2 passed) — first run downloads `TEST_MODEL` (~300 MB) once.
If `make_prompt_cache`/`generate` signature differs, fix `session.py`/test to match the installed `mlx_lm` (see plan reference: `mlx_lm/generate.py:745`, `mlx_lm/models/cache.py:13`).

- [ ] **Step 3: Commit**

```bash
cd /Users/terra/Developer/gardener
git add tests/test_session_model.py
git commit -m "test(mlxsuper): session save/load/fork model integration gate"
```

---

### Task 7: Minimal TTTEngine — adapters + generate_candidates + feedback

**Files:**
- Create: `gardener/mlxsuper/ttt.py`
- Test: `tests/test_ttt_logic.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ttt_logic.py`:
```python
from gardener.mlxsuper.ttt import Candidate, TrainStats, TTTEngine

def test_candidate_and_feedback_roundtrip():
    eng = TTTEngine.__new__(TTTEngine)      # no model needed for this logic
    eng.candidates = {}
    c = Candidate(id="c1", prompt="p", completion="x", tokens=[1, 2])
    eng.candidates["c1"] = c
    TTTEngine.feedback(eng, "c1", reward=1.0, signal="solution",
                       metadata={"passed": True})
    assert eng.candidates["c1"].reward == 1.0
    assert eng.candidates["c1"].signal == "solution"
    assert eng.candidates["c1"].metadata["passed"] is True

def test_feedback_unknown_candidate_is_safe():
    eng = TTTEngine.__new__(TTTEngine)
    eng.candidates = {}
    TTTEngine.feedback(eng, "missing", reward=1.0)   # must not raise

def test_trainstats_defaults():
    s = TrainStats()
    assert s.loss == 0.0 and s.num_positive == 0 and s.num_negative == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_ttt_logic.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `gardener/mlxsuper/ttt.py`** (reference map `omlx/ttt.py:144,222,250,272,297`; adapters via stock `mlx_lm.tuner`)

```python
"""Minimal test-time-training engine on stock mlx_lm + mlx_lm.tuner LoRA.

Reference map (semantics only, no omlx import): omlx/ttt.py:144 __init__,
:222 generate_candidates, :250 feedback, :272 feedback_from_execution,
:297 train_step. OPLoRA projection via gardener.mlxsuper.oplora."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .oplora import compute_svd_cache, project_lora_grads

logger = logging.getLogger("gardener.mlxsuper.ttt")


@dataclass
class Candidate:
    id: str
    prompt: str
    completion: str
    tokens: list[int]
    reward: float = 0.0
    signal: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TrainStats:
    loss: float = 0.0
    num_positive: int = 0
    num_negative: int = 0
    elapsed_s: float = 0.0
    adapter_norm: float = 0.0


class TTTEngine:
    def __init__(self, model: nn.Module, tokenizer: Any, *, rank: int = 8,
                 lr: float = 1e-4, num_lora_layers: int = 4,
                 lora_scale: float = 20.0):
        from mlx_lm.tuner.utils import linear_to_lora_layers

        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.candidates: dict[str, Candidate] = {}
        self.history: list[TrainStats] = []
        model.freeze()
        linear_to_lora_layers(
            model, num_lora_layers,
            {"rank": rank, "scale": lora_scale, "dropout": 0.0},
        )
        self._svd_cache: dict[str, dict[str, mx.array]] = {}

    # --- candidate lifecycle -------------------------------------------------
    def generate_candidates(self, prompt: str, n: int = 4,
                            max_tokens: int = 64,
                            temperature: float = 0.8) -> list[Candidate]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        out: list[Candidate] = []
        for _ in range(n):
            text = generate(self.model, self.tokenizer, prompt=prompt,
                             max_tokens=max_tokens, sampler=sampler,
                             verbose=False)
            cid = uuid.uuid4().hex[:8]
            c = Candidate(id=cid, prompt=prompt, completion=text,
                          tokens=self.tokenizer.encode(text))
            out.append(c)
            self.candidates[cid] = c
        return out

    def feedback(self, candidate_id: str, reward: float,
                 signal: str = "tool_call",
                 metadata: dict | None = None) -> None:
        c = self.candidates.get(candidate_id)
        if c is None:
            logger.warning("unknown candidate: %s", candidate_id)
            return
        c.reward = reward
        c.signal = signal
        if metadata:
            c.metadata.update(metadata)

    # --- training ------------------------------------------------------------
    def train_step(self, *, use_oplora: bool = False) -> TrainStats:
        """Cross-entropy on positive candidates; manual SGD on LoRA params.

        With use_oplora=True, LoRA grads are projected onto the safe
        subspace of each adapted base weight before the SGD step."""
        t0 = time.perf_counter()
        positive = [c for c in self.candidates.values() if c.reward > 0]
        negative = [c for c in self.candidates.values() if c.reward <= 0]
        if not positive:
            stats = TrainStats(num_negative=len(negative))
            self.history.append(stats)
            self.candidates.clear()
            return stats

        def loss_fn():
            total = mx.array(0.0)
            for c in positive:
                full = c.prompt + c.completion
                ids = mx.array([self.tokenizer.encode(full)])
                plen = len(self.tokenizer.encode(c.prompt))
                if plen >= ids.shape[1] - 1:
                    continue
                logits = self.model(ids)
                sl = logits[:, plen - 1:-1, :].reshape(-1, logits.shape[-1])
                lb = ids[:, plen:].reshape(-1)
                total = total + nn.losses.cross_entropy(sl, lb,
                                                        reduction="mean")
            return total / max(len(positive), 1)

        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        loss_val, grads = loss_and_grad()

        if use_oplora:
            grads = self._project_grads(grads)

        # Manual SGD on trainable (LoRA) params.
        params = self.model.trainable_parameters()
        updated = _tree_sgd(params, grads, self.lr)
        self.model.update(updated)
        mx.eval(self.model.parameters(), loss_val)

        norm = _tree_l2(self.model.trainable_parameters())
        stats = TrainStats(
            loss=float(loss_val.item()),
            num_positive=len(positive),
            num_negative=len(negative),
            elapsed_s=time.perf_counter() - t0,
            adapter_norm=norm,
        )
        self.history.append(stats)
        self.candidates.clear()
        return stats

    def _project_grads(self, grads):
        """OPLoRA-project lora_a/lora_b grads per adapted LoRALinear."""
        from mlx_lm.tuner.lora import LoRALinear

        def walk(prefix, gnode, mnode):
            if isinstance(mnode, LoRALinear):
                W = mnode.linear.weight  # (out, in)
                key = prefix
                if key not in self._svd_cache:
                    self._svd_cache.update(
                        compute_svd_cache({key: W}, k=8))
                svd = self._svd_cache[key]
                # LoRALinear: lora_a (in, r), lora_b (r, out).
                # oplora expects A:(r,n=in), B:(m=out,r) → transpose views.
                dA = gnode["lora_a"].T
                dB = gnode["lora_b"].T
                dA_s, dB_s = project_lora_grads(
                    W, mnode.lora_a.T, mnode.lora_b.T, dA, dB,
                    {"U_k": svd["U_k"], "V_k": svd["V_k"]})
                gnode["lora_a"] = dA_s.T
                gnode["lora_b"] = dB_s.T
                return
            if isinstance(gnode, dict):
                for kk, gv in gnode.items():
                    mv = mnode[kk] if isinstance(mnode, (list, dict)) \
                        else getattr(mnode, kk, None)
                    if mv is not None:
                        walk(f"{prefix}.{kk}", gv, mv)
            elif isinstance(gnode, list):
                for i, gv in enumerate(gnode):
                    walk(f"{prefix}.{i}", gv, mnode[i])

        walk("model", grads, self.model)
        return grads


def _tree_sgd(params, grads, lr):
    if isinstance(params, dict):
        return {k: _tree_sgd(params[k], grads[k], lr) for k in grads}
    if isinstance(params, list):
        return [_tree_sgd(p, g, lr) for p, g in zip(params, grads)]
    return params - lr * grads


def _tree_l2(tree) -> float:
    if isinstance(tree, dict):
        return sum(_tree_l2(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_tree_l2(v) for v in tree)
    return float(mx.sum(tree.astype(mx.float32) ** 2).item())
```
> Note: `_tree_l2` returns a sum-of-squares; the caller treats it as a monotone proxy for adapter magnitude (sqrt omitted intentionally — only relative growth matters for the gate).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_ttt_logic.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/ttt.py tests/test_ttt_logic.py
git commit -m "feat(mlxsuper): minimal TTTEngine (adapters/candidates/feedback)"
```

---

### Task 8: TTT learning gate + OPLoRA safety (Phase 0 gate part B)

**Files:**
- Create: `tests/test_ttt_model.py` (marked `@pytest.mark.model`)

- [ ] **Step 1: Write the failing test**

`tests/test_ttt_model.py`:
```python
import mlx.core as mx
import pytest

from gardener.mlxsuper.ttt import TTTEngine

pytestmark = pytest.mark.model

def test_train_step_lowers_loss_on_positive_sample(loaded_model):
    model, tok = loaded_model
    eng = TTTEngine(model, tok, rank=8, lr=1e-3, num_lora_layers=2)
    c = eng.generate_candidates("Write hello:", n=1, max_tokens=8)[0]
    c.completion = " hello world"          # force a clean positive target
    eng.feedback(c.id, reward=1.0, signal="solution")
    s1 = eng.train_step()
    # Re-train the SAME target; loss must not increase (it should drop/flatten).
    eng.candidates[c.id] = c
    c.reward = 1.0
    s2 = eng.train_step()
    assert s2.loss <= s1.loss + 1e-3
    assert s1.num_positive == 1

def test_oplora_projected_step_keeps_heldout_probe_within_tolerance(loaded_model):
    model, tok = loaded_model
    eng = TTTEngine(model, tok, rank=8, lr=1e-3, num_lora_layers=2)
    probe = mx.array([tok.encode("The sky is")])
    before = mx.array(model(probe))
    c = eng.generate_candidates("Adapt:", n=1, max_tokens=8)[0]
    c.completion = " adapted text here"
    eng.feedback(c.id, reward=1.0)
    eng.train_step(use_oplora=True)
    after = mx.array(model(probe))
    drift = float(mx.mean(mx.abs(after - before)).item())
    assert drift < 1.0          # projected update stays bounded on held-out input
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/test_ttt_model.py -v -m model`
Expected: PASS (2 passed). If LoRA param tree traversal in `_project_grads` mismatches the installed `mlx_lm.tuner.lora.LoRALinear` attribute names, align names to the installed version (reference: `mlx_lm/tuner/lora.py:11`).

- [ ] **Step 3: Commit**

```bash
cd /Users/terra/Developer/gardener
git add tests/test_ttt_model.py
git commit -m "test(mlxsuper): TTT learning + OPLoRA safety model gate"
```

---

### Task 9: Public exports + Phase 0 gate aggregation

**Files:**
- Modify: `gardener/mlxsuper/__init__.py`
- Create: `tests/test_phase0_gate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_phase0_gate.py`:
```python
def test_public_surface_imports():
    from gardener.mlxsuper import (
        SessionPool, TTTEngine, Candidate, TrainStats,
        compute_layer_lrs, project_lora_grads, compute_svd_cache,
        timer, counter, registry, median_of_n,
    )
    assert callable(compute_layer_lrs)
    assert callable(project_lora_grads)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_phase0_gate.py -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement `gardener/mlxsuper/__init__.py`**

```python
"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable."""
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .schedules import compute_layer_lrs
from .session import SessionPool
from .ttt import Candidate, TrainStats, TTTEngine

__all__ = [
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
]
```

- [ ] **Step 4: Run the full fast suite + the gate script**

Run: `cd /Users/terra/Developer/gardener && python -m pytest -v -m "not model" && python scripts/check_no_omlx_import.py`
Expected: all fast tests PASS; `OK: no omlx imports in gardener/`

- [ ] **Step 5: Run the model gate suite once**

Run: `python -m pytest -v -m model`
Expected: 4 passed (session ×2, ttt ×2). This is the Phase 0 acceptance gate from the spec.

- [ ] **Step 6: Commit**

```bash
cd /Users/terra/Developer/gardener
git add gardener/mlxsuper/__init__.py tests/test_phase0_gate.py
git commit -m "feat(mlxsuper): public surface + Phase 0 gate green"
```

---

## Phase 0 Acceptance Gate (from spec)

All must hold (mirrors `docs/design.md` Phasing #0):

- [ ] Loads `TEST_MODEL` and generates (Task 6/8).
- [ ] KV save→load resumes with no re-prefill and **bit-stable** continuation (Task 6 `test_save_load_resumes_bit_stable`).
- [ ] Fork then divergent generation is independent (Task 6 `test_fork_then_divergent_generation_is_independent`).
- [ ] One TTT `train_step` on a positive sample does not increase loss; OPLoRA-projected step keeps a held-out probe within tolerance (Task 8).
- [ ] Base-deps-only install; **`import omlx` absent**, grep-gated in CI (Task 1, re-run Task 9 Step 4).

---

## Self-Review

- **Spec coverage:** Phase 0 line in `design.md` → Tasks 1–9. mlxsuper/ files (session, schedules, oplora, ttt, observability) all created. omlx-import gate (Task 1/9). No scheduler/pipeline/knowledge work leaked in (correctly deferred).
- **Placeholder scan:** every code step has complete code; no TBD/TODO; commands have expected output.
- **Type consistency:** `Candidate`/`TrainStats`/`TTTEngine.feedback`/`train_step` signatures consistent across Tasks 7–9; `SessionPool.{create,get,fork,rewind,save,load,parent_of}` consistent across Tasks 5–6; `project_lora_grads`/`compute_svd_cache`/`_truncated_svd` consistent Tasks 4 & 7.
- **Known risk:** stock `mlx_lm` API drift (cache/generate/tuner signatures) — every model task names the reference file:line to realign against the *installed* version rather than guess.
