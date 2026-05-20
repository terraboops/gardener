# Gardener MVP+ — Deferred Items Plan

> Execute via subagent-driven-development. Closes the deferred items from the
> v0 prototype with the smallest implementations that still earn the feature
> in the architecture.

**Goal:** finish the substantive deferred items so the README's right-hand
column shrinks to "Pi RPC adapter" and "Chalet git-projection UI" only.

**Architecture:** every task is additive to the v0 surface — no breaking
changes to existing public APIs.

## Tasks

### Task P1 — Knowledge hardening (K1, K4, K5, K8, K9)

**Files:**
- Modify: `gardener/knowledge.py`
- Test: extend `tests/test_knowledge.py`

**Defects to fix:**

- **K1 unbounded growth** — `KnowledgeStore(..., max_entries=N)` (default 1000). On `write()`, if entries > N, evict oldest by `updated_at`.
- **K4 scattered validation** — `validate_object()` already runs in `__post_init__`; expose `get_strict(oid)` that *raises* on a malformed file instead of silently returning corrupted state. Keep `all()`'s permissive skip behavior for iteration.
- **K5 atomic-write read race** — use `tempfile.mkstemp` + `os.replace` for atomic writes (replaces direct `write_text`).
- **K8 concurrent-write guard** — per-store `threading.Lock` around `write/mark_tested`. (Cross-process locking deferred — single-daemon assumption.)
- **K9 idea_context validation** — `_validate_idea_context(list)` ensures all elements are non-empty strs; order-stable on round-trip (already true via list, but pin it with a test).

**Add tests:**
```python
def test_K1_cap_evicts_oldest_by_updated_at(tmp_path): ...
def test_K4_get_strict_raises_on_malformed_file(tmp_path): ...
def test_K5_atomic_write_no_partial_reads(tmp_path): ...
def test_K8_concurrent_writers_no_lost_update(tmp_path): ...
def test_K9_idea_context_validated(tmp_path): ...
```

Commit: `feat(knowledge): K1/K4/K5/K8/K9 hardening`.

---

### Task P2 — Knowledge merge + curation (K3 + K6 + K7)

**Files:**
- Modify: `gardener/knowledge.py` (add `merge()` and `apply_curation()`)
- Test: extend `tests/test_knowledge.py`

**Defects:**

- **K3 hash collision** — current 16-char hash is wider than trellis's 8 but still finite. Add explicit collision detection on `write()`: if `_path(oid)` exists and the file content has *different* predicates, raise.
- **K6 LLM action validation** — `apply_curation(actions: list[dict])` accepts a list of `{action: "keep"|"merge"|"drop", id: str, ids?: list, predicates?: list, …}` items, validates each strictly (unknown action / missing id → raise; never silent skip), and applies.
- **K7 merge recomputes id** — `merge(ids: list, into_predicates: list)` deletes the source objects, computes a fresh id from `into_predicates` via `semantic_hash`, writes a new object with combined `idea_context`, summed `helped`, max `confidence`.

**Add tests:**
```python
def test_K3_collision_with_different_content_raises(tmp_path): ...
def test_K6_invalid_action_raises_loud(tmp_path): ...
def test_K7_merge_recomputes_id_and_combines_provenance(tmp_path): ...
```

Commit: `feat(knowledge): K3/K6/K7 merge + curation`.

---

### Task P3 — TTT sleep cycle + bit-equivalence promote gate (Phase 5)

**Files:**
- Create: `gardener/sleep.py`
- Test: `tests/test_sleep.py` (mixed: pure-logic + `@pytest.mark.model`)

**Surface:**

```python
class SleepCycle:
    def __init__(self, engine, store, *, probe_prompts: list[str],
                 max_drift: float = 0.5):
        ...
    def run(self) -> SleepResult:
        """Pull consolidatable knowledge → build candidates → train →
        promote if held-out probe drift < max_drift else rewind."""
```

`SleepResult` = `{state: "promoted"|"rewound"|"no_op", trained: int,
drift: float, before_loss: float, after_loss: float}`.

Algorithm:
1. `consolidatable = list(store.consolidatable())` — empty → return `no_op`.
2. Snapshot probe logits (mean over probe_prompts) `before`.
3. For each item: build a `Candidate(prompt=<seed>, completion=item.insight,
   tokens=tokenize(insight))`; feed `engine.candidates[cid]=c; c.reward=+1`.
4. `engine.save_checkpoint()` (add this method if missing — adapter snapshot).
5. `stats = engine.train_step(use_oplora=True)`.
6. Snapshot probe logits `after`. `drift = mean(abs(after - before))` (fp32).
7. If `drift > max_drift` → `engine.rewind()` (add if missing) → `rewound`.
   Else → mark each knowledge object `helped+=0` (already tested) → `promoted`.

**Add to `TTTEngine`** (`gardener/mlxsuper/ttt.py`): `save_checkpoint()` and
`rewind()` that snapshot/restore the model's trainable parameter state.

**Tests:**
- Pure: `SleepCycle.run()` on a `FakeEngine` + `FakeStore` returns `no_op`
  when nothing consolidatable; `promoted` with mocked drift below threshold;
  `rewound` with drift above threshold.
- `@pytest.mark.model`: real `TTTEngine` + small knowledge → run cycle → some
  terminal state reached without crashing.

Commit: `feat(sleep): TTT sleep cycle with bit-equivalence promote gate`.

---

### Task P4 — Prose DSL + `compose_pipeline` capability (Phase 3 deep)

**Files:**
- Create: `gardener/pipeline/prose_parser.py`
- Modify: `gardener/pipeline/yaml_loader.py` (re-export `compose_pipeline`)
- Modify: `gardener/pipeline/__init__.py` (export `compose_pipeline`)
- Test: `tests/test_prose_parser.py`

**Prose DSL** — new lark grammar expressive enough for the full DAG IR.
Reuses lark + indentation-token technique (not trellis's tiny grammar).
Supported constructs:

```
pipeline NAME:
  deadline NUMBER

  agent ID:                    # an agent node
    prompt: "..."              # → params.system_prompt
    max_tokens: N              # → params.max_tokens
    retry: N                   # → retry.max
    input INPUT_NAME from ID   # an edge (can repeat)

  human ID:
    ask: "..."
    timeout: NUMBER
    on_timeout: dead-letter|escalate|fallback-agent
    input INPUT_NAME from ID

  composite ID:
    retry: N
    body:                      # nested pipeline
      … (recursive)
```

**`compose_pipeline(spec: str) -> Pipeline`** — single entry-point an agent
calls; auto-detects YAML vs prose (heuristic: starts with `pipeline ` →
prose; starts with `name:` or `{` → YAML); returns validated `Pipeline`.

**Tests:**
- Round-trip: a known YAML pipeline can be expressed in prose, both load to
  IRs with identical `name`/node ids/edges/order.
- A diamond DAG in prose validates and topo-orders correctly.
- A `composite` with a nested body parses.
- `compose_pipeline("pipeline …")` and `compose_pipeline("name: …")` both
  work; corrupt input raises `ValueError` (no silent `{}`).

Commit: `feat(pipeline): prose DSL + compose_pipeline capability`.

---

### Task P5 — Scheduler-lite: priority queue + cadence triggers (Phase 2)

**Files:**
- Create: `gardener/scheduler.py`
- Modify: `gardener/daemon.py` (add `submit_async`, `start`/`stop` methods)
- Test: `tests/test_scheduler.py`

**Surface:**

```python
@dataclass(order=True)
class Job:
    priority: float      # higher runs first
    enqueued_at: float
    pipeline: Pipeline = field(compare=False)
    initial_inputs: dict | None = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)

class Scheduler:
    def __init__(self, executor: PipelineExecutor, max_concurrent: int = 2,
                 starvation_boost_after_s: float = 60.0):
        ...
    def submit(self, pipeline: Pipeline, priority: float = 5.0,
               initial_inputs: dict | None = None) -> Future:
        ...
    def add_cadence(self, name: str, pipeline_factory: Callable,
                    cron: str) -> None:    # simple "every Ns" string, not full cron
        ...
    def start(self) -> None: ...   # spawns worker threads
    def stop(self) -> None: ...
```

Inverted priority (heapq min-heap → negate). Starvation prevention: a thread
periodically re-prioritizes long-pending jobs (priority += boost where age >
threshold). Concurrency: bounded threadpool of `max_concurrent`. Cadence:
a separate ticker thread re-submits pipelines on a `every Ns` string spec
(`"every 30s"`, `"every 5m"`).

**Tests:**
- Two jobs, second has higher priority → runs first.
- Starvation boost: enqueue low-priority, wait, then high-priority — low one
  eventually gets boosted (use short threshold like 0.05s in test).
- `add_cadence("every 0.1s", …)` fires ≥2 times in 0.3s.
- `stop()` terminates workers cleanly.

Commit: `feat(scheduler): priority queue + starvation boost + cadence`.

---

### Task P6 — Block-pool pre-cached subagents (Phase 6)

**Files:**
- Create: `gardener/subagents.py`
- Modify: `gardener/harness/mlx_agent.py` (accept optional pre-warmed session)
- Test: `tests/test_subagents.py` (`@pytest.mark.model`)

**Surface:**

```python
@dataclass
class AgentProfile:
    name: str
    system_prompt: str
    params: dict        # max_tokens, temperature, …
    warmup_text: str    # text to prefill into the session

class SubagentRegistry:
    def __init__(self, model, tokenizer, root: Path):
        ...
    def register(self, profile: AgentProfile) -> None:
        """Prefill warmup_text into a fresh session, save to root/<name>.cache."""
    def runner(self, profile_name: str) -> AgentRunner:
        """Return an AgentRunner that loads the pre-warmed session per call,
        forks it (independent branch), runs the agent, then discards the fork."""
```

**Tests** (`@pytest.mark.model`):
- `register("answerer", warmup_text="You answer questions.")` creates a
  cache file.
- A `runner` produced for that profile, when given a question, returns a
  non-empty string.
- The registry persists across `Registry` re-construction (cache file is
  the durable artifact).

Commit: `feat(subagents): block-pool of pre-cached agent profiles`.

---

### Task P7 — Extended demo: prose + scheduler + subagent + sleep

**Files:**
- Create: `examples/demo_full.py`
- Create: `tests/test_demo_full_smoke.py` (`@pytest.mark.model`)

The demo:

1. Register a pre-warmed "answerer" subagent profile.
2. Compose a pipeline *in prose*: seed → answerer → review.
3. Submit via the `Scheduler` (priority=5.0).
4. After it succeeds, write a `KnowledgeObject` and `mark_tested`.
5. Run a `SleepCycle` over the knowledge → expect a terminal state.
6. Print: prose source, compiled IR summary, schedule result, journal, sleep
   result.

Commit: `feat(demo): extended demo — prose + scheduler + subagent + sleep`.

---

### Task P8 — Update README "Status" table; doc Pi/Chalet seams

**Files:**
- Modify: `README.md`
- Modify: `gardener/docs/design.md` (add "Why Pi RPC + Chalet UI are deferred"
  rationale, brief)

- Move K1/K3-K9, prose DSL + `compose_pipeline`, TTT sleep, scheduler, and
  block-pool from the "Deferred" column to the "v0" column.
- Doc the two remaining seams.

Commit: `docs: MVP+ status — close 5 deferred items`.

---

## Acceptance gate

- [ ] All pre-existing tests still pass (39 fast + 6 model).
- [ ] Each Task P1–P7 adds its own tests, all passing.
- [ ] `python -m examples.demo_full` reaches `state: succeeded` for the
  pipeline and a non-`no_op` (`promoted` or `rewound`) state for the sleep
  cycle.
- [ ] `scripts/check_no_omlx_import.py` still passes.
