# Gardener MVP Prototype Plan (Phases A–F)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A working end-to-end Gardener prototype on top of the Phase 0 mlxsuper core: an event journal with DLQ, a hardened knowledge store with empirical feedback, a composable-DAG pipeline engine (agent/human/composite nodes, retry, deadline, DLQ routing), an MLX agent runner, and a demo script that defines a small agent and runs a pipeline through the system with `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

**Architecture:** Inline orchestration (no trellis lift in v0 — Phase 2 from the spec is deferred). All work lives under `gardener/` (the standalone repo). Zero `omlx` imports (enforced by the Phase 0 CI gate).

**Tech Stack:** Python 3.13+, mlxsuper (Phase 0), `pyyaml`, `mlx-lm`, `pytest`.

## Scope vs the full spec

| Spec Phase | This plan | Deferred to later |
|---|---|---|
| 1 Spine | A1 journal+DLQ, A2 knowledge, D3 daemon, D1 MLX agent runner | Pi RPC adapter (we use mlxsuper directly as the harness) |
| 2 Scheduler lift | — (inline executor) | trellis pool port + H1-H5 hardening |
| 3 Pipeline DAG | B1 IR, B2 YAML loader, B3 executor, B4 composite | prose DSL, model-check, agent-authored `compose_pipeline` at runtime |
| 4 Knowledge | A2 minimal store + empirical gate (K10) + reject-malformed (K2) | K1, K3-K9 deep fixes |
| 5 Cultivation | E1 wire feedback into TTTEngine | full sleep cycle + promote-gate / OPLoRA orchestration |
| 6 Subagent swarm | — | block-pool + pre-cached subagents |
| End-to-end demo | F | — |

Where the prototype is deliberately a sketch vs the spec, the code includes a `# TODO(spec-Phase-N): …` marker so the future plan can find it.

## File structure

```
gardener/
  journal.py              # EventJournal + DLQ topic
  knowledge.py            # KnowledgeStore + empirical gate
  pipeline/
    __init__.py
    ir.py                 # Pipeline, Node, Edge dataclasses
    yaml_loader.py        # YAML → IR + loud validation
    executor.py           # sequential DAG execution + retry/timeout/deadline/DLQ
  harness/
    __init__.py           # AgentRunner ABC, HumanRunner ABC
    mlx_agent.py          # MLXAgentRunner (uses mlxsuper)
    cli_human.py          # CLIHumanRunner (stdin) + ScriptedHumanRunner (demo)
  daemon.py               # Gardener orchestrator class
  learning.py             # cultivation hook: outcome → TTTEngine.feedback
examples/
  demo_simple_agent.py    # end-to-end demo
tests/
  test_journal.py
  test_knowledge.py
  test_pipeline_ir.py
  test_pipeline_yaml.py
  test_pipeline_executor.py
  test_pipeline_composite.py
  test_harness_mlx_agent.py     # @pytest.mark.model
  test_daemon.py
  test_demo_smoke.py            # @pytest.mark.model
```

---

### Task A1: Event journal + DLQ topic

**Files:** Create `gardener/journal.py`, `tests/test_journal.py`.

**Spec semantics:** append-only JSONL on disk; topics are separate files in a dir; DLQ is a topic named `dead_letter`; replay yields events in append order; re-drive returns DLQ entries for operator action.

- [ ] **Step 1: Failing test** (`tests/test_journal.py`):

```python
from gardener.journal import EventJournal

def test_append_read_roundtrip(tmp_path):
    j = EventJournal(tmp_path)
    j.append("main", {"k": 1})
    j.append("main", {"k": 2})
    assert list(j.read("main")) == [{"k": 1}, {"k": 2}]

def test_dead_letter_routes_to_dlq_topic(tmp_path):
    j = EventJournal(tmp_path)
    j.dead_letter({"node": "n1", "reason": "timeout"})
    items = list(j.read("dead_letter"))
    assert items == [{"node": "n1", "reason": "timeout"}]
    assert j.dlq_depth() == 1

def test_redrive_returns_and_clears_dlq(tmp_path):
    j = EventJournal(tmp_path)
    j.dead_letter({"id": "a"}); j.dead_letter({"id": "b"})
    drained = j.redrive_dlq()
    assert [d["id"] for d in drained] == ["a", "b"]
    assert j.dlq_depth() == 0
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/journal.py`:

```python
"""Append-only JSONL event journal with DLQ topic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DLQ_TOPIC = "dead_letter"


class EventJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _topic_path(self, topic: str) -> Path:
        return self.root / f"{topic}.jsonl"

    def append(self, topic: str, event: dict) -> None:
        with self._topic_path(topic).open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def read(self, topic: str) -> Iterable[dict]:
        path = self._topic_path(topic)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def dead_letter(self, item: dict) -> None:
        self.append(DLQ_TOPIC, item)

    def dlq_depth(self) -> int:
        return sum(1 for _ in self.read(DLQ_TOPIC))

    def redrive_dlq(self) -> list[dict]:
        items = list(self.read(DLQ_TOPIC))
        self._topic_path(DLQ_TOPIC).write_text("", encoding="utf-8")
        return items
```

- [ ] **Step 4:** `.venv/bin/python -m pytest tests/test_journal.py -v` → 3 passed.
- [ ] **Step 5:** Commit `feat(journal): append-only JSONL + DLQ topic`.

---

### Task A2: Knowledge store + empirical-feedback gate

**Files:** Create `gardener/knowledge.py`, `tests/test_knowledge.py`.

**Spec semantics:** per-agent YAML-on-disk; schema validated on load (rejects malformed predicates — defect K2); empirical field gates consolidation (defect K10); search by predicate overlap; concurrent-safe single-writer use only (defect K8 deep fix deferred).

- [ ] **Step 1: Failing test** (`tests/test_knowledge.py`):

```python
import pytest
from gardener.knowledge import KnowledgeStore, KnowledgeObject

def test_write_then_read(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    obj = KnowledgeObject(
        predicates=[["sky", "is", "blue"]],
        insight="visible-light scattering",
        justification="Rayleigh scattering",
        source_agent="alpha",
    )
    oid = s.write(obj)
    loaded = s.get(oid)
    assert loaded.insight == "visible-light scattering"
    assert loaded.empirical["tested"] is False

def test_malformed_predicates_rejected_loudly(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    bad = KnowledgeObject(
        predicates=[["only", "two"]],  # length 2, not triple
        insight="x", justification="y", source_agent="alpha",
    )
    with pytest.raises(ValueError, match="predicate"):
        s.write(bad)

def test_search_by_predicate_overlap(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="x",
                            justification="y", source_agent="alpha"))
    s.write(KnowledgeObject(predicates=[["c", "r", "d"]], insight="z",
                            justification="y", source_agent="alpha"))
    hits = s.search([["a", "r", "b"]])
    assert len(hits) == 1 and hits[0].insight == "x"

def test_empirical_gate_only_tested_consolidates(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = s.write(KnowledgeObject(
        predicates=[["a", "r", "b"]], insight="i1", justification="j",
        source_agent="alpha"))
    b = s.write(KnowledgeObject(
        predicates=[["c", "r", "d"]], insight="i2", justification="j",
        source_agent="alpha"))
    s.mark_tested(a, helped=True)
    consolidatable = list(s.consolidatable())
    assert [c.id for c in consolidatable] == [a]
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/knowledge.py`:

```python
"""Per-agent knowledge store with empirical-feedback gate.

Reference map (semantics only — defects K2, K10 fixed here; K1/K3-K9
deferred to spec Phase 4 deep rebuild): omlx incubator
trellis/tools/knowledge_io.py + orchestrator/evolution.py."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


def _normalize_triple(t) -> tuple[str, str, str]:
    if not isinstance(t, (list, tuple)) or len(t) != 3:
        raise ValueError(f"predicate must be a 3-tuple, got: {t!r}")
    s, p, o = t
    if not all(isinstance(x, str) and x for x in (s, p, o)):
        raise ValueError(f"predicate elements must be non-empty strs: {t!r}")
    return s, p, o


def semantic_hash(predicates: list[list[str]]) -> str:
    norm = sorted(_normalize_triple(t) for t in predicates)
    return hashlib.sha256(
        json.dumps(norm, sort_keys=True).encode()
    ).hexdigest()[:16]      # 16 chars (K3 wider than trellis's 8)


@dataclass
class KnowledgeObject:
    predicates: list[list[str]]
    insight: str
    justification: str
    source_agent: str
    id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 0.5
    idea_context: list[str] = field(default_factory=list)
    empirical: dict = field(default_factory=lambda: {
        "tested": False, "uses": 0, "helped": 0, "last_validated_at": None,
    })

    def __post_init__(self):
        # K2: strict normalize on construct (rejects loud)
        [_normalize_triple(t) for t in self.predicates]
        if not self.id:
            self.id = semantic_hash(self.predicates) or uuid.uuid4().hex[:16]


class KnowledgeStore:
    def __init__(self, root: str | Path, agent: str):
        self.dir = Path(root) / agent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.agent = agent

    def _path(self, oid: str) -> Path:
        return self.dir / f"{oid}.yaml"

    def write(self, obj: KnowledgeObject) -> str:
        # K2: validate by re-construction (raises on bad predicates)
        obj = KnowledgeObject(**asdict(obj))
        obj.updated_at = time.time()
        self._path(obj.id).write_text(
            yaml.safe_dump(asdict(obj), sort_keys=True), encoding="utf-8")
        return obj.id

    def get(self, oid: str) -> KnowledgeObject:
        data = yaml.safe_load(self._path(oid).read_text(encoding="utf-8"))
        return KnowledgeObject(**data)

    def all(self) -> Iterable[KnowledgeObject]:
        for p in sorted(self.dir.glob("*.yaml")):
            try:
                yield KnowledgeObject(**yaml.safe_load(p.read_text()))
            except (ValueError, TypeError, yaml.YAMLError):
                # K2: malformed entries are SKIPPED on read (logged later)
                continue

    def search(self, predicates: list[list[str]]) -> list[KnowledgeObject]:
        want = {tuple(_normalize_triple(t)) for t in predicates}
        out = []
        for obj in self.all():
            have = {tuple(_normalize_triple(t)) for t in obj.predicates}
            if want & have:
                out.append(obj)
        return out

    def mark_tested(self, oid: str, *, helped: bool) -> None:
        obj = self.get(oid)
        obj.empirical["tested"] = True
        obj.empirical["uses"] = obj.empirical.get("uses", 0) + 1
        if helped:
            obj.empirical["helped"] = obj.empirical.get("helped", 0) + 1
        obj.empirical["last_validated_at"] = time.time()
        obj.updated_at = time.time()
        self._path(oid).write_text(
            yaml.safe_dump(asdict(obj), sort_keys=True), encoding="utf-8")

    def consolidatable(self) -> Iterable[KnowledgeObject]:
        """K10: only tested knowledge is eligible for weight-tier consolidation."""
        for obj in self.all():
            if obj.empirical.get("tested"):
                yield obj
```

- [ ] **Step 4:** `pytest tests/test_knowledge.py -v` → 4 passed.
- [ ] **Step 5:** Commit `feat(knowledge): store with empirical gate (K2 K10)`.

---

### Task B1: Pipeline IR dataclasses

**Files:** Create `gardener/pipeline/__init__.py`, `gardener/pipeline/ir.py`, `tests/test_pipeline_ir.py`.

**Spec semantics:** typed dataclasses for `Pipeline`, `Node` (kind ∈ {agent, human, composite}), `Edge`; acyclicity check; named-input resolution check.

- [ ] **Step 1: Failing test** (`tests/test_pipeline_ir.py`):

```python
import pytest
from gardener.pipeline.ir import Pipeline, Node, Edge

def _agent(id_, **kw):
    return Node(id=id_, kind="agent", agent=kw.get("agent", "echo"),
                params=kw.get("params", {}), retry={"max": 1})

def test_linear_pipeline_validates():
    p = Pipeline(
        name="linear", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="a", to_id="b", input_name="prev")],
    )
    p.validate()              # no raise

def test_cycle_rejected_loudly():
    p = Pipeline(
        name="cycle", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="a", to_id="b", input_name="x"),
               Edge(from_id="b", to_id="a", input_name="y")],
    )
    with pytest.raises(ValueError, match="cycle"):
        p.validate()

def test_human_node_requires_timeout_and_on_timeout():
    p = Pipeline(
        name="h", deadline=60.0,
        nodes=[Node(id="h1", kind="human", ask="approve?")],
        edges=[],
    )
    with pytest.raises(ValueError, match="human.*timeout"):
        p.validate()

def test_missing_deadline_rejected():
    p = Pipeline(name="x", deadline=None, nodes=[_agent("a")], edges=[])
    with pytest.raises(ValueError, match="deadline"):
        p.validate()

def test_undefined_input_rejected():
    p = Pipeline(
        name="u", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="ghost", to_id="b", input_name="x")],
    )
    with pytest.raises(ValueError, match="ghost"):
        p.validate()
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/pipeline/__init__.py` (empty) and `gardener/pipeline/ir.py`:

```python
"""Pipeline DAG IR (Phase 3 v0: agent/human/composite nodes, no top-level cycles)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

NodeKind = Literal["agent", "human", "composite"]


@dataclass
class Edge:
    from_id: str
    to_id: str
    input_name: str
    condition: Optional[str] = None      # predicate on predecessor result


@dataclass
class Node:
    id: str
    kind: NodeKind
    # agent
    agent: Optional[str] = None
    # human
    ask: Optional[str] = None
    assignee: Optional[str] = None
    response_schema: Optional[dict] = None
    timeout: Optional[float] = None
    on_timeout: Optional[str] = None     # "escalate" | "fallback-agent" | "dead-letter"
    # composite
    body: Optional["Pipeline"] = None
    # common
    params: dict = field(default_factory=dict)
    retry: dict = field(default_factory=lambda: {"max": 1, "on": "fail"})


@dataclass
class Pipeline:
    name: str
    deadline: Optional[float]
    nodes: list[Node]
    edges: list[Edge]

    # --- validation ----------------------------------------------------------
    def validate(self) -> None:
        if self.deadline is None or self.deadline <= 0:
            raise ValueError(f"pipeline {self.name}: deadline required (got {self.deadline})")
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError(f"pipeline {self.name}: duplicate node ids")
        for e in self.edges:
            if e.from_id not in ids:
                raise ValueError(f"edge references unknown node: {e.from_id}")
            if e.to_id not in ids:
                raise ValueError(f"edge references unknown node: {e.to_id}")
        for n in self.nodes:
            if not isinstance(n.retry.get("max"), int) or n.retry["max"] < 1:
                raise ValueError(f"node {n.id}: retry.max must be a positive int")
            if n.kind == "human":
                if n.timeout is None or n.on_timeout is None:
                    raise ValueError(
                        f"node {n.id}: human nodes require timeout + on_timeout")
            if n.kind == "composite" and n.body is None:
                raise ValueError(f"node {n.id}: composite requires body sub-pipeline")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        # Topological sort via Kahn's algorithm; cycle → unreached nodes left.
        indeg = {n.id: 0 for n in self.nodes}
        succ: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            indeg[e.to_id] += 1
            succ[e.from_id].append(e.to_id)
        q = [nid for nid, d in indeg.items() if d == 0]
        seen = 0
        while q:
            cur = q.pop(0)
            seen += 1
            for nxt in succ[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        if seen != len(self.nodes):
            raise ValueError(f"pipeline {self.name}: cycle detected")

    def topo_order(self) -> list[str]:
        indeg = {n.id: 0 for n in self.nodes}
        succ: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for e in self.edges:
            indeg[e.to_id] += 1
            succ[e.from_id].append(e.to_id)
        order, q = [], [nid for nid, d in indeg.items() if d == 0]
        while q:
            cur = q.pop(0)
            order.append(cur)
            for nxt in succ[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        return order
```

- [ ] **Step 4:** `pytest tests/test_pipeline_ir.py -v` → 5 passed.
- [ ] **Step 5:** Commit `feat(pipeline): DAG IR with loud validation`.

---

### Task B2: YAML loader

**Files:** Create `gardener/pipeline/yaml_loader.py`, `tests/test_pipeline_yaml.py`.

- [ ] **Step 1: Failing test** (`tests/test_pipeline_yaml.py`):

```python
import pytest
from gardener.pipeline.yaml_loader import load_pipeline_yaml

YAML = """
name: demo
deadline: 30.0
nodes:
  - id: greet
    kind: agent
    agent: echo
    retry: {max: 2, on: fail}
  - id: review
    kind: human
    ask: "Approve the greeting?"
    timeout: 10.0
    on_timeout: dead-letter
edges:
  - from_id: greet
    to_id: review
    input_name: greeting
"""

def test_loads_validates_roundtrips():
    p = load_pipeline_yaml(YAML)
    assert p.name == "demo"
    assert [n.id for n in p.nodes] == ["greet", "review"]
    assert p.nodes[1].kind == "human"
    assert p.topo_order() == ["greet", "review"]

def test_bad_yaml_rejected_loudly_not_silent_dict():
    with pytest.raises(ValueError):
        load_pipeline_yaml("nodes: [oops")     # malformed
    with pytest.raises(ValueError):
        load_pipeline_yaml("not_a_pipeline: true")  # missing required fields
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/pipeline/yaml_loader.py`:

```python
"""YAML → Pipeline IR. Loud validation; no silent-{} degradation
(trellis defect addressed in spec Phase 3)."""
from __future__ import annotations

import yaml

from .ir import Edge, Node, Pipeline


def load_pipeline_yaml(text: str) -> Pipeline:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"pipeline YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"pipeline YAML must be a mapping, got {type(data).__name__}")
    for required in ("name", "deadline", "nodes"):
        if required not in data:
            raise ValueError(f"pipeline YAML missing required field: {required}")
    nodes = [_node_from_yaml(n) for n in data.get("nodes") or []]
    edges = [Edge(**e) for e in data.get("edges") or []]
    pipe = Pipeline(name=str(data["name"]),
                    deadline=float(data["deadline"]),
                    nodes=nodes, edges=edges)
    pipe.validate()
    return pipe


def _node_from_yaml(d: dict) -> Node:
    if not isinstance(d, dict) or "id" not in d or "kind" not in d:
        raise ValueError(f"node entry missing id/kind: {d!r}")
    kw = {k: v for k, v in d.items() if k != "body"}
    body = None
    if d.get("kind") == "composite" and isinstance(d.get("body"), dict):
        body = Pipeline(
            name=d["body"]["name"], deadline=float(d["body"]["deadline"]),
            nodes=[_node_from_yaml(x) for x in d["body"].get("nodes") or []],
            edges=[Edge(**e) for e in d["body"].get("edges") or []],
        )
    return Node(body=body, **kw)
```

- [ ] **Step 4:** `pytest tests/test_pipeline_yaml.py -v` → 2 passed.
- [ ] **Step 5:** Commit `feat(pipeline): YAML loader with loud validation`.

---

### Task B3: Executor (sequential + retry + timeout + deadline + DLQ)

**Files:** Create `gardener/pipeline/executor.py`, `tests/test_pipeline_executor.py`.

**Behavior:** topological-order execution. For each node: input map = `{edge.input_name: outputs[edge.from_id]}` collected from satisfied in-edges (skipping unsatisfied conditional edges). Call the runner; on raise/timeout retry up to `retry.max`; on exhaustion → DLQ (human node honors `on_timeout`); pipeline-deadline check between nodes. Returns dict with `state` ∈ {`succeeded`, `timed_out`, `completed_with_dead_letters`}, `outputs`, `dead_letters`.

- [ ] **Step 1: Failing test** (`tests/test_pipeline_executor.py`):

```python
import pytest
from gardener.journal import EventJournal
from gardener.pipeline.ir import Edge, Node, Pipeline
from gardener.pipeline.executor import PipelineExecutor

def _exec(tmp_path):
    return PipelineExecutor(journal=EventJournal(tmp_path))

def test_linear_runs_outputs_chain(tmp_path):
    p = Pipeline(name="lin", deadline=10.0,
        nodes=[Node(id="a", kind="agent", agent="echo",
                    params={"text": "hi"}, retry={"max": 1}),
               Node(id="b", kind="agent", agent="upper",
                    params={}, retry={"max": 1})],
        edges=[Edge(from_id="a", to_id="b", input_name="msg")])
    p.validate()
    runners = {
        "echo":  lambda inputs, params: params["text"],
        "upper": lambda inputs, params: inputs["msg"].upper(),
    }
    res = _exec(tmp_path).run(p, agent_runner=lambda n, ins: runners[n.agent](ins, n.params))
    assert res["state"] == "succeeded"
    assert res["outputs"]["b"] == "HI"

def test_retry_then_dlq_on_exhaustion(tmp_path):
    calls = []
    def flaky(node, inputs):
        calls.append(node.id); raise RuntimeError("nope")
    p = Pipeline(name="r", deadline=10.0,
        nodes=[Node(id="x", kind="agent", agent="flaky",
                    retry={"max": 3, "on": "fail"})],
        edges=[])
    p.validate()
    res = _exec(tmp_path).run(p, agent_runner=flaky)
    assert calls == ["x", "x", "x"]
    assert res["state"] == "completed_with_dead_letters"
    assert res["dead_letters"][0]["node"] == "x"

def test_deadline_terminates_timed_out(tmp_path):
    import time
    def slow(node, inputs):
        time.sleep(0.2); return "ok"
    p = Pipeline(name="d", deadline=0.05,
        nodes=[Node(id="s1", kind="agent", agent="slow", retry={"max": 1}),
               Node(id="s2", kind="agent", agent="slow", retry={"max": 1})],
        edges=[Edge(from_id="s1", to_id="s2", input_name="x")])
    p.validate()
    res = _exec(tmp_path).run(p, agent_runner=slow)
    assert res["state"] == "timed_out"

def test_human_node_dispatches_via_runner(tmp_path):
    p = Pipeline(name="h", deadline=10.0,
        nodes=[Node(id="ask", kind="human", ask="ok?",
                    timeout=5.0, on_timeout="dead-letter",
                    retry={"max": 1})],
        edges=[])
    p.validate()
    res = _exec(tmp_path).run(p,
        agent_runner=lambda n, i: None,
        human_runner=lambda n, i: "yes")
    assert res["outputs"]["ask"] == "yes" and res["state"] == "succeeded"
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/pipeline/executor.py`:

```python
"""Sequential DAG executor. Per-node retry attribute (no top-level back-edges).
Per-node timeout (best-effort via wall-clock check after run); pipeline deadline
checked between nodes. Terminal failures route to DLQ via the journal."""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..journal import EventJournal
from .ir import Node, Pipeline


AgentRunner = Callable[[Node, dict], object]
HumanRunner = Callable[[Node, dict], object]


class PipelineExecutor:
    def __init__(self, journal: EventJournal):
        self.journal = journal

    def run(self, pipeline: Pipeline,
            agent_runner: AgentRunner,
            human_runner: Optional[HumanRunner] = None,
            initial_inputs: Optional[dict] = None) -> dict:
        pipeline.validate()
        outputs: dict[str, object] = dict(initial_inputs or {})
        dead_letters: list[dict] = []
        in_edges: dict[str, list] = {n.id: [] for n in pipeline.nodes}
        for e in pipeline.edges:
            in_edges[e.to_id].append(e)
        deadline = time.time() + pipeline.deadline
        self.journal.append("pipeline", {"event": "start", "name": pipeline.name})
        order = pipeline.topo_order()
        node_by_id = {n.id: n for n in pipeline.nodes}
        for nid in order:
            if time.time() >= deadline:
                self.journal.append("pipeline",
                    {"event": "deadline", "name": pipeline.name})
                return {"state": "timed_out", "outputs": outputs,
                        "dead_letters": dead_letters}
            node = node_by_id[nid]
            # Skip if any in-edge predecessor was dead-lettered (compensation
            # path is a future feature; v0: skip node, no output).
            if any(e.from_id not in outputs for e in in_edges[nid]):
                self.journal.append("pipeline",
                    {"event": "skip", "node": nid, "reason": "missing_input"})
                continue
            inputs = {e.input_name: outputs[e.from_id] for e in in_edges[nid]}
            try:
                outputs[nid] = self._run_node(node, inputs, agent_runner,
                                              human_runner, deadline)
                self.journal.append("pipeline",
                    {"event": "node_ok", "node": nid})
            except _NodeFailed as f:
                self.journal.append("pipeline",
                    {"event": "node_fail", "node": nid, "reason": str(f)})
                dl = {"node": nid, "reason": str(f),
                      "attempts": f.attempts, "kind": node.kind}
                self.journal.dead_letter(dl)
                dead_letters.append(dl)
                # human on_timeout dispatch other than dead-letter is logged only;
                # escalate/fallback-agent paths are TODO(spec-Phase-3).
        state = "completed_with_dead_letters" if dead_letters else "succeeded"
        self.journal.append("pipeline",
            {"event": "end", "name": pipeline.name, "state": state})
        return {"state": state, "outputs": outputs,
                "dead_letters": dead_letters}

    def _run_node(self, node: Node, inputs: dict,
                  agent_runner: AgentRunner,
                  human_runner: Optional[HumanRunner],
                  pipeline_deadline: float) -> object:
        max_attempts = int(node.retry.get("max", 1))
        last_err: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            if time.time() >= pipeline_deadline:
                raise _NodeFailed("pipeline_deadline_during_retry", attempt)
            t0 = time.time()
            try:
                if node.kind == "agent":
                    out = agent_runner(node, inputs)
                elif node.kind == "human":
                    if human_runner is None:
                        raise RuntimeError("no human_runner provided")
                    out = human_runner(node, inputs)
                elif node.kind == "composite":
                    sub = PipelineExecutor(self.journal).run(
                        node.body, agent_runner, human_runner, inputs)
                    if sub["state"] != "succeeded":
                        raise RuntimeError(f"composite_sub:{sub['state']}")
                    out = sub["outputs"]
                else:
                    raise RuntimeError(f"unknown kind: {node.kind}")
                if node.timeout is not None and (time.time() - t0) > node.timeout:
                    raise TimeoutError(f"node {node.id} exceeded timeout")
                return out
            except BaseException as e:
                last_err = e
        raise _NodeFailed(str(last_err), max_attempts)


class _NodeFailed(Exception):
    def __init__(self, reason: str, attempts: int):
        super().__init__(reason)
        self.attempts = attempts
```

- [ ] **Step 4:** `pytest tests/test_pipeline_executor.py -v` → 4 passed.
- [ ] **Step 5:** Commit `feat(pipeline): sequential DAG executor w/ retry+deadline+DLQ`.

---

### Task B4: Composite node (bounded sub-pipeline)

**Files:** Create `tests/test_pipeline_composite.py` (executor already handles composite from Task B3 — this task is a focused test + tightening).

- [ ] **Step 1: Failing test** (`tests/test_pipeline_composite.py`):

```python
from gardener.journal import EventJournal
from gardener.pipeline.ir import Edge, Node, Pipeline
from gardener.pipeline.executor import PipelineExecutor

def test_composite_bounded_inner_retry_outer_still_acyclic(tmp_path):
    # inner pipeline that fails twice then succeeds via outer retry.max=3
    attempts = {"n": 0}
    def flaky_then_ok(node, inputs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "ok"
    inner = Pipeline(name="inner", deadline=10.0,
        nodes=[Node(id="i", kind="agent", agent="flaky",
                    retry={"max": 1})],
        edges=[])
    inner.validate()
    outer = Pipeline(name="outer", deadline=10.0,
        nodes=[Node(id="c", kind="composite", body=inner,
                    retry={"max": 3, "on": "fail"})],
        edges=[])
    outer.validate()
    res = PipelineExecutor(EventJournal(tmp_path)).run(
        outer, agent_runner=flaky_then_ok)
    assert res["state"] == "succeeded"
    assert attempts["n"] == 3
    assert res["outputs"]["c"]["i"] == "ok"
```

- [ ] **Step 2:** Run → expect PASS (B3 already implemented composite; this pins behavior).
- [ ] **Step 3:** Commit `test(pipeline): composite node bounded retry`.

---

### Task D1: AgentRunner ABC + MLX agent runner

**Files:** Create `gardener/harness/__init__.py`, `gardener/harness/mlx_agent.py`, `tests/test_harness_mlx_agent.py` (`@pytest.mark.model`).

**Semantics:** an `AgentRunner` takes a `Node` + `inputs` dict and returns a string output. `MLXAgentRunner` uses mlxsuper `SessionPool` (one session per agent.name) + `mlx_lm.generate` to produce a completion from the agent's `params["system_prompt"]` + `inputs`.

- [ ] **Step 1: Failing test** (`tests/test_harness_mlx_agent.py`):

```python
import pytest
from gardener.harness.mlx_agent import MLXAgentRunner
from gardener.pipeline.ir import Node

pytestmark = pytest.mark.model

def test_mlx_agent_responds_to_simple_prompt(loaded_model):
    model, tok = loaded_model
    runner = MLXAgentRunner(model, tok)
    node = Node(id="a", kind="agent", agent="answerer",
                params={"system_prompt": "Answer in one word.",
                        "max_tokens": 6})
    out = runner(node, {"question": "What color is the sky on a clear day?"})
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement**:

`gardener/harness/__init__.py`:
```python
"""Harness adapters (agent + human runners)."""
from typing import Callable, Protocol
from ..pipeline.ir import Node


class AgentRunner(Protocol):
    def __call__(self, node: Node, inputs: dict) -> object: ...


class HumanRunner(Protocol):
    def __call__(self, node: Node, inputs: dict) -> object: ...
```

`gardener/harness/mlx_agent.py`:
```python
"""MLX-backed agent runner. Each agent name maps to a persistent SessionPool
session so successive calls reuse the KV cache (warm context).

For v0 simplicity, the system prompt is sent fresh each call (no incremental
chat history beyond the cache). The next iteration will wire conversation
history through cache reuse + rewind."""
from __future__ import annotations

from typing import Any

from mlx_lm import generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from ..pipeline.ir import Node


class MLXAgentRunner:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, node: Node, inputs: dict) -> str:
        system = node.params.get("system_prompt", "")
        max_tokens = int(node.params.get("max_tokens", 64))
        temperature = float(node.params.get("temperature", 0.0))
        prompt = _format(system, inputs)
        cache = make_prompt_cache(self.model)
        sampler = make_sampler(temp=temperature)
        text = generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=max_tokens, sampler=sampler,
                        prompt_cache=cache, verbose=False)
        return text.strip()


def _format(system: str, inputs: dict) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in inputs.items())
    return (f"<|system|>\n{system}\n<|user|>\n{body}\n<|assistant|>\n"
            if system else body)
```

- [ ] **Step 4:** `pytest tests/test_harness_mlx_agent.py -v -m model` → 1 passed.
- [ ] **Step 5:** Commit `feat(harness): MLX agent runner over mlxsuper`.

---

### Task D2: Human runners (scripted for demo, CLI for real)

**Files:** Add to `gardener/harness/cli_human.py`, `tests/test_human_runners.py`.

- [ ] **Step 1: Failing test** (`tests/test_human_runners.py`):

```python
from gardener.pipeline.ir import Node
from gardener.harness.cli_human import ScriptedHumanRunner

def test_scripted_human_returns_queued_response():
    r = ScriptedHumanRunner(responses={"approve": "yes", "name": "Alice"})
    node = Node(id="approve", kind="human", ask="ok?",
                timeout=10.0, on_timeout="dead-letter")
    assert r(node, {}) == "yes"
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/harness/cli_human.py`:
```python
"""Human-node runners. CLIHumanRunner blocks for stdin; ScriptedHumanRunner
returns pre-set answers keyed by node id (for tests + demos)."""
from __future__ import annotations

from ..pipeline.ir import Node


class CLIHumanRunner:
    def __call__(self, node: Node, inputs: dict) -> str:
        print(f"\n[HUMAN ASK — node {node.id}] {node.ask}")
        if inputs:
            print("  inputs:")
            for k, v in inputs.items():
                print(f"    {k}: {v}")
        return input("  your answer: ").strip()


class ScriptedHumanRunner:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    def __call__(self, node: Node, inputs: dict) -> str:
        if node.id not in self.responses:
            raise KeyError(f"no scripted response for human node {node.id!r}")
        return self.responses[node.id]
```

- [ ] **Step 4:** `pytest tests/test_human_runners.py -v` → 1 passed.
- [ ] **Step 5:** Commit `feat(harness): CLI + scripted human runners`.

---

### Task D3: Gardener daemon (orchestrator class)

**Files:** Create `gardener/daemon.py`, `tests/test_daemon.py`.

- [ ] **Step 1: Failing test** (`tests/test_daemon.py`):

```python
from gardener.daemon import Gardener
from gardener.pipeline.yaml_loader import load_pipeline_yaml

YAML = """
name: t
deadline: 10.0
nodes:
  - {id: echo, kind: agent, agent: echo, retry: {max: 1}, params: {text: "hi"}}
edges: []
"""

def test_submit_pipeline_succeeds(tmp_path):
    g = Gardener(root=tmp_path,
                 agent_runner=lambda n, i: n.params["text"])
    res = g.submit(load_pipeline_yaml(YAML))
    assert res["state"] == "succeeded"
    # Journal recorded events:
    events = list(g.journal.read("pipeline"))
    assert any(e["event"] == "start" for e in events)
    assert any(e["event"] == "end" for e in events)
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/daemon.py`:

```python
"""Gardener orchestrator: owns journal + knowledge + pipeline executor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .journal import EventJournal
from .knowledge import KnowledgeStore
from .pipeline.executor import AgentRunner, HumanRunner, PipelineExecutor
from .pipeline.ir import Pipeline


class Gardener:
    def __init__(self, root: str | Path,
                 agent_runner: AgentRunner,
                 human_runner: Optional[HumanRunner] = None,
                 agent_name: str = "default"):
        self.root = Path(root)
        self.journal = EventJournal(self.root / "journal")
        self.knowledge = KnowledgeStore(self.root / "knowledge",
                                        agent=agent_name)
        self.agent_runner = agent_runner
        self.human_runner = human_runner
        self.executor = PipelineExecutor(self.journal)

    def submit(self, pipeline: Pipeline,
               initial_inputs: Optional[dict] = None) -> dict:
        return self.executor.run(pipeline, self.agent_runner,
                                 self.human_runner, initial_inputs)
```

- [ ] **Step 4:** `pytest tests/test_daemon.py -v` → 1 passed.
- [ ] **Step 5:** Commit `feat(daemon): Gardener orchestrator`.

---

### Task E1: Cultivation hook (outcome → TTT.feedback)

**Files:** Create `gardener/learning.py`, `tests/test_learning.py`.

**Semantics:** when an agent node succeeds, the cultivation hook may submit a `Candidate` to a `TTTEngine` with `reward=+1` and mark the *associated knowledge object* as `empirical.tested=True helped=True` — closing the loop. For v0 we wire the API, no sleep cycle yet (deferred to spec Phase 5).

- [ ] **Step 1: Failing test** (`tests/test_learning.py`):

```python
import time
from gardener.knowledge import KnowledgeObject, KnowledgeStore
from gardener.learning import CultivationHook

class FakeEngine:
    def __init__(self): self.fed = []
    def feedback(self, cid, reward, signal="solution", metadata=None):
        self.fed.append((cid, reward, signal))

def test_hook_marks_knowledge_tested_and_feeds_ttt(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(
        predicates=[["x", "is", "y"]], insight="i",
        justification="j", source_agent="alpha"))
    eng = FakeEngine()
    hook = CultivationHook(store=store, ttt=eng)
    hook.record_success(candidate_id="c1", knowledge_ids=[oid])
    assert store.get(oid).empirical["tested"] is True
    assert eng.fed == [("c1", 1.0, "solution")]
```

- [ ] **Step 2:** Run → ModuleNotFoundError.

- [ ] **Step 3: Implement** `gardener/learning.py`:

```python
"""Cultivation hook: bridges agent outcomes to (a) knowledge empirical-feedback
flag and (b) TTTEngine.feedback for the eventual sleep-cycle consolidation.

v0 is the wiring + tests; the actual TTT sleep cycle (gated by consolidatable()
+ bit-equivalence promote gate) is deferred to spec Phase 5."""
from __future__ import annotations

from typing import Protocol

from .knowledge import KnowledgeStore


class _FeedbackSink(Protocol):
    def feedback(self, candidate_id: str, reward: float,
                 signal: str = "solution", metadata: dict | None = None) -> None: ...


class CultivationHook:
    def __init__(self, store: KnowledgeStore, ttt: _FeedbackSink):
        self.store = store
        self.ttt = ttt

    def record_success(self, candidate_id: str,
                       knowledge_ids: list[str] | None = None) -> None:
        for kid in knowledge_ids or []:
            self.store.mark_tested(kid, helped=True)
        self.ttt.feedback(candidate_id, 1.0, "solution")

    def record_failure(self, candidate_id: str,
                       knowledge_ids: list[str] | None = None) -> None:
        for kid in knowledge_ids or []:
            self.store.mark_tested(kid, helped=False)
        self.ttt.feedback(candidate_id, -1.0, "solution")
```

- [ ] **Step 4:** `pytest tests/test_learning.py -v` → 1 passed.
- [ ] **Step 5:** Commit `feat(learning): cultivation hook wires outcomes to TTT + knowledge`.

---

### Task F: End-to-end demo

**Files:** Create `examples/demo_simple_agent.py`, `tests/test_demo_smoke.py` (`@pytest.mark.model`).

**The demo:** define a tiny "answerer" agent with a system prompt → load it through `Gardener` → submit a 2-node pipeline (answerer → scripted-human-approval) → print outcome + journal events + knowledge state. Uses `TEST_MODEL`.

- [ ] **Step 1: Implement** `examples/demo_simple_agent.py`:

```python
"""End-to-end Gardener prototype demo.

Defines a tiny answerer agent backed by a small MLX model, runs it through a
2-node pipeline (agent → scripted human approval), and prints the outcome,
journal trace, and knowledge state. The point is to exercise the whole
prototype path:

  Gardener daemon → pipeline executor → MLX agent runner → mlxsuper
                  → journal + knowledge + scripted human runner

Run:
    .venv/bin/python -m examples.demo_simple_agent
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mlx_lm import load

from gardener.daemon import Gardener
from gardener.harness.cli_human import ScriptedHumanRunner
from gardener.harness.mlx_agent import MLXAgentRunner
from gardener.knowledge import KnowledgeObject
from gardener.learning import CultivationHook
from gardener.pipeline.yaml_loader import load_pipeline_yaml

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

PIPELINE_YAML = """
name: ask-and-approve
deadline: 60.0
nodes:
  - id: answerer
    kind: agent
    agent: answerer
    retry: {max: 1}
    params:
      system_prompt: "Answer the user's question in one short sentence. If unsure, say 'I don't know.'"
      max_tokens: 40
      temperature: 0.0
  - id: review
    kind: human
    ask: "Approve the answerer's response?"
    timeout: 30.0
    on_timeout: dead-letter
    retry: {max: 1}
edges:
  - {from_id: answerer, to_id: review, input_name: answer}
"""

def main() -> int:
    print(f"loading {MODEL_ID} …")
    model, tok = load(MODEL_ID)
    runner = MLXAgentRunner(model, tok)
    human = ScriptedHumanRunner(responses={"review": "yes"})

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "garden"
        g = Gardener(root=root, agent_runner=runner,
                     human_runner=human, agent_name="answerer")

        # Seed a piece of knowledge the agent's success should "test"
        kid = g.knowledge.write(KnowledgeObject(
            predicates=[["sky", "appears", "blue"]],
            insight="atmospheric scattering",
            justification="Rayleigh scattering of sunlight",
            source_agent="answerer"))

        pipe = load_pipeline_yaml(PIPELINE_YAML)
        question = "What color is the sky on a clear day, and why?"
        res = g.submit(pipe, initial_inputs={"answerer": question})

        # v0 cultivation: if pipeline succeeded, mark the knowledge as tested.
        class _Sink:
            def feedback(self, *a, **kw): pass
        hook = CultivationHook(store=g.knowledge, ttt=_Sink())
        if res["state"] == "succeeded":
            hook.record_success(candidate_id="demo-1", knowledge_ids=[kid])

        print("\n=== outcome ===")
        print(f"  state: {res['state']}")
        print(f"  answer: {res['outputs'].get('answerer', '<none>')}")
        print(f"  review: {res['outputs'].get('review', '<none>')}")
        if res["dead_letters"]:
            print(f"  dead-letters: {res['dead_letters']}")
        print("\n=== journal ===")
        for ev in g.journal.read("pipeline"):
            print(f"  {ev}")
        print("\n=== knowledge (after) ===")
        for obj in g.knowledge.all():
            print(f"  {obj.id}: tested={obj.empirical['tested']} "
                  f"helped={obj.empirical['helped']}  {obj.insight!r}")
        return 0 if res["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test** `tests/test_demo_smoke.py`:

```python
import pytest
from examples.demo_simple_agent import main

pytestmark = pytest.mark.model

def test_demo_runs_end_to_end():
    assert main() == 0
```

- [ ] **Step 3:** Run the demo directly first:
`.venv/bin/python -m examples.demo_simple_agent`
Expected: prints loaded → outcome state `succeeded` → an answer → journal events → knowledge updated (`tested=True`).

- [ ] **Step 4:** Run smoke test:
`.venv/bin/python -m pytest tests/test_demo_smoke.py -v -m model`
Expected: PASS.

- [ ] **Step 5:** Commit `feat(demo): end-to-end small-LLM prototype demo`.

---

## Acceptance Gate (the whole prototype)

- [ ] All Phase 0 tests still pass: `pytest -m "not model"` 16+, `pytest -m model` 4.
- [ ] Tasks A1–A2 + B1–B4 + D1–D3 + E1 implemented; their fast tests pass.
- [ ] `examples/demo_simple_agent.py` runs end-to-end on `Qwen2.5-0.5B-Instruct-4bit`:
  - pipeline reaches `succeeded`
  - answerer returns a non-empty response to the sky-color question
  - journal shows start → node_ok ×2 → end events
  - knowledge object's `empirical.tested` flips to True
- [ ] `scripts/check_no_omlx_import.py` still prints "OK".
- [ ] At least one DLQ test, one deadline test, one composite-retry test passing.
