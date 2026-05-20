# 🌱 Gardener

**A place to grow long-running agents on your own hardware.**

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/runs%20on-Apple%20Silicon-black?logo=apple)](#)
[![Status: v0 prototype](https://img.shields.io/badge/status-v0%20prototype-orange)](#status)
[![Tests](https://img.shields.io/badge/tests-39%20fast%20%2B%206%20model-green)](#)

Gardener is a single-operator, **local** agent platform for Apple Silicon. You give it a small LLM and a YAML pipeline; it runs the pipeline as a DAG of agents and humans, records every step, and only ever lets your model learn from things you've actually verified worked.

It is built on the **Software Garden** philosophy: *agents are coworkers, not machines. You cultivate the conditions for growth; you don't command execution.* Concretely, that means three commitments that ripple through every design decision:

1. **Pipelines guarantee they end.** Acyclic graphs, bounded retry, per-node timeouts, and a required pipeline `deadline` — together with a **dead-letter queue** for terminal failures — make "runs forever" structurally unrepresentable and "silently lost work" impossible.
2. **Human and agent work are equal first-class citizens.** A `human` node has typed inputs/outputs, routing, journal, and a required timeout policy — exactly like an `agent` node. Agents delegate to humans as peers, not as approval gates.
3. **Knowledge must be tested before it's consolidated.** Every knowledge object carries an `empirical` field. Only entries an execution or a human has *validated* are eligible for the slow-tier learning loop that updates model weights.

The whole thing depends only on stock `mlx-lm` and a few small libraries. There is **no cloud-API fallback** — by design, and CI-enforced.

---

## Quickstart

Requires macOS on Apple Silicon (M-series), Python 3.13+, and ~1 GB free for the demo model.

```bash
git clone git@github.com:terraboops/gardener.git
cd gardener
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run the end-to-end demo (downloads ~300 MB Qwen2.5-0.5B-Instruct-4bit on first run)
.venv/bin/python -m examples.demo_simple_agent
```

You should see:

```
loading mlx-community/Qwen2.5-0.5B-Instruct-4bit …

=== outcome ===
  state: succeeded
  seed:     What color is the sky on a clear day, and why?
  answer:   The sky is usually blue on a clear day. This is because the
            air is transparent and allows light to pass through, creating
            the blue color.
  review:   yes

=== journal ===
  {'event': 'start', 'name': 'ask-and-approve'}
  {'event': 'node_ok', 'node': 'seed'}
  {'event': 'node_ok', 'node': 'answerer'}
  {'event': 'node_ok', 'node': 'review'}
  {'event': 'end', 'name': 'ask-and-approve', 'state': 'succeeded'}

=== knowledge (after) ===
  583f781750d721cf: tested=True helped=1  'atmospheric scattering'
```

Swap `ScriptedHumanRunner` for `CLIHumanRunner` in the demo and you'll answer the approval prompt by hand.

---

## What's in the box

### 🧠 `mlxsuper` — stock `mlx-lm` made superpowered

A thin layer of capabilities every Gardener service needs, ported (never imported) from the hypercar research. CI gate ensures `gardener/` contains zero `omlx` imports.

| Module | What it gives you |
|---|---|
| `SessionPool` | Named KV-cache sessions. `save` / `load` (bit-stable resume, no re-prefill), `fork` (independent branch), `rewind` (O(1) tail drop). Built on `mlx_lm.models.cache.save_prompt_cache` + a deepcopy. |
| `TTTEngine` | Test-time training: `generate_candidates` → `feedback` → `train_step`. LoRA via stock `mlx_lm.tuner`; manual SGD; optional OPLoRA gradient projection. |
| `oplora` | `project_lora_grads` + `compute_svd_cache`. Removes the component of each LoRA gradient along the frozen weight's top-k singular vectors — bounding drift on held-out behavior. |
| `schedules` | `compute_layer_lrs(base_lr, n, schedule)` — uniform / reservoir (shallow frozen → deep adaptive) / cosine. |
| `observability` | `timer`, `counter`, `registry`, `median_of_n`. Zero-cost when `OMLX_OBSERVABILITY=0`. |

### 🧩 Pipelines as composable DAGs

Pipelines are typed graphs validated against a strict IR. Three node kinds, all equal graph citizens:

- **`agent`** — runs an `AgentRunner` (the default `MLXAgentRunner` calls your local model).
- **`human`** — delegates work to a human via a `HumanRunner` (`CLIHumanRunner` for stdin, `ScriptedHumanRunner` for demos/tests). `timeout` + `on_timeout` are compile-time required.
- **`composite`** — a bounded sub-pipeline. The escape hatch for multi-node feedback loops (implement → validate → fix) without ever drawing a back-edge in the outer graph.

```yaml
name: ask-and-approve
deadline: 60.0                       # required — outer termination guarantee
nodes:
  - id: answerer
    kind: agent
    agent: answerer
    retry: { max: 2, on: fail }      # node-attribute retry, not a back-edge
    params:
      system_prompt: "Answer in one short sentence."
      max_tokens: 40
  - id: review
    kind: human
    ask: "Approve the answerer's response?"
    timeout: 30.0
    on_timeout: dead-letter          # required for every human node
    retry: { max: 1 }
edges:
  - { from_id: answerer, to_id: review, input_name: answer }
```

The validator rejects cycles, missing deadlines, unresolved inputs, and human nodes without timeouts — with a *located* error, never a silent `{}`.

### 📓 Journal + Dead-Letter Queue

Append-only JSONL per topic. Every pipeline event is durable and replayable. Terminal failures (retry exhausted, timeout, unresolved input) route to a `dead_letter` topic that's inspectable and re-drivable:

```python
g.journal.dlq_depth()        # 0
g.journal.redrive_dlq()      # → [{'node': 'flaky', 'reason': '...', 'attempts': 3}, ...]
```

Pipelines always end in a *consistent* terminal state: `succeeded`, `timed_out`, or `completed_with_dead_letters`. Never an ambiguous hang.

### 🌾 Knowledge with an empirical-feedback gate

Per-agent `KnowledgeStore` of typed `KnowledgeObject`s (`predicates`, `insight`, `justification`, `source_agent`, `confidence`, …). Predicates are strictly validated on write — malformed triples are *loudly rejected*, not silently emptied (a defect the trellis knowledge system had).

The decisive bit: every object has an `empirical` field. `consolidatable()` only yields entries whose `empirical.tested` is `True`. That's what eventually gates the weight-tier learning loop from consuming untested noise:

```python
oid = store.write(KnowledgeObject(
    predicates=[["sky", "appears", "blue"]],
    insight="atmospheric scattering",
    justification="Rayleigh scattering of sunlight",
    source_agent="answerer",
))
# … pipeline runs and validates the insight …
store.mark_tested(oid, helped=True)
list(store.consolidatable())   # ← now eligible for the slow-tier learning cycle
```

### 🔄 Cultivation hook (the loop closer)

`CultivationHook` bridges outcomes back into the system: an agent success marks the relevant knowledge objects as `empirical.tested=True helped+=1` *and* feeds a `TTTEngine` a positive reward. A failure does the opposite. The hook is the wiring; the full sleep cycle (LoRA SGD + OPLoRA projection + bit-equivalence promote gate) is mlxsuper-ready and deferred to v1.

---

## Architecture

```
                ┌──────────────────── Gardener daemon ────────────────────┐
                │                                                          │
   submit(pipe) │   PipelineExecutor ──dispatch──▶  AgentRunner            │
       ───────▶ │      ▲                              (MLXAgentRunner ───┐ │
                │      │                                                 │ │
                │      │                            HumanRunner          │ │
                │      │                              (Scripted/CLI)     │ │
                │      │                                                 │ │
                │   EventJournal  ◀─every step─                          │ │
                │   (JSONL + DLQ)                                        │ │
                │      │                                                 │ │
                │   KnowledgeStore ◀─CultivationHook─ outcome            │ │
                │   (empirical gate)                                     │ │
                └────────────────────────────────────────────────────────┼─┘
                                                                         │
                              ┌──────────────────────────────────────────┴─┐
                              │           mlxsuper (stock mlx_lm +)         │
                              │  SessionPool · TTTEngine · OPLoRA · …       │
                              └──────────────────────────────────────────────┘
```

---

## Project layout

```
gardener/
  mlxsuper/            # ported: sessions, TTT, OPLoRA, LR schedules, observability
    __init__.py
    session.py
    ttt.py
    oplora.py
    schedules.py
    observability.py
  pipeline/            # composable DAG: IR + YAML loader + executor
    ir.py
    yaml_loader.py
    executor.py
  harness/             # runners for agent and human nodes
    __init__.py        # AgentRunner / HumanRunner protocols
    mlx_agent.py       # MLXAgentRunner — talks to a local MLX model
    cli_human.py       # CLIHumanRunner + ScriptedHumanRunner
  journal.py           # append-only JSONL + DLQ + redrive
  knowledge.py         # KnowledgeStore + KnowledgeObject + empirical gate
  daemon.py            # Gardener orchestrator class
  learning.py          # CultivationHook (outcome → TTT + knowledge)
docs/
  design.md            # the full v0 design spec
  plans/               # phase-gated implementation plans
examples/
  demo_simple_agent.py # the end-to-end demo above
scripts/
  check_no_omlx_import.py   # CI gate: no `import omlx` allowed
tests/                 # 39 fast + 6 model tests
```

---

## Status

This is a **v0 prototype** — the smallest end-to-end vertical that exercises the whole architecture. The pieces are real and tested; the deep hardening of each subsystem is scheduled, not done.

| In v0 ✅ | Deferred 📋 |
|---|---|
| mlxsuper core (session save/load/fork/rewind, TTT, OPLoRA, schedules, observability) | Block-pool pre-cached subagent swarm |
| Composable-DAG pipeline IR + YAML loader + executor | Prose DSL, visual composer, agent-authored `compose_pipeline` at runtime |
| Journal + DLQ + re-drive | Trellis scheduler lift + priority/starvation/deadline hardening (Phase 2) |
| Knowledge store with K2 (strict validation) + K10 (empirical gate) | K1/K3-K9 deep knowledge defects (Phase 4) |
| Cultivation hook wiring | Full TTT "sleep" cycle + bit-equivalence promote gate (Phase 5) |
| Bidirectional human↔agent (`human` node kind) | Chalet-style git-projection human surface |
| MLX agent runner (local Apple Silicon model) | Pi RPC harness adapter |

See [`docs/design.md`](docs/design.md) for the full design and [`docs/plans/`](docs/plans/) for the phase-by-phase plans.

---

## Tests

```bash
# Fast unit tests (no model load):
.venv/bin/python -m pytest -m "not model" -v

# Model integration tests (load Qwen2.5-0.5B; ~6s after warmup):
.venv/bin/python -m pytest -m model -v

# CI gate (must always pass — repo is port-not-depend):
.venv/bin/python scripts/check_no_omlx_import.py
```

Current: **39 fast + 6 model tests passing.**

---

## Philosophy: the Software Garden

Gardener is the first hypercar-native instantiation of an idea that predates it: an agent system designed for **emergence through cultivation** rather than command. The Garden has six core principles; the three that shape Gardener most:

> **Agents are coworkers, not machines.** Work flows in both directions — humans delegate to agents *and* agents delegate to humans, as peers.
>
> **Everything is observable; nothing is silently lost.** Every step is journaled. Every failure is dead-lettered, never dropped. Every termination state is named (`succeeded`, `timed_out`, `completed_with_dead_letters`).
>
> **Knowledge gets tested, not just accumulated.** The empirical-feedback gate is the smallest possible expression of this: a single bit (`tested`) per knowledge object, separating "we think this works" from "we've verified it works." Only the latter is allowed to teach the model.

---

## Contributing

Issues and PRs welcome. By contributing, you agree your contributions are licensed under the same AGPL-3.0 as the project. If you're planning a non-trivial change, please open an issue first so we can align on direction.

---

## License

[GNU Affero General Public License v3.0](LICENSE). Strong copyleft + network-use clause — if you run a modified Gardener as a hosted service that users interact with over a network, you must offer source under the same terms.
