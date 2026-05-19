# The Gardener — a hypercar-native Software Garden

**Status:** Design (approved for spec, pre-implementation-plan)
**Date:** 2026-05-18
**Author:** terra + Claude (brainstorming session)
**Spec location:** `gardener/docs/design.md` (self-contained project folder;
to be carved out of `omlx-mamba3` into its own repo later)

---

## Context

`omlx-mamba3` ("hypercar") is a high-performance local LLM inference engine. Its
research has produced primitives that, taken together, enable something no
cloud-API agent can be: a persistent agent that **genuinely grows** —
weight-level continual learning (`omlx/ttt.py`), catastrophic-forgetting safety
(`omlx/oplora.py`), O(1) context fork/rewind, and quantized session save/load
(`omlx/agentic.py`, `omlx/turboquant_kv.py`).

The user wants to turn these primitives into a **prototype platform for
cultivating long-running persistent agents**: agents handle routine knowledge
work (coding, social, ops); humans handle the novel. The portfolio already has
the doctrine for this — the **Software Garden** philosophy (mahdi
`knowledge/software-garden-design.md`): *agents are coworkers not machines;
cultivation over command; emergence through shared and **tested** knowledge;
event-sourcing backbone; role-tailored projections for human contribution.*

The intended outcome: the first hypercar-native instantiation of the Software
Garden — code-named **the Gardener** — reusing proven mechanisms from the
`incubator/trellis` portfolio where they are solid and rebuilding them where
they are fragile, with **Pi** (https://pi.dev) as the thin worker shell and
**Hermes Agent** (Nous) as design inspiration.

### Decisions locked during brainstorming

- **Learning depth:** *both, layered* — fast memory/skills tier + periodic
  weight-level TTT/OPLoRA "sleep" consolidation.
- **Harness:** *Pi as the shell*, Hermes patterns reimplemented against
  hypercar-native APIs. Not a new agent framework.
- **Use cases (IS):** persistent portfolio engineer · self-improving from
  experience · parallel subagent swarm · long-horizon autonomous tasks ·
  an orchestration layer for agentic pipelines ("autonav on steroids +
  trellis-style swarming").
- **Work is bidirectional and symmetric.** Humans delegating to agents and
  agents delegating to humans are *equally* first-class. A `human` node is the
  structural peer of an `agent` node — typed inputs/outputs, routing, journal,
  timeout/escalation — not a passive review gate. (Software Garden principle
  5; enables "agents handle routine, humans handle novel" to run unattended.)
- **Pipelines are flexible composable graphs, not rigid phases.** Explicit
  correction of trellis's central mistake: trellis pipelines are an *ordered
  list of stages run sequentially per idea* (`incubator/trellis/docs/pipelines.md`
  — `agents:` is a linear list; `parallel_groups` only controls serialization).
  In the Gardener, a pipeline is a **DAG of agent invocations** that **agents
  themselves can author and submit at runtime**, not a fixed phase ladder. We
  keep trellis's good idea — a dual interchangeable surface format (YAML + a
  `prose` declarative DSL) compiling to one canonical IR
  (`incubator/trellis/core/pipeline_format.py`, `prose_parser.py`) — but
  redesign that IR from linear stages to a composable graph.
- **Every pipeline is guaranteed to terminate, and no failed work is lost.**
  Four composing bounds (acyclic graph · bounded node retry · per-node timeout
  · required configurable whole-pipeline `deadline`) guarantee a terminal
  state. Terminal failures (retry exhausted, timeout, unresolved input) route
  to a **dead-letter queue** — a durable, inspectable, re-drivable journal
  topic — so pipelines end *consistently* (`succeeded` / `timed_out` /
  `completed_with_dead_letters`), never hang, and never silently drop work.
- **Non-goals (IS NOT) for v1:** not a cloud-model wrapper · not multi-user
  SaaS · not a synchronous IDE copilot · not a from-scratch framework.
  **But** the architecture must not foreclose later: distributed agent teams,
  multi-user access, a synchronous copilot mode, and first-class human
  co-editing alongside agents.

---

## Architecture (recommended approach: Gardener daemon + thin Pi workers)

A long-running **Gardener daemon** owns the platform and its persistent state.
**Pi instances are disposable workers** the daemon drives over Pi's RPC/SDK
mode. Five internal services, each a clean seam so the v1 prototype can grow
into the full distributed Software Garden without a rewrite.

```
        ┌─────────────────────────── Gardener daemon ───────────────────────────┐
        │                                                                        │
   task │   Scheduler ──dispatch──▶ Pi worker (RPC/SDK) ──/v1/chat──▶ hypercar    │
  intake│      ▲                         │ outcome                     server     │
        │      │                         ▼                                        │
        │  Knowledge store ◀──fast-tier── event journal ──feeds──▶ Learning       │
        │  (hardened)            (append-only JSONL)              ("sleep":       │
        │      │                         │                         TTT+OPLoRA)    │
        │      └──── empirical gate ──────┘                              │        │
        │                                                                ▼        │
        │  Session/Subagent pool ◀── hypercar /v1/sessions/* + block-pool          │
        └────────────────────────────────────────────────────────────────────────┘
```

### Dependency stance: port, don't depend (the map principle)

The Gardener **does not import `omlx`**. omlx-mamba3 is a **reference map**,
not a dependency. The Gardener depends only on the same *base* libraries the
fork does (`mlx`, `mlx-lm`, `numpy`, `pyyaml`, `lark`, plus a web framework for
the daemon) and **ports only the hypercar techniques that make base MLX
superpowered** into `gardener/`. Every omlx `file:line` below is a *spec to
re-derive against and test*, not an import target — the
`feedback_verify_repo_claims` discipline applied structurally: re-implement the
primitive on stock mlx/mlx_lm, pin it with our own tests, never trust the
churning research fork at runtime. Consequence: a new **Phase 0** that ports a
thin "superpowered-MLX core" (`gardener/mlxsuper/`): a minimal mlx_lm-based
inference + TQ-style KV cache with fork/rewind/save-load, the TTT+OPLoRA
learning ops, and an observability shim. This is the price of a clean
carve-out; it is bounded (we port only what the five services consume).

### Service reuse map (empirically grounded)

omlx refs = **port-from spec**, not import. Base-lib deps only.

| Service | Built from | Verdict | Reference map (re-derive + test) |
|---|---|---|---|
| **Superpowered-MLX core** (Phase 0) | hypercar KV/session + TTT/OPLoRA techniques on stock mlx_lm | **Port the technique** onto base mlx/mlx_lm; own tests | `omlx/turboquant_kv.py:1248-1765` (TQ KV: `fork`/`rewind_to`/`save_to_disk`/`load_from_disk`); `omlx/hypercar_server.py:410-680` (session model); `omlx/ttt.py:144-600`; `omlx/oplora.py:87-150`; `omlx/ttt_schedules.py:23` |
| **Session / Subagent pool** | the ported core's session/KV primitives + block-pool; quantize-on-save | **Build on ported core** (no omlx runtime) | maps `omlx/hypercar_server.py:410-680`, `omlx/turboquant_kv.py:1624-1765` |
| **Scheduler** | trellis priority queue + TLA+ invariants | **Reuse + harden**; extend job model to DAG-node dependency gating | `incubator/trellis/orchestrator/pool.py:102-1116`, `job_queue.py:1-192`, `incubator/specs/pool_scheduler.tla` |
| **Pipeline engine** | trellis dual-format (YAML + prose DSL) *idea*; IR redesigned linear→DAG | **Reuse format/composer idea, reimplement IR** | idea: `incubator/trellis/core/pipeline_format.py`, `prose_parser.py`, `incubator/trellis/docs/pipelines.md` |
| **Knowledge store** | trellis `KnowledgeObject` *concept only* | **Reimplement** (10 known defects) | concept: `incubator/trellis/tools/knowledge_io.py`, `orchestrator/evolution.py` |
| **Learning ("sleep")** | the ported TTT+OPLoRA ops; Gardener orchestrates the cycle | **Build on ported core** | maps `omlx/ttt.py`, `omlx/oplora.py`, `omlx/ttt_schedules.py` |
| **Event journal** | Chattermax-shaped append-only JSONL (not Kafka yet) | **New, minimal** — the future-distributed seam | inspiration: `~/Developer/chattermax` hook system |

**Pi wiring** follows the existing integration pattern exactly
(`omlx/integrations/opencode.py:43-68`): a provider entry pointing at
`http://localhost:<port>/v1`. No Pi fork; the Gardener never imports Pi
directly — it speaks Pi's RPC/SDK across an adapter (the autonav
pluggable-harness idea, `~/Developer/autonav/packages/autonav/src/harness/types.ts`).

---

## Composable pipelines (the trellis-rigidity correction)

Trellis's central mistake is rigidity: a pipeline is a fixed, ordered phase
ladder authored ahead of time by a human, and `parallel_groups` only decides
what *can't* overlap. Agents cannot compose new pipelines; branching and
data-dependent flow are impossible.

**The Gardener's pipeline is a DAG, and pipelines are first-class artifacts
agents can author at runtime.**

### Why a validated IR (not just a dict in the middle)

Trellis already compiles its formats into one canonical dict
(`pipeline_format.py`) — yet `load_pipeline`'s own docstring admits corrupt
input was coerced to `{}` silently. That proves the point: the benefit is not
the indirection, it is that the IR is a **typed, validated, normalized
contract**. That contract is precisely what makes agent-authored pipelines
safe to operate:

1. **One validation/safety chokepoint.** Acyclicity, budget/depth caps,
   agent-name resolution, input resolution, and the auto-human-review-gate
   rule are enforced *once*, at compile-to-IR — not re-checked (or, like
   trellis, missed) at every consumer. An agent cannot submit a structurally
   dangerous graph.
2. **Self-correction loop.** A typed compile step emits located errors
   (`cycle: A→B→A`, `node 'review' references undefined input 'patch'`) the
   authoring agent can read and repair autonomously. Opaque failures break
   that loop.
3. **Pipelines are manipulable data.** Lossless IR ↔ prose ↔ YAML round-trip
   lets an agent load an existing pipeline, mutate nodes/edges
   programmatically, and re-emit — "agents construct pipelines" is only
   feasible if a pipeline is editable structure, not spliced strings.
4. **Stable analysis substrate.** Dependency-gated scheduling, dry-run/cost
   estimation, observability mapping, and the TLA+ acyclicity+dependency-order
   invariant all operate on the one normalized model; properties are provable
   against an IR, not against N surface dialects.
5. **Decoupled evolution.** Surface syntax (prose grammar, visual composer,
   another agent's format) can change freely while it compiles to the same
   IR; the executor and any *learned* pipeline-authoring behavior don't break.

Tradeoff acknowledged: an IR costs a compile/validate layer and a spec to
maintain. For trellis's original fixed/human-authored/linear assumption that
cost was marginal — which is why a loose dict survived. The IR's payoff
scales directly with how much **agents, not humans, are the authors**, which
is the platform's premise. This is why the IR is in scope for v1 despite the
"prototype, not a framework" non-goal.

### Canonical IR — agent DAG

A pipeline is a graph:

- **Nodes** have a `kind`. Three kinds, all equal graph citizens with the
  same `{id, inputs, params}` shape, the same edge/condition/journal/round-trip
  treatment:
  - `agent` — an agent invocation: `{kind:agent, id, agent, params, gate}`.
  - `human` — a unit of work delegated **to a human** (see "Bidirectional
    human↔agent work" below): `{kind:human, id, ask, assignee, response_schema,
    timeout, on_timeout}`.
  - `composite` — a node whose body is a bounded sub-pipeline (the loop
    escape hatch; keeps the outer graph acyclic).
- **Edges** = dependency + data flow: a node runs once its predecessors
  satisfy their edge condition; a predecessor's output is addressable as a
  named input to successors.
- **Edge conditions** = optional predicate on a predecessor's result
  (`proceed` / `iterate` / custom signal) → enables branching and
  conditional fan-out without a separate "phase" concept.
- **Fan-out / fan-in** = a node may map over a list (spawn N subagents) and a
  join node aggregates — this is the swarm primitive, expressed in the graph
  rather than bolted on as `post_ready`.
- A linear 4-stage trellis pipeline is just the degenerate path-graph case, so
  nothing is lost — only un-rigidified.

### Bidirectional human↔agent work (first-class, both directions)

A core Software-Garden tenet (principle 5, *accessible contribution*; "agents
are coworkers, not machines") is that **work flows both ways**. Humans
delegating to agents is obvious; agents delegating to humans must be **exactly
as first-class** — not a side-channel, not merely a review checkpoint.

Two distinct mechanisms, deliberately separated:

- **`gate` (lightweight, kept from trellis):** pause-and-decide on an *agent*
  node's output — `auto` / `human-review` / `llm-decides`. The human only
  approves / sends back for `iterate`. No new work product.
- **`human` node (new, first-class):** an actual unit of work *assigned to a
  human*, symmetric with an `agent` node. It has `inputs` (data the agent
  hands the human), an `ask` (what's needed), an `assignee` (routing — a
  person/role), a `response_schema` (typed result, so downstream nodes consume
  it like any node output), and — critically for autonomy — a `timeout` with
  an `on_timeout` policy edge (`escalate` / `fallback-agent` / `dead-letter`;
  `dead-letter` is the safe default — the ask is preserved and re-drivable,
  not lost). The
  human's response is a normal node output: it flows along edges, satisfies
  conditions, participates in fan-out/fan-in and budgets (a **time budget**
  rather than a $ budget).

Why the distinction matters: a gate can't carry a work product or be routed,
scheduled, retried, or learned from. A `human` node can — it is the structural
peer of an `agent` node.

**Delivery surface:** human nodes are dispatched through the projection seam
(Chalet-style per-document-git surface, see Seams) — that read projection
becomes an **active inbound work channel**: the human receives the ask with
its inputs, answers in-surface, and the typed response re-enters the graph.

**Operability / autonomy safety:** because a human node can block
indefinitely, `timeout` + `on_timeout` are **required** fields (compile-time
enforced) — long-horizon autonomous pipelines must never deadlock waiting on
a human. This is what lets "agents handle the routine, humans handle the
novel" run unattended: unanswered human asks escalate or fall back by policy.

**Learning signal:** a human node's response is the highest-value
empirical-feedback signal in the system — it feeds the cultivation loop's
reward exactly like an execution result (a human "this is right/wrong/do-this
instead" is a strong supervised signal for the TTT sleep cycle). See the
cultivation loop.

### Dual surface format (technique kept from trellis, grammar + IR new)

What we keep from trellis is the *technique*, not the grammar: the
`pipeline_format.py` dispatch pattern (extension → format → one canonical IR)
and the lark + indentation-token approach in `prose_parser.py`. Trellis's
actual prose grammar is **inadequate and rigid** — it has exactly five
constructs (`pipeline`, `description`, `session`, `gate`, a single `parallel:`
block) and compiles to the same flat linear-stage dict; it has no concept of
edges, data flow, conditions, branching, per-node inputs/params, loops, or
nesting. It is a syntax skin over the rigid model, **not** a general
orchestration language.

Two interchangeable authoring formats compiling to the one DAG IR:

- **YAML** — explicit `nodes:` / `edges:` for tooling and the visual composer.
- **Prose DSL** — a **new grammar** (reusing the lark/indentation technique,
  not trellis's production) expressive enough for the full DAG IR: named
  nodes, edges with result-predicate conditions, per-node inputs/params,
  fan-out/fan-in, and bounded retry/loop. This is the format agents emit, so
  it must round-trip losslessly with the IR.

`load_pipeline` must **not** repeat trellis's failure mode (its own docstring
admits corrupt templates became opaque 500s or were silently dropped, coerced
to `{}`): the Gardener validates to a typed IR and **rejects loudly** with a
located parse error — never silent degradation.

### Agents author and submit pipelines at runtime

A planner agent has a `compose_pipeline(spec)` capability: it emits a prose/
YAML pipeline, the engine compiles + validates it to the DAG IR, and submits
it to the scheduler. Each ready DAG node becomes a scheduler **job**; the
existing priority dispatch, parallel-group serialization, and gating
(`auto` / `human-review` / `llm-decides`, `pool.py:611-626`) apply per node.
The scheduler extension is dependency gating: a node-job is dispatchable only
when its in-edges are satisfied. This reuses the TLA+-verified core; the
added invariant to model-check is **DAG acyclicity + no node runs before its
dependencies**.

### Termination guarantee (decision)

Every pipeline is **guaranteed to reach a terminal state**, enforced by four
composing bounds:

1. **Acyclic graph** — no top-level back-edges (below), so no infinite
   traversal.
2. **Bounded node retry** — retry is a **node attribute**
   (`retry:{max,on,backoff}`), never a graph back-edge; `max` is a required
   bounded int. Multi-node feedback loops (implement→validate→fix) use a
   `composite` node (a bounded sub-pipeline with its own internal retry), so
   the cycle is encapsulated and bounded, never a free back-edge. This is the
   structured-bounded-loop vs. goto choice — agent-authorship makes
   "non-termination is unrepresentable" the dominant safety property.
3. **Per-node timeout** — every node has a wall-clock cap; `human` nodes
   require it explicitly.
4. **Pipeline deadline** — a **required, configurable whole-pipeline
   timeout** (`deadline`, a **policy default, overridable per pipeline**; the
   concrete default value for long-horizon work is deferred to the
   implementation plan). When the deadline elapses, in-flight nodes are
   cancelled and the pipeline transitions to terminal `timed_out`. This is the
   outer guarantee even if a node's own bound is mis-set.

### Dead-letter queue (terminal-failure handling, for consistency)

A node that exhausts its bounded retry, hits its per-node timeout with no
recovery edge, or whose required input never resolves does **not** crash or
hang the pipeline. The
work item — node id, inputs, last error, attempt history — is routed to a
**dead-letter queue (DLQ)**. The DLQ is a durable topic of the event journal
(append-only, inspectable, **re-drivable**). v1 scope: **operator-triggered
re-drive only**; an autonomous DLQ-handler *agent* is a documented seam (the
DLQ being a journal topic means a handler agent subscribes later with no
schema change), deliberately out of v1 to keep the failure path
human-supervised in the prototype. Consequences:

- The pipeline always reaches a *consistent* terminal state — `succeeded`,
  `timed_out`, or `completed_with_dead_letters` — never an ambiguous hang.
- No failed work is silently lost (the explicit anti-pattern from trellis's
  silent-`{}` and silent-knowledge-drop failures).
- An edge may be conditioned on a predecessor being dead-lettered, so a
  pipeline can author its own compensation/cleanup path.
- DLQ depth is a first-class observability signal (registry counter).

### Safety rails (agent-authored graphs are powerful)

- Compile-time: acyclicity check, max node/depth caps, every `agent` resolves
  in the registry, every named input resolves to an upstream output, every
  `human` node has a `timeout` + `on_timeout` policy, every `retry.max` is a
  bounded int, and **every pipeline declares a `deadline`** (or inherits the
  policy default) — all required, rejected loudly if missing.
- Run-time: the pipeline `deadline` cancels in-flight nodes and terminates the
  run; per-node timeout + bounded retry feed the DLQ on exhaustion; per-node
  budget (reuse trellis `max_budget_usd`; `human` nodes carry a **time
  budget**); a human-review gate auto-inserted when an agent-authored pipeline
  exceeds a configured node/budget threshold.

## The cultivation loop (what makes it "grow")

Two learning tiers, gated by **empirical feedback** — the Software Garden's
principle #6 and exactly the piece trellis's knowledge system lacks.

1. **Fast tier (every turn):** a Pi worker completes a task; the outcome
   (test result / tool result) is appended to the event journal and recorded as
   a **knowledge object** (hardened store).
2. **Empirical gate:** knowledge is marked *tested* only when a signal
   validated it — an execution result via `TTTEngine.feedback_from_execution()`
   (`omlx/ttt.py:272`, reward ±1 by pass/fail), **or a `human` node response**,
   which is the highest-value supervised signal in the system (a human
   "right / wrong / do this instead" maps directly to reward and is weighted
   above an automated pass/fail). Untested knowledge is retrievable but
   **never consolidated**.
3. **Slow tier ("sleep" / Evolution Cycle):** the Gardener periodically runs
   `TTTEngine.train_step()` (`omlx/ttt.py:297`) on `reward > 0` candidates only,
   with `OPLoRA.project_lora_grads` (`omlx/oplora.py:87`) projecting gradients
   onto the safe subspace (anti-forgetting) and `compute_layer_lrs("reservoir")`
   (`omlx/ttt_schedules.py:23`) freezing shallow layers.
4. **Promote gate:** before an adapter checkpoint is promoted, a
   **bit-equivalence + held-out regression** check runs (the pattern hypercar
   already uses for the TTT head router, `tests/test_ttt_head_router.py`). On
   regression, `engine.rewind()` (`omlx/hypercar_server.py:781-793`).

This directly closes trellis knowledge-system defect #10 (untested accumulation)
and gives the platform a weight substrate no cloud agent can have.

---

## Pre-cached subagents (explicit requirement)

Each subagent is a **named hypercar session**:

- Prefill the subagent's role/context once → `/v1/sessions/save` → frozen as
  **TQ3-quantized on the fly** (`turboquant_kv.py:1670` quantizes the fp16
  warmup buffer before writing).
- Spawn-time `/v1/sessions/load` resumes in ~1.5s with **no re-prefill**.
- `/v1/sessions/fork` branches a subagent for divergent exploration
  (`turboquant_kv.py:1624`, O(cache_size), no re-prefill).
- `/v1/sessions/rewind` is O(1) backtrack (`turboquant_kv.py:1648`).
- Scheduler **parallel groups** (`pool.py:68-99` `can_schedule`) map onto
  concurrent forked sessions held resident in the block-pool.

---

## Hardening required for the trellis lifts

The lifts are **not** copy-paste. Each carries fixes, each fix gets a
failing→passing test (user directive: *take carefully, test liberally*).

### Scheduler (REUSE + harden) — `pool.py`, `job_queue.py`

Solid core (TLA+-verified invariants: max-concurrent, serial-group, run-count,
iter-bound, killed-terminal, no-starvation). Harden:

- **H1** `_coerce_priority` applied at *every* status read, not just
  `_get_active_ideas` (else hand-edited `priority_score: null` → `TypeError`).
- **H2** Cap `JobQueue` depth; cull stale jobs (currently unbounded).
- **H3** Validate `pipeline.parallel_groups` at load (silent serialization
  break on config drift today).
- **H4** Cap feedback-job fan-out per scan (currently O(live ideas) per cycle).
- **H5** Make `MAX_RATE_LIMIT_HITS` configurable (hard-coded 6 today).

### Knowledge store (REIMPLEMENT) — concept from `knowledge_io.py`/`evolution.py`

Schema retained (id, predicates, insight, justification, idea_context,
timestamps, source_agent, confidence) **plus** an `empirical` field:
`{tested: bool, uses: int, helped: int, last_validated_at}`. Defects to
test-fence:

| # | Trellis defect | Required behavior + test |
|---|---|---|
| K1 | Unbounded growth | per-agent cap + age/decay archival; test: N+1 insert evicts oldest |
| K2 | Lossy predicate validation | strict normalize-or-reject on load; test: malformed triple rejected, not silently emptied |
| K3 | 8-char hash collision | full-length id or collision-detect on save; test: forced collision fails loudly |
| K4 | Scattered null coercion | single `validate_object()` on load; test: `null` justification rejected |
| K5 | Atomic-write read race | directory snapshot read; test: concurrent write→read never partial |
| K6 | Invalid LLM curation actions | action schema validation; test: malformed action logged + skipped, not silent |
| K7 | Merge keeps stale id | recompute id on merge; test: merged entry id ≠ any source id unless content-identical |
| K8 | No concurrent-write guard | per-agent file lock; test: two writers, no lost update |
| K9 | Loose idea_context | validated slug list, order-stable; test |
| K10 | **No empirical feedback** | `empirical` field + gate; test: untested knowledge never consolidated |

### Agent-creator wizard (REIMPLEMENT — does not exist in trellis)

trellis README promises it; `core/agent_factory.py` is a deterministic loader
with zero validation. Build a real interactive wizard: schema-validated
`registry.yaml` entry, **tool-name validation against the live SDK tool list**
(typos silently fail at runtime today), `prompt.py` contract check.

### Observability (REIMPLEMENT — trellis is a 10s JSON snapshot)

Reuse hypercar's own observability stack instead
(`omlx/observability/` — `timer`, `mlx_timer`, `median_of_n`, `counter`,
`registry`, `trace_class`/`trace_module`). Gardener emits structured per-agent
events into the journal; the registry surfaces p50/p95 per service.

---

## Seams left deliberately open (the "leave options open" requirement)

- **Event journal interface** → swap local JSONL for a Chattermax XMPP/Kafka
  stream later (distributed agent teams) with no consumer changes.
- **Projection / human-work interface** → already bidirectional in v1: it
  delivers `human` node asks (inbound work to humans) and accepts typed
  responses back into the graph, plus read projections. v1 ships a minimal
  surface (CLI/file inbox); the seam lets a Chalet-style per-document-git
  surface drop in later for richer first-class human co-editing (Chalet's
  reusable idea: git history as the async human+agent coordination layer).
  The interface is symmetric by design — not a read-only projection that
  grows write later.
- **Harness adapter** → Pi today; the Gardener depends on an adapter interface,
  never on Pi internals (autonav's LLM-agnostic, pluggable-harness pattern).

---

## Critical files

**To create (all under the self-contained `~/Developer/gardener/` repo):**
- `mlxsuper/` — **ported** superpowered-MLX core on stock mlx/mlx_lm: TQ-style
  KV cache with `fork`/`rewind`/`save`/`load`, session model, TTT+OPLoRA ops,
  observability shim (Phase 0)
- Gardener daemon entrypoint + service wiring
- `scheduler/` — lifted + hardened from trellis pool/job_queue, extended with
  DAG-node dependency gating
- `pipeline/` — canonical DAG IR + dual-format loader (YAML + prose DSL) +
  compile/validate + `compose_pipeline` agent capability
- `knowledge/` — reimplemented store with empirical-feedback gate
- `journal/` — append-only JSONL event log + replay + DLQ topic
- `harness/` — Pi RPC/SDK adapter (interface + Pi impl)
- `learning/` — orchestrator around the ported TTT/OPLoRA "sleep" cycle

**Base-library dependencies only** (same as the fork, never `omlx` itself):
`mlx`, `mlx-lm`, `numpy`, `pyyaml`, `lark`, a daemon web framework. Pinned to
the same major versions omlx-mamba3 uses (read its `pyproject.toml` as the
version map).

**Reference map to port-from + test (omlx — NOT imported):**
- `omlx/turboquant_kv.py:1248-1765` (TQ KV cache, fork/rewind/save/load)
- `omlx/hypercar_server.py:410-680` (session model)
- `omlx/ttt.py:144-600`, `omlx/oplora.py:87-150`, `omlx/ttt_schedules.py:23`
- `omlx/observability/*` (timer/registry shim shape)
- `omlx/integrations/base.py:16-80` + `opencode.py:43-68` (Pi provider wiring
  pattern)

**To lift with hardening (trellis — copied source, then hardened):**
- `incubator/trellis/orchestrator/pool.py`, `job_queue.py`,
  `incubator/specs/pool_scheduler.tla`

---

## Phasing (ordered by dependency & acceptance gate, not calendar)

0. **Superpowered-MLX core (port)** — `gardener/mlxsuper/` on stock
   mlx/mlx_lm: load a model, generate, a TQ-style KV cache with
   fork/rewind/save/load, the TTT+OPLoRA training ops, an observability shim.
   No `omlx` import anywhere.
   *Gate:* loads a small MLX model and generates; KV save→load resumes with
   no re-prefill and bit-stable continuation; fork then divergent generation
   is independent; one TTT `train_step` on a positive sample lowers loss and
   an OPLoRA-projected step leaves a held-out probe within tolerance;
   `pip`-installable with base deps only — `import omlx` is absent
   (grep-gated in CI).
1. **Spine** — Gardener daemon + event journal (+ DLQ) + session pool on the
   ported core + one Pi worker round-trip.
   *Gate:* a Pi worker completes a task driven by the daemon; journal replays
   it deterministically; a forced node failure lands in the DLQ.
2. **Scheduler lift** — port trellis pool/job_queue + H1–H5.
   *Gate:* TLA+ invariants hold; H1–H5 regression tests pass; trellis's own
   existing scheduler tests pass against the lift unchanged.
3. **Pipeline DAG engine** — canonical DAG IR (`agent` / `human` / `composite`
   node kinds), dual-format loader (YAML + prose DSL) with loud validation,
   scheduler dependency-gating extension, `compose_pipeline` agent capability,
   bidirectional human-work interface (minimal CLI/file inbox).
   *Gate:* a linear trellis template imports unchanged (degenerate path graph);
   a diamond DAG and a conditional branch execute correctly; an agent-authored
   prose pipeline round-trips and runs; a `human` node dispatches an ask,
   accepts a typed response that flows downstream, and a `timeout` triggers
   its `on_timeout` policy; node-attribute retry is bounded and a `composite`
   node runs a bounded internal loop with the outer graph still acyclic;
   retry-exhaustion/timeout/unresolved-input routes to the DLQ (durable,
   re-drivable) and the run ends `completed_with_dead_letters`; a pipeline
   past its `deadline` terminates `timed_out` with in-flight nodes cancelled;
   cyclic / unresolved-input / corrupt / missing-`human`-timeout /
   missing-`deadline` templates are **rejected with a located error** (no
   silent `{}`); the acyclicity + dependency-order invariant model-checks.
4. **Knowledge store rebuild** — new schema + K1–K10 property tests +
   empirical-feedback field.
   *Gate:* each of K1–K10 has a failing→passing test.
5. **Cultivation loop** — wire feedback → sleep (TTT/OPLoRA) + promote gate.
   *Gate:* bit-equivalence before adapter promote; `hypercar_bench --quick`
   gates unchanged; a measured task improves post-sleep with **zero regression**
   on a held-out set.
6. **Subagent swarm + pre-cache** — block-pool, fork/rewind, DAG fan-out/fan-in.
   *Gate:* N concurrent subagents within the M4 Pro 48 GB budget; cold-spawn
   < 2 s via quantized `load`; a fan-out/fan-in pipeline node maps over a list
   and the join aggregates correctly.

---

## Verification strategy

- **Trellis lifts:** run the lifted code against its *own existing tests*
  before modifying (regression baseline), then add H/K tests for each fix.
- **Knowledge system:** property tests, one per defect K1–K10; each must fail
  on the trellis behavior and pass on the rebuild.
- **Pipeline engine:** golden round-trip tests YAML↔IR↔prose; degenerate
  linear-trellis-template import; diamond DAG + conditional branch + fan-out/
  fan-in execution; corrupt/cyclic/unresolved-input templates rejected with a
  located error (regression against trellis's silent-`{}` failure mode);
  agent-authored `compose_pipeline` round-trip; model-check acyclicity +
  dependency-order invariant alongside the existing pool TLA+ spec.
- **Bidirectional human work:** a `human` node round-trips (ask dispatched
  with inputs → typed response → consumed downstream by condition + data
  edge); `timeout` fires `on_timeout` (`escalate`/`fallback-agent`/
  `dead-letter`) and the pipeline does not deadlock; compile rejects a
  `human` node missing `timeout`/`on_timeout`; a human response routes into
  the cultivation loop as a (higher-weighted) reward signal.
- **Termination & DLQ:** retry exhaustion / per-node timeout / unresolved
  input routes the item to the DLQ (durable in the journal, re-drivable),
  pipeline reaches `completed_with_dead_letters` not a hang; a pipeline
  exceeding its `deadline` cancels in-flight nodes and terminates `timed_out`;
  compile rejects a pipeline with no `deadline`; a DLQ item replays
  successfully after a fix; an edge conditioned on a dead-lettered predecessor
  runs the compensation path.
- **hypercar gates:** `.venv/bin/python -m omlx.bench.hypercar_bench --quick`
  must stay green at every phase boundary (CLAUDE.md: non-negotiable). Full
  `--full` before any commit that touches an `omlx/` hot path.
- **Cultivation loop:** bit-equivalence adapter gate (mirror
  `tests/test_ttt_head_router.py`); held-out task set measured pre/post sleep;
  promotion blocked on any held-out regression.
- **End-to-end:** start the Gardener daemon, submit a routine coding task,
  observe (a) Pi worker dispatch, (b) journal entry, (c) knowledge object with
  `empirical.tested=false` → true after a passing test, (d) a sleep cycle
  consolidating only the tested knowledge, (e) `registry` p50/p95 per service.

---

## Open questions for the implementation plan

- ~~Where does the Gardener live?~~ **Resolved:** its own repo at
  `~/Developer/gardener` (created). The earlier `omlx-mamba3/gardener/` design
  doc is the source of truth and will be copied in. **Zero `omlx` dependency**
  — base libraries only; the superpowered-MLX core is ported (Phase 0).
  omlx-mamba3 is a reference map, not a runtime or import boundary, so the
  repo is independent from commit one.
- Pi RPC vs. SDK embedding for the worker adapter — pick after a Pi
  integration spike (Phase 1).
- Codename: "Gardener" is provisional; final name TBD.
