"""Germination state tracking — T3 measurement infrastructure.

Per terra (2026-05-23): T3 = "first 3 calls post-SeedPlanted produce a
non-error InteractionRecorded". This module owns the state machine and
the journal event vocabulary; query/pipeline call sites invoke
`record_call` to feed it.

State transitions:
    pending  --(3 calls, 0 errors)--> passed
    pending  --(3 calls, ≥1 error)--> failed
    pending  --(<3 calls)----------> pending  (still observing)

Per-agent agent.yaml schema (new germination block):

    germination:
      status: pending | passed | failed
      calls_observed: 0..3
      errors: ["call N: <reason>", ...]
      on_fail: alert | regenerate | ignore
      drafted_by: <drafter model id or "template">
      planted_at: ISO-8601 timestamp

The `on_fail` knob is per-agent policy; the *threshold* that aggregates
T3 across agents stays in the global wizard config.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Optional

OBSERVATION_WINDOW = 3   # first N calls determine germination


@dataclass
class GerminationState:
    status: str = "pending"        # pending | passed | failed
    calls_observed: int = 0
    errors: list = field(default_factory=list)
    on_fail: str = "alert"         # alert | regenerate | ignore
    drafted_by: str = "template"   # drafter id, "template", or "user-edited"
    planted_at: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("pending", "passed", "failed"):
            raise ValueError(f"invalid germination status: {self.status!r}")
        if self.on_fail not in ("alert", "regenerate", "ignore"):
            raise ValueError(f"invalid on_fail policy: {self.on_fail!r}")
        if not (0 <= self.calls_observed <= OBSERVATION_WINDOW):
            raise ValueError(
                f"calls_observed must be in [0,{OBSERVATION_WINDOW}]; "
                f"got {self.calls_observed}"
            )

    @classmethod
    def new(
        cls,
        *,
        drafted_by: str = "template",
        on_fail: str = "alert",
    ) -> "GerminationState":
        return cls(
            status="pending",
            calls_observed=0,
            errors=[],
            on_fail=on_fail,
            drafted_by=drafted_by,
            planted_at=dt.datetime.now().isoformat(timespec="seconds"),
        )

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GerminationState":
        if not d:
            return cls.new()
        # Tolerate unknown keys (forward-compat).
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def is_observing(self) -> bool:
        return self.status == "pending"


def record_call(
    state: GerminationState,
    *,
    succeeded: bool,
    error_reason: Optional[str] = None,
) -> tuple[GerminationState, list[dict]]:
    """Advance germination state given a call outcome.

    Returns the new state + a list of journal events to emit. Always emits
    a `GerminationProgress` event for observability; emits `Germinated`
    when the state transitions out of pending.

    Idempotency: calls past the observation window are no-ops; the state
    is returned unchanged and no events fire.
    """
    if not state.is_observing():
        return state, []

    new_calls = state.calls_observed + 1
    new_errors = list(state.errors)
    if not succeeded:
        new_errors.append(
            f"call {new_calls}: {error_reason or 'unspecified failure'}"
        )

    if new_calls >= OBSERVATION_WINDOW:
        new_status = "failed" if new_errors else "passed"
    else:
        new_status = "pending"

    new_state = GerminationState(
        status=new_status,
        calls_observed=new_calls,
        errors=new_errors,
        on_fail=state.on_fail,
        drafted_by=state.drafted_by,
        planted_at=state.planted_at,
    )

    events: list[dict] = [
        {
            "kind": "GerminationProgress",
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "call_index": new_calls,
            "succeeded": succeeded,
            "error_reason": error_reason,
            "status_after": new_status,
        }
    ]
    if new_status != "pending":
        events.append(
            {
                "kind": "Germinated",
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                "status": new_status,
                "calls_observed": new_calls,
                "error_count": len(new_errors),
                "errors": list(new_errors),
                "on_fail": state.on_fail,
            }
        )

    return new_state, events


@dataclass(frozen=True)
class GerminationAction:
    """Describes what the cultivation loop should do in response to a
    germination state transition.

    `policy` is the per-agent `on_fail` value verbatim. `summary` is a
    short human-readable message suitable for stderr or dashboards.
    `journal_event` is the structured event the caller appends to the
    agent's journal under topic `germination` for downstream observability.
    """

    policy: str           # alert | regenerate | ignore
    summary: str
    journal_event: dict


def react_to_germination(
    agent_name: str,
    state: GerminationState,
) -> Optional["GerminationAction"]:
    """Decide what to do when an agent's germination state transitions.

    Returns None when no action is needed (still pending, or transitioned
    to `passed`). Returns a GerminationAction when transitioned to `failed`.

    The caller is responsible for actually carrying out the policy:
      - alert      → print summary to stderr + append the journal event
      - regenerate → append journal event + surface the recommended
                     `gardener wizard --regenerate` command (autonomous
                     regeneration requires re-running the wizard with
                     the original purpose; not yet stored — followup F4.1)
      - ignore     → append the journal event without surfacing
    """
    if state.status != "failed":
        return None

    n_errors = len(state.errors)
    base_event = {
        "kind": "GerminationFailureReaction",
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "agent": agent_name,
        "error_count": n_errors,
        "drafted_by": state.drafted_by,
        "policy": state.on_fail,
    }

    if state.on_fail == "alert":
        return GerminationAction(
            policy="alert",
            summary=(
                f"⚠ germination failed for {agent_name!r}: "
                f"{n_errors} error(s) in {state.calls_observed} call(s) "
                f"(drafted_by={state.drafted_by}). "
                f"Inspect with `gardener observe {agent_name} --topic germination`."
            ),
            journal_event=base_event,
        )
    if state.on_fail == "regenerate":
        return GerminationAction(
            policy="regenerate",
            summary=(
                f"⚠ germination failed for {agent_name!r}; on_fail=regenerate. "
                f"Re-run: `gardener wizard --regenerate {agent_name}` "
                f"(autonomous regeneration not yet implemented; "
                f"run the wizard manually to re-seed)."
            ),
            journal_event={
                **base_event,
                "note": "manual re-seed recommended; autonomous TBD (F4.1)",
            },
        )
    # ignore
    return GerminationAction(
        policy="ignore",
        summary=f"(germination failed for {agent_name!r}; on_fail=ignore, no surface)",
        journal_event=base_event,
    )


def is_response_error(response: str) -> bool:
    """Heuristic: did the agent's response indicate an error?

    Conservative — only an empty response (after strip) counts as a
    germination failure. Richer heuristics (refusal-pattern matching,
    safety-flag scanning) can extend this later without changing the
    state machine.
    """
    return not (response or "").strip()


# ---------------------------------------------------------------------------
# Persistence (separate file from agent.yaml to avoid races with config writes)
# ---------------------------------------------------------------------------

GERMINATION_FILENAME = "germination.yaml"


def load(agent_dir) -> GerminationState:
    """Load germination state from <agent_dir>/germination.yaml.

    Returns a fresh pending state if the file doesn't exist (e.g. agents
    planted before Q2.1).
    """
    from pathlib import Path
    import yaml

    p = Path(agent_dir) / GERMINATION_FILENAME
    if not p.exists():
        return GerminationState.new()
    data = yaml.safe_load(p.read_text()) or {}
    return GerminationState.from_dict(data)


def aggregate(
    agents: dict,
) -> "GerminationAggregate":
    """Aggregate germination state across a registry of agents.

    Input: {name: path} dict (same shape as gardener.cli.registry.load_registry()).
    Returns a snapshot suitable for serialization or threshold checks.

    Agents whose germination.yaml is missing OR can't be parsed are
    counted as `unknown` and listed separately — never silently dropped.
    """
    from pathlib import Path

    per_status: dict[str, int] = {"pending": 0, "passed": 0, "failed": 0}
    per_drafter: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    unknown: list[str] = []
    by_agent: list[dict] = []

    for name, path_str in sorted(agents.items()):
        path = Path(path_str)
        try:
            state = load(path)
        except Exception as e:  # noqa: BLE001 — surface as unknown, don't crash
            unknown.append(f"{name}: {type(e).__name__}: {e}")
            continue

        per_status[state.status] = per_status.get(state.status, 0) + 1
        drafter_bucket = per_drafter.setdefault(
            state.drafted_by, {"pending": 0, "passed": 0, "failed": 0}
        )
        drafter_bucket[state.status] = drafter_bucket.get(state.status, 0) + 1

        if state.status == "failed":
            failures.append(
                {
                    "agent": name,
                    "drafted_by": state.drafted_by,
                    "planted_at": state.planted_at,
                    "on_fail": state.on_fail,
                    "errors": list(state.errors),
                }
            )

        by_agent.append(
            {
                "agent": name,
                "status": state.status,
                "calls_observed": state.calls_observed,
                "drafted_by": state.drafted_by,
                "on_fail": state.on_fail,
                "planted_at": state.planted_at,
            }
        )

    decided = per_status["passed"] + per_status["failed"]
    t3_fail_rate = (per_status["failed"] / decided) if decided else None

    return GerminationAggregate(
        total_agents=len(agents),
        per_status=per_status,
        per_drafter=per_drafter,
        failures=failures,
        unknown=unknown,
        by_agent=by_agent,
        t3_germination_fail_rate=t3_fail_rate,
        decided_count=decided,
    )


@dataclass(frozen=True)
class GerminationAggregate:
    """Snapshot of germination state across all registered agents."""

    total_agents: int
    per_status: dict        # {pending, passed, failed} -> count
    per_drafter: dict       # drafter_id -> {pending, passed, failed} count
    failures: list          # list of {agent, drafted_by, errors, ...}
    unknown: list           # ["name: ParseError: ..."]
    by_agent: list          # full per-agent rows (for table render)
    # None when no agents have decided yet (everyone still pending).
    t3_germination_fail_rate: float | None
    decided_count: int      # number of agents in passed|failed status

    def to_dict(self) -> dict:
        return {
            "total_agents": self.total_agents,
            "decided_count": self.decided_count,
            "t3_germination_fail_rate": self.t3_germination_fail_rate,
            "per_status": dict(self.per_status),
            "per_drafter": {
                k: dict(v) for k, v in self.per_drafter.items()
            },
            "failures": list(self.failures),
            "unknown": list(self.unknown),
            "by_agent": list(self.by_agent),
        }


def save(agent_dir, state: GerminationState) -> None:
    """Write germination state to <agent_dir>/germination.yaml (atomic)."""
    import os
    import tempfile
    from pathlib import Path
    import yaml

    p = Path(agent_dir) / GERMINATION_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".germination.", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(state.to_dict(), f, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
