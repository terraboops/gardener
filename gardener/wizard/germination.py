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
