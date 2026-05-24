"""Drafter Protocol + MockDrafter.

The Drafter Protocol is the boundary between CI-runnable wizard code and
local-only MLX inference. MockDrafter replays fixture-recorded responses
so parser/adversarial/threshold/calibrate code can be exercised with zero
inference cost.

The real MLX-backed drafter (gardener.wizard.drafter_mlx) implements the
same Protocol behind the `model` pytest marker.

Fixture entry shapes (mock_drafter_outputs.yaml):

    - input: "a Python coding agent"
      stratum: narrow
      output: |
        ### System Prompt
        You are a Python expert.

    - input: "ambiguous"
      stratum: broad
      outputs:                      # list cycles per call; wraps when exhausted
        - "### System Prompt\\nVariant 1."
        - "### System Prompt\\nVariant 2."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol, runtime_checkable

import yaml


class DrafterReplayMiss(KeyError):
    """Raised by MockDrafter when an unrecorded input is requested.

    The message points the developer at where to add the missing fixture.
    """


@runtime_checkable
class Drafter(Protocol):
    """Anything that can draft a seed from a plain-language purpose."""

    name: str

    def draft(self, purpose: str, agent_name: str = "agent") -> str:
        """Return raw drafter output (sectioned-markdown text).

        Implementations should NOT parse — parsing is the parser's job and
        T1 measurement requires seeing raw output.
        """
        ...


@dataclass
class _ReplayEntry:
    outputs: list[str]
    stratum: Optional[str]
    sub_type: Optional[str]
    cursor: int = 0

    def next_output(self) -> str:
        out = self.outputs[self.cursor % len(self.outputs)]
        self.cursor += 1
        return out


class MockDrafter:
    """Replay-based drafter for CI tests and calibrate-command tests."""

    name: str

    def __init__(
        self,
        fixtures: Iterable[dict] | dict,
        *,
        name: str = "mock-drafter",
    ) -> None:
        self.name = name
        # Accept dict (loaded from YAML root) or iterable of entries.
        entries: Iterable[dict]
        if isinstance(fixtures, dict):
            entries = fixtures.get("fixtures", [])
        else:
            entries = fixtures
        self._by_input: dict[str, _ReplayEntry] = {}
        for entry in entries:
            inp = entry.get("input")
            if inp is None:
                raise ValueError(f"fixture entry missing 'input': {entry!r}")
            if "output" in entry and "outputs" in entry:
                raise ValueError(
                    f"fixture entry for {inp!r} has both 'output' and 'outputs'; "
                    "pick one"
                )
            if "outputs" in entry:
                outputs = list(entry["outputs"])
            elif "output" in entry:
                outputs = [entry["output"]]
            else:
                raise ValueError(
                    f"fixture entry for {inp!r} missing 'output' or 'outputs'"
                )
            if not outputs:
                raise ValueError(
                    f"fixture entry for {inp!r} has empty outputs list"
                )
            if inp in self._by_input:
                raise ValueError(
                    f"duplicate fixture input: {inp!r}"
                )
            self._by_input[inp] = _ReplayEntry(
                outputs=outputs,
                stratum=entry.get("stratum"),
                sub_type=entry.get("sub_type"),
            )

    @classmethod
    def from_yaml(cls, path: Path | str, *, name: str = "mock-drafter") -> "MockDrafter":
        p = Path(path)
        raw = yaml.safe_load(p.read_text()) or []
        return cls(fixtures=raw, name=name)

    def draft(self, purpose: str, agent_name: str = "agent") -> str:
        entry = self._by_input.get(purpose)
        if entry is None:
            raise DrafterReplayMiss(
                f"no fixture recorded for input {purpose!r}; add an entry to "
                "tests/fixtures/mock_drafter_outputs.yaml (or to the fixture "
                "dict/list passed to MockDrafter())"
            )
        return entry.next_output()

    # Introspection — used by calibrate-command tests and adversarial scorer
    # to know which stratum a given input belongs to without a separate
    # corpus lookup.
    def stratum_of(self, purpose: str) -> Optional[str]:
        entry = self._by_input.get(purpose)
        return entry.stratum if entry else None

    def sub_type_of(self, purpose: str) -> Optional[str]:
        entry = self._by_input.get(purpose)
        return entry.sub_type if entry else None

    def inputs(self) -> list[str]:
        return list(self._by_input)
