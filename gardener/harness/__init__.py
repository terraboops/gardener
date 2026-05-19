"""Harness adapters (agent + human runners)."""
from typing import Callable, Protocol
from ..pipeline.ir import Node


class AgentRunner(Protocol):
    def __call__(self, node: Node, inputs: dict) -> object: ...


class HumanRunner(Protocol):
    def __call__(self, node: Node, inputs: dict) -> object: ...
