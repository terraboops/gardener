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
