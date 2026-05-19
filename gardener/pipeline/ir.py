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
    body: Optional[Pipeline] = None
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
