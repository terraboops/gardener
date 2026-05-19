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
