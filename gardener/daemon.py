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
