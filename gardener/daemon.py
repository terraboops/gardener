"""Gardener orchestrator: owns journal + knowledge + pipeline executor."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .journal import EventJournal
from .knowledge import KnowledgeStore
from .pipeline.executor import AgentRunner, HumanRunner, PipelineExecutor
from .pipeline.ir import Pipeline
from .scheduler import Scheduler


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
        self._scheduler: Optional[Scheduler] = None

    def submit(self, pipeline: Pipeline,
               initial_inputs: Optional[dict] = None) -> dict:
        return self.executor.run(pipeline, self.agent_runner,
                                 self.human_runner, initial_inputs)

    def _ensure_scheduler(self, max_concurrent: int = 2) -> Scheduler:
        if self._scheduler is None:
            s = Scheduler(self.executor, max_concurrent=max_concurrent)
            s._agent_runner = self.agent_runner
            s._human_runner = self.human_runner
            s.start()
            self._scheduler = s
        return self._scheduler

    def submit_async(self, pipeline: Pipeline, *, priority: float = 5.0,
                     initial_inputs: Optional[dict] = None,
                     max_concurrent: int = 2):
        """Enqueue on the scheduler; returns a concurrent.futures.Future."""
        sched = self._ensure_scheduler(max_concurrent=max_concurrent)
        return sched.submit(pipeline, priority=priority,
                            initial_inputs=initial_inputs)

    def stop(self) -> None:
        if self._scheduler is not None:
            self._scheduler.stop()
            self._scheduler = None
