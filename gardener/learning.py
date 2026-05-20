"""Cultivation hook: bridges agent outcomes to (a) knowledge empirical-feedback
flag and (b) TTTEngine.feedback for the eventual sleep-cycle consolidation.

v0 is the wiring + tests; the actual TTT sleep cycle (gated by consolidatable()
+ bit-equivalence promote gate) is deferred to spec Phase 5."""
from __future__ import annotations

from typing import Protocol

from .knowledge import KnowledgeStore


class _FeedbackSink(Protocol):
    def feedback(self, candidate_id: str, reward: float,
                 signal: str = "solution", metadata: dict | None = None) -> None: ...


class CultivationHook:
    def __init__(self, store: KnowledgeStore, ttt: _FeedbackSink):
        self.store = store
        self.ttt = ttt

    def record_success(self, candidate_id: str,
                       knowledge_ids: list[str] | None = None) -> None:
        for kid in knowledge_ids or []:
            self.store.mark_tested(kid, helped=True)
        self.ttt.feedback(candidate_id, 1.0, "solution")

    def record_failure(self, candidate_id: str,
                       knowledge_ids: list[str] | None = None) -> None:
        for kid in knowledge_ids or []:
            self.store.mark_tested(kid, helped=False)
        self.ttt.feedback(candidate_id, -1.0, "solution")
