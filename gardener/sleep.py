"""TTT sleep cycle: consolidates tested knowledge into LoRA weights, gated
by a held-out probe (bit-equivalence drift). On regression, rewind."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .knowledge import KnowledgeStore
from .mlxsuper.ttt import Candidate, TTTEngine


@dataclass
class SleepResult:
    state: str               # "no_op" | "promoted" | "rewound"
    trained: int             # number of candidates fed
    drift: float             # held-out mean(|after - before|), fp32; -1.0 if no_op
    elapsed_s: float


class SleepCycle:
    def __init__(self, engine: TTTEngine, store: KnowledgeStore,
                 *, probe_prompts: list[str], max_drift: float = 0.5):
        self.engine = engine
        self.store = store
        self.probe_prompts = probe_prompts
        self.max_drift = max_drift

    def _probe_logits(self) -> list[mx.array]:
        outs = []
        for p in self.probe_prompts:
            ids = mx.array([self.engine.tokenizer.encode(p)])
            outs.append(self.engine.model(ids).astype(mx.float32))
        return outs

    def _drift(self, before: list[mx.array], after: list[mx.array]) -> float:
        d = 0.0
        n = 0
        for b, a in zip(before, after):
            d += float(mx.sum(mx.abs(a - b)).item())
            n += a.size
        return d / max(n, 1)

    def run(self) -> SleepResult:
        t0 = time.perf_counter()
        items = list(self.store.consolidatable())
        if not items:
            return SleepResult(state="no_op", trained=0, drift=-1.0,
                               elapsed_s=time.perf_counter() - t0)

        before = self._probe_logits()
        self.engine.save_checkpoint()

        # Build candidates from knowledge: prompt = predicates as a question,
        # completion = the insight. Reward = +1 (already tested).
        for i, obj in enumerate(items):
            prompt = "Recall: " + "; ".join(
                f"{s} {r} {o}" for s, r, o in obj.predicates) + "\n"
            cid = f"sleep-{i}"
            c = Candidate(id=cid, prompt=prompt, completion=obj.insight,
                          tokens=self.engine.tokenizer.encode(
                              prompt + obj.insight))
            c.reward = 1.0
            c.signal = "solution"
            self.engine.candidates[cid] = c

        try:
            self.engine.train_step(use_oplora=True)
        except Exception:
            # Catastrophic failure → rewind to last checkpoint and surface.
            self.engine.rewind()
            return SleepResult(state="rewound", trained=len(items), drift=-1.0,
                               elapsed_s=time.perf_counter() - t0)

        after = self._probe_logits()
        drift = self._drift(before, after)

        if drift > self.max_drift:
            self.engine.rewind()
            return SleepResult(state="rewound", trained=len(items),
                               drift=drift,
                               elapsed_s=time.perf_counter() - t0)
        return SleepResult(state="promoted", trained=len(items),
                           drift=drift,
                           elapsed_s=time.perf_counter() - t0)
