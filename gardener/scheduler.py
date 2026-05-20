"""Priority-queue scheduler for Gardener pipelines.

Threadpool-backed. Submit returns a Future. Starvation prevention boosts
long-pending jobs. Cadence triggers re-submit on a 'every Ns' / 'every Nm'
string spec (cron-lite). All public methods are thread-safe."""
from __future__ import annotations

import heapq
import itertools
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .pipeline.executor import PipelineExecutor
from .pipeline.ir import Pipeline


_CADENCE_RE = re.compile(r"^every\s+(\d+(?:\.\d+)?)\s*(s|ms|m|h)$")


def _parse_cadence(spec: str) -> float:
    """'every 30s' -> 30.0; 'every 5m' -> 300.0; 'every 100ms' -> 0.1; etc."""
    m = _CADENCE_RE.match(spec.strip().lower())
    if not m:
        raise ValueError(f"invalid cadence spec: {spec!r} "
                         "(expected 'every <N><s|ms|m|h>')")
    n, unit = float(m.group(1)), m.group(2)
    return n * {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


@dataclass(order=True)
class _Job:
    # Min-heap on (-priority, seq) so highest priority pops first and ties
    # break in FIFO order.
    neg_priority: float
    seq: int
    enqueued_at: float = field(compare=False)
    pipeline: Pipeline = field(compare=False)
    initial_inputs: Optional[dict] = field(compare=False, default=None)
    future: Future = field(compare=False, default_factory=Future)

    @property
    def priority(self) -> float:
        return -self.neg_priority


class Scheduler:
    def __init__(self, executor: PipelineExecutor,
                 *, max_concurrent: int = 2,
                 starvation_boost_after_s: float = 60.0,
                 starvation_boost_amount: float = 1.0,
                 starvation_scan_interval_s: float = 1.0):
        self._exec = executor
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent,
                                        thread_name_prefix="gardener-worker")
        self._semaphore = threading.Semaphore(max_concurrent)
        self._heap: list[_Job] = []
        self._heap_lock = threading.Lock()
        self._cv = threading.Condition(self._heap_lock)
        self._seq = itertools.count()
        self._boost_after = starvation_boost_after_s
        self._boost_amount = starvation_boost_amount
        self._scan_interval = starvation_scan_interval_s
        self._stop = threading.Event()
        self._cadences: list[tuple[str, Callable[[], Pipeline],
                                   float, float]] = []   # (name, factory, interval_s, next_fire)
        self._cadence_lock = threading.Lock()
        self._dispatcher: Optional[threading.Thread] = None
        self._starvation: Optional[threading.Thread] = None
        self._cadence_runner: Optional[threading.Thread] = None
        self._running_inflight = 0
        self._inflight_lock = threading.Lock()

    # --- public API ---------------------------------------------------------

    def submit(self, pipeline: Pipeline, *, priority: float = 5.0,
               initial_inputs: Optional[dict] = None) -> Future:
        job = _Job(neg_priority=-priority, seq=next(self._seq),
                   enqueued_at=time.time(), pipeline=pipeline,
                   initial_inputs=initial_inputs)
        with self._cv:
            heapq.heappush(self._heap, job)
            self._cv.notify()
        return job.future

    def add_cadence(self, name: str, pipeline_factory: Callable[[], Pipeline],
                    cron: str) -> None:
        interval = _parse_cadence(cron)
        with self._cadence_lock:
            self._cadences.append((name, pipeline_factory, interval,
                                   time.time() + interval))

    def start(self) -> None:
        if self._dispatcher is not None:
            return
        self._stop.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="gardener-dispatch", daemon=True)
        self._starvation = threading.Thread(
            target=self._starvation_loop, name="gardener-starv", daemon=True)
        self._cadence_runner = threading.Thread(
            target=self._cadence_loop, name="gardener-cadence", daemon=True)
        self._dispatcher.start()
        self._starvation.start()
        self._cadence_runner.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in (self._dispatcher, self._starvation, self._cadence_runner):
            if t is not None:
                t.join(timeout=timeout)
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._dispatcher = None
        self._starvation = None
        self._cadence_runner = None

    def depth(self) -> int:
        with self._heap_lock:
            return len(self._heap)

    # --- internals ----------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            # Acquire a concurrency slot before popping — keeps the heap intact
            # while all workers are busy, allowing starvation boosts to apply.
            acquired = False
            while not acquired and not self._stop.is_set():
                acquired = self._semaphore.acquire(timeout=0.1)
            if self._stop.is_set():
                if acquired:
                    self._semaphore.release()
                return

            with self._cv:
                while not self._heap and not self._stop.is_set():
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    self._semaphore.release()
                    return
                job = heapq.heappop(self._heap)

            with self._inflight_lock:
                self._running_inflight += 1
            self._pool.submit(self._run_job, job)

    def _run_job(self, job: _Job) -> None:
        try:
            result = self._exec.run(job.pipeline, agent_runner=self._agent_runner,
                                    human_runner=self._human_runner,
                                    initial_inputs=job.initial_inputs)
            job.future.set_result(result)
        except BaseException as e:
            job.future.set_exception(e)
        finally:
            with self._inflight_lock:
                self._running_inflight -= 1
            self._semaphore.release()

    def _starvation_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._scan_interval)
            now = time.time()
            with self._cv:
                touched = False
                for job in self._heap:
                    age = now - job.enqueued_at
                    if age >= self._boost_after:
                        job.neg_priority -= self._boost_amount
                        touched = True
                if touched:
                    heapq.heapify(self._heap)
                    self._cv.notify_all()

    def _cadence_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.05)
            now = time.time()
            fired: list[tuple[Callable[[], Pipeline], int]] = []
            with self._cadence_lock:
                for i, (name, factory, interval, next_fire) in \
                        enumerate(self._cadences):
                    if now >= next_fire:
                        fired.append((factory, i))
            # Build replacement entries (next_fire advanced) and submit.
            for factory, i in fired:
                try:
                    pipe = factory()
                    self.submit(pipe, priority=4.5)   # below default
                except Exception:
                    pass    # cadence pipelines must not crash the scheduler
                with self._cadence_lock:
                    name, fac, interval, _ = self._cadences[i]
                    self._cadences[i] = (name, fac, interval, time.time() + interval)

    # Injected by Gardener:
    _agent_runner = None
    _human_runner = None
