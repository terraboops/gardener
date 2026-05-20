"""gardener swarm — run a prose pipeline through the scheduler."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

from ...journal import EventJournal
from ...pipeline import compose_pipeline
from ...pipeline.executor import PipelineExecutor
from ...scheduler import Scheduler
from ..agent_dir import load_agent, load_prompt
from ..registry import resolve, load_registry


class _AgentRunnerFactory:
    """Build (and cache) MLXAgentRunner closures keyed by agent name.

    Each distinct model loads exactly once per swarm run.
    The built-in ``echo`` agent returns its ``text`` param (or empty string)
    without loading any model — useful for seed/test nodes.
    """

    def __init__(self):
        self._models: dict[str, tuple[Any, Any]] = {}  # model_id -> (model, tok)
        self._agents: dict[str, tuple[Any, str]] = {}  # name -> (cfg, prompt)

    def _ensure_loaded(self, agent_name: str) -> None:
        if agent_name in self._agents:
            return
        path = resolve(agent_name)
        cfg = load_agent(path)
        prompt = load_prompt(path, cfg)
        if cfg.model not in self._models:
            print(
                f"[swarm] loading model {cfg.model} for agent {agent_name}",
                file=sys.stderr,
                flush=True,
            )
            try:
                from mlx_lm import load as mlx_load
            except ImportError as e:
                raise RuntimeError(f"mlx_lm not available: {e}") from e
            self._models[cfg.model] = mlx_load(cfg.model)
        self._agents[agent_name] = (cfg, prompt)

    def preload(self, names: list[str]) -> None:
        for n in names:
            self._ensure_loaded(n)

    def dispatch(self, node, inputs: dict) -> str:
        """Execute one pipeline node. Called by the executor."""
        # Built-in echo: no model load required.
        if node.agent == "echo":
            return node.params.get("text", "")

        self._ensure_loaded(node.agent)
        cfg, system_prompt = self._agents[node.agent]
        model, tok = self._models[cfg.model]

        body = "\n".join(f"{k}: {v}" for k, v in inputs.items())
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body or "(no input)"},
        ]
        if hasattr(tok, "apply_chat_template"):
            prompt_text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = body

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise RuntimeError(f"mlx_lm not available: {e}") from e

        sampler = make_sampler(temp=cfg.temperature)
        return generate(
            model,
            tok,
            prompt=prompt_text,
            max_tokens=cfg.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()


def cmd_swarm(args) -> int:
    try:
        pipe_text = Path(args.pipeline).read_text()
    except FileNotFoundError:
        print(f"gardener swarm: pipeline file not found: {args.pipeline}",
              file=sys.stderr)
        return 1

    try:
        pipe = compose_pipeline(pipe_text)
    except ValueError as e:
        print(f"gardener swarm: pipeline parse error: {e}", file=sys.stderr)
        return 1

    factory = _AgentRunnerFactory()

    # Collect all agent names the pipeline references (excluding built-in echo).
    pipeline_agents = sorted(
        {n.agent for n in pipe.nodes
         if n.kind == "agent" and n.agent and n.agent != "echo"}
    )

    # Preload --agent overrides first (cache-warming intent), then pipeline agents.
    if args.agent:
        try:
            factory.preload(args.agent)
        except (KeyError, FileNotFoundError, ValueError) as e:
            print(f"gardener swarm: --agent preload failed: {e}", file=sys.stderr)
            return 1

    try:
        factory.preload(pipeline_agents)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"gardener swarm: agent load failed: {e}", file=sys.stderr)
        return 1

    # Journal root: explicit arg or a fresh temp dir.
    if args.journal_root:
        journal_root = Path(args.journal_root)
    else:
        journal_root = Path(
            tempfile.mkdtemp(
                prefix=f"gardener-swarm-{int(time.time())}-"
            )
        )
    journal_root.mkdir(parents=True, exist_ok=True)
    journal = EventJournal(journal_root)

    initial_inputs = None
    if args.inputs:
        try:
            initial_inputs = json.loads(args.inputs)
        except json.JSONDecodeError as e:
            print(f"gardener swarm: --inputs is not valid JSON: {e}",
                  file=sys.stderr)
            return 1

    executor = PipelineExecutor(journal)
    scheduler = Scheduler(executor, max_concurrent=args.max_concurrent)
    scheduler._agent_runner = factory.dispatch
    scheduler.start()

    try:
        agent_label = ", ".join(pipeline_agents) if pipeline_agents else "(none)"
        print(
            f"[swarm] pipeline: {pipe.name}  agents: {agent_label}",
            flush=True,
        )
        print(f"[swarm] dispatching with priority={args.priority}", flush=True)

        t0 = time.time()
        future = scheduler.submit(
            pipe,
            priority=args.priority,
            initial_inputs=initial_inputs,
        )

        try:
            res = future.result(timeout=args.timeout)
        except FutureTimeout:
            print(
                f"[swarm] FAILED: future timed out after {args.timeout}s",
                file=sys.stderr,
                flush=True,
            )
            return 2
        except Exception as e:
            print(f"[swarm] FAILED: {e}", file=sys.stderr, flush=True)
            return 1

        elapsed = time.time() - t0
        print(f"[swarm] state={res['state']}  elapsed={elapsed:.1f}s", flush=True)

        print("=== outputs ===", flush=True)
        for k, v in res["outputs"].items():
            preview = str(v).replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  {k:<12s} {preview}", flush=True)

        dls = res.get("dead_letters", [])
        print(f"=== dead letters ({len(dls)}) ===", flush=True)
        for dl in dls:
            print(f"  {dl}", flush=True)

        print(f"[swarm] journal: {journal_root}", flush=True)

        return 0 if res["state"] == "succeeded" else 1

    finally:
        scheduler.stop()
