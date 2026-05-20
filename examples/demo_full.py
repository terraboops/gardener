"""Gardener MVP+ extended demo.

Exercises every MVP+ feature end-to-end with a small local model:

  prose DSL  →  compose_pipeline  →  Scheduler  →  PipelineExecutor
                                                       │
                                                       ▼
                                          SubagentRegistry runner
                                                       │
                                                       ▼
                                          Journal + Knowledge + Sleep cycle
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mlx_lm import load

from gardener.daemon import Gardener
from gardener.harness.cli_human import ScriptedHumanRunner
from gardener.knowledge import KnowledgeObject
from gardener.mlxsuper.ttt import TTTEngine
from gardener.pipeline import compose_pipeline
from gardener.scheduler import Scheduler
from gardener.sleep import SleepCycle
from gardener.subagents import AgentProfile, SubagentRegistry

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

PROSE = """
pipeline ask-and-approve:
  deadline 60.0

  agent seed:
    prompt: "Repeat the user's question back verbatim."
    max_tokens: 60

  agent answerer:
    prompt: "Answer in one short sentence. If unsure, say 'I don't know.'"
    max_tokens: 40
    input question from seed

  human review:
    ask: "Approve the answerer's response?"
    timeout: 30.0
    on_timeout: dead-letter
    input answer from answerer
"""


def main() -> int:
    print(f"loading {MODEL_ID} …")
    model, tok = load(MODEL_ID)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "garden"
        warm_root = root / "warm_caches"
        warm_root.mkdir(parents=True, exist_ok=True)

        # 1) Register a pre-warmed subagent profile.
        reg = SubagentRegistry(model, tok, root=warm_root)
        reg.register(AgentProfile(
            name="answerer",
            system_prompt="Answer in one short sentence. If unsure, "
                           "say 'I don't know.'",
            warmup_text="You are a careful answerer. Keep answers short.",
            params={"max_tokens": 40},
        ))
        answerer_runner = reg.runner("answerer")

        # The 'seed' agent just echoes the initial input via a tiny inline runner.
        def agent_runner(node, inputs):
            if node.agent == "seed":
                return inputs.get("starter", "What color is the sky on a clear day, and why?")
            if node.agent == "answerer":
                return answerer_runner(node, inputs)
            raise RuntimeError(f"unknown agent: {node.agent}")

        human = ScriptedHumanRunner(responses={"review": "yes"})

        # 2) Compose the pipeline from prose.
        pipe = compose_pipeline(PROSE)
        print("\n=== prose source ===")
        print(PROSE.strip())
        print(f"\n=== compiled IR ===")
        print(f"  pipeline {pipe.name!r} · deadline={pipe.deadline}s "
              f"· {len(pipe.nodes)} nodes · {len(pipe.edges)} edges")
        for n in pipe.nodes:
            print(f"  - {n.kind:9s} {n.id}")

        # 3) Submit via the Scheduler (P5).
        g = Gardener(root=root, agent_runner=agent_runner,
                     human_runner=human, agent_name="answerer")
        # initial_inputs seeds the 'seed' node's input dict so the question
        # flows through the edges.
        future = g.submit_async(pipe,
                                 priority=7.0,
                                 initial_inputs={"starter": "What color is the sky on a clear day, and why?"})
        res = future.result(timeout=120)

        # 4) Seed knowledge, mark tested if the pipeline succeeded.
        kid = g.knowledge.write(KnowledgeObject(
            predicates=[["sky", "appears", "blue"]],
            insight="The sky appears blue.",
            justification="Rayleigh scattering of sunlight.",
            source_agent="answerer"))
        if res["state"] == "succeeded":
            g.knowledge.mark_tested(kid, helped=True)

        print("\n=== pipeline outcome ===")
        print(f"  state:  {res['state']}")
        print(f"  answer: {res['outputs'].get('answerer', '<none>')}")
        print(f"  review: {res['outputs'].get('review', '<none>')}")
        if res["dead_letters"]:
            print(f"  DLQ:    {res['dead_letters']}")

        print("\n=== journal (pipeline topic) ===")
        for ev in g.journal.read("pipeline"):
            print(f"  {ev}")

        # 5) Sleep cycle.
        eng = TTTEngine(model, tok, rank=4, lr=1e-4, num_lora_layers=1)
        cycle = SleepCycle(eng, g.knowledge,
                            probe_prompts=["The sky is"],
                            max_drift=1e6)   # generous threshold
        sleep_res = cycle.run()
        print("\n=== sleep cycle ===")
        print(f"  state:   {sleep_res.state}")
        print(f"  trained: {sleep_res.trained}")
        print(f"  drift:   {sleep_res.drift:.4g}")
        print(f"  elapsed: {sleep_res.elapsed_s:.2f}s")

        g.stop()
        return 0 if res["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
