"""End-to-end Gardener prototype demo.

Defines a tiny answerer agent backed by a small MLX model, runs it through a
3-node pipeline (seed → agent → scripted human approval), and prints the
outcome, journal trace, and knowledge state. The point is to exercise the
whole prototype path:

  Gardener daemon → pipeline executor → composite runner (echo + MLX)
                  → journal + knowledge + scripted human runner

Run:
    .venv/bin/python -m examples.demo_simple_agent
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mlx_lm import load

from gardener.daemon import Gardener
from gardener.harness.cli_human import ScriptedHumanRunner
from gardener.harness.mlx_agent import MLXAgentRunner
from gardener.knowledge import KnowledgeObject
from gardener.learning import CultivationHook
from gardener.pipeline.yaml_loader import load_pipeline_yaml

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

PIPELINE_YAML = """
name: ask-and-approve
deadline: 120.0
nodes:
  - id: seed
    kind: agent
    agent: echo
    retry: {max: 1}
    params:
      text: "What color is the sky on a clear day, and why?"
  - id: answerer
    kind: agent
    agent: answerer
    retry: {max: 1}
    params:
      system_prompt: "Answer the user's question in one short sentence. If unsure, say 'I don't know.'"
      max_tokens: 40
      temperature: 0.0
  - id: review
    kind: human
    ask: "Approve the answerer's response?"
    timeout: 30.0
    on_timeout: dead-letter
    retry: {max: 1}
edges:
  - {from_id: seed, to_id: answerer, input_name: question}
  - {from_id: answerer, to_id: review, input_name: answer}
"""


def main() -> int:
    print(f"loading {MODEL_ID} …")
    model, tok = load(MODEL_ID)

    mlx_runner = MLXAgentRunner(model, tok)

    def agent_runner(node, inputs):
        if node.agent == "echo":
            return node.params.get("text", "")
        return mlx_runner(node, inputs)

    human = ScriptedHumanRunner(responses={"review": "yes"})

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "garden"
        g = Gardener(root=root, agent_runner=agent_runner,
                     human_runner=human, agent_name="answerer")

        # Seed a piece of knowledge the agent's success should "test".
        kid = g.knowledge.write(KnowledgeObject(
            predicates=[["sky", "appears", "blue"]],
            insight="atmospheric scattering",
            justification="Rayleigh scattering of sunlight",
            source_agent="answerer"))

        pipe = load_pipeline_yaml(PIPELINE_YAML)
        res = g.submit(pipe)

        # v0 cultivation: if pipeline succeeded, mark the knowledge as tested.
        class _Sink:
            def feedback(self, *a, **kw): pass

        hook = CultivationHook(store=g.knowledge, ttt=_Sink())
        if res["state"] == "succeeded":
            hook.record_success(candidate_id="demo-1", knowledge_ids=[kid])

        print("\n=== outcome ===")
        print(f"  state: {res['state']}")
        print(f"  seed:     {res['outputs'].get('seed', '<none>')}")
        print(f"  answer:   {res['outputs'].get('answerer', '<none>')}")
        print(f"  review:   {res['outputs'].get('review', '<none>')}")
        if res["dead_letters"]:
            print(f"  dead-letters: {res['dead_letters']}")
        print("\n=== journal ===")
        for ev in g.journal.read("pipeline"):
            print(f"  {ev}")
        print("\n=== knowledge (after) ===")
        for obj in g.knowledge.all():
            print(f"  {obj.id}: tested={obj.empirical['tested']} "
                  f"helped={obj.empirical['helped']}  {obj.insight!r}")
        return 0 if res["state"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
