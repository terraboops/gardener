"""Test Gardener daemon orchestrator."""
from gardener.daemon import Gardener
from gardener.pipeline.yaml_loader import load_pipeline_yaml

YAML = """
name: t
deadline: 10.0
nodes:
  - {id: echo, kind: agent, agent: echo, retry: {max: 1}, params: {text: "hi"}}
edges: []
"""

def test_submit_pipeline_succeeds(tmp_path):
    g = Gardener(root=tmp_path,
                 agent_runner=lambda n, i: n.params["text"])
    res = g.submit(load_pipeline_yaml(YAML))
    assert res["state"] == "succeeded"
    # Journal recorded events:
    events = list(g.journal.read("pipeline"))
    assert any(e["event"] == "start" for e in events)
    assert any(e["event"] == "end" for e in events)
