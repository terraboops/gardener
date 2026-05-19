import pytest
from gardener.pipeline.yaml_loader import load_pipeline_yaml

YAML = """
name: demo
deadline: 30.0
nodes:
  - id: greet
    kind: agent
    agent: echo
    retry: {max: 2, on: fail}
  - id: review
    kind: human
    ask: "Approve the greeting?"
    timeout: 10.0
    on_timeout: dead-letter
edges:
  - from_id: greet
    to_id: review
    input_name: greeting
"""


def test_loads_validates_roundtrips():
    p = load_pipeline_yaml(YAML)
    assert p.name == "demo"
    assert [n.id for n in p.nodes] == ["greet", "review"]
    assert p.nodes[1].kind == "human"
    assert p.topo_order() == ["greet", "review"]


def test_bad_yaml_rejected_loudly_not_silent_dict():
    with pytest.raises(ValueError):
        load_pipeline_yaml("nodes: [oops")     # malformed
    with pytest.raises(ValueError):
        load_pipeline_yaml("not_a_pipeline: true")  # missing required fields
