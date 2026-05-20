import pytest
from gardener.pipeline import (parse_pipeline_prose, load_pipeline_yaml,
                                compose_pipeline)

PROSE = """
pipeline demo:
  deadline 30.0

  agent greet:
    prompt: "Greet warmly"
    max_tokens: 16
    retry: 2

  human review:
    ask: "Approve the greeting?"
    timeout: 10.0
    on_timeout: dead-letter
    input greeting from greet
"""

YAML = """
name: demo
deadline: 30.0
nodes:
  - {id: greet, kind: agent, agent: greet, retry: {max: 2, on: fail},
     params: {system_prompt: "Greet warmly", max_tokens: 16}}
  - {id: review, kind: human, ask: "Approve the greeting?",
     timeout: 10.0, on_timeout: dead-letter, retry: {max: 1}}
edges:
  - {from_id: greet, to_id: review, input_name: greeting}
"""


def test_prose_parses_validates_topo_order():
    p = parse_pipeline_prose(PROSE)
    assert p.name == "demo" and p.deadline == 30.0
    assert [n.id for n in p.nodes] == ["greet", "review"]
    assert p.topo_order() == ["greet", "review"]
    g = next(n for n in p.nodes if n.id == "greet")
    assert g.params["system_prompt"] == "Greet warmly"
    assert g.params["max_tokens"] == 16
    assert g.retry == {"max": 2, "on": "fail"}
    r = next(n for n in p.nodes if n.id == "review")
    assert r.kind == "human" and r.timeout == 10.0 and r.on_timeout == "dead-letter"


def test_prose_and_yaml_produce_equivalent_dag():
    a = parse_pipeline_prose(PROSE)
    b = load_pipeline_yaml(YAML)
    assert [n.id for n in a.nodes] == [n.id for n in b.nodes]
    assert a.topo_order() == b.topo_order()
    assert [(e.from_id, e.to_id, e.input_name) for e in a.edges] == \
           [(e.from_id, e.to_id, e.input_name) for e in b.edges]


def test_compose_pipeline_autodetects_prose():
    p = compose_pipeline(PROSE)
    assert p.name == "demo"


def test_compose_pipeline_autodetects_yaml():
    p = compose_pipeline(YAML)
    assert p.name == "demo"


def test_prose_diamond_dag():
    src = """
pipeline diamond:
  deadline 10.0

  agent a:
    prompt: "start"
    retry: 1

  agent b:
    prompt: "left"
    retry: 1
    input x from a

  agent c:
    prompt: "right"
    retry: 1
    input x from a

  agent d:
    prompt: "join"
    retry: 1
    input lb from b
    input rc from c
"""
    p = parse_pipeline_prose(src)
    order = p.topo_order()
    assert order[0] == "a" and order[-1] == "d"
    assert set(order[1:3]) == {"b", "c"}


def test_prose_composite_with_nested_body():
    src = """
pipeline outer:
  deadline 30.0
  composite c:
    retry: 2
    body:
      pipeline inner:
        deadline 10.0
        agent i:
          prompt: "inner work"
          retry: 1
"""
    p = parse_pipeline_prose(src)
    assert len(p.nodes) == 1 and p.nodes[0].kind == "composite"
    assert p.nodes[0].body.nodes[0].id == "i"


def test_prose_corrupt_raises_loudly_not_silent_dict():
    with pytest.raises(ValueError):
        parse_pipeline_prose("pipeline broken: deadline 10\n")   # malformed


def test_compose_pipeline_empty_raises():
    with pytest.raises(ValueError):
        compose_pipeline("# only comments\n# more\n")
