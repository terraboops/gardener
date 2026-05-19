import pytest
from gardener.pipeline.ir import Pipeline, Node, Edge

def _agent(id_, **kw):
    return Node(id=id_, kind="agent", agent=kw.get("agent", "echo"),
                params=kw.get("params", {}), retry={"max": 1})

def test_linear_pipeline_validates():
    p = Pipeline(
        name="linear", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="a", to_id="b", input_name="prev")],
    )
    p.validate()              # no raise

def test_cycle_rejected_loudly():
    p = Pipeline(
        name="cycle", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="a", to_id="b", input_name="x"),
               Edge(from_id="b", to_id="a", input_name="y")],
    )
    with pytest.raises(ValueError, match="cycle"):
        p.validate()

def test_human_node_requires_timeout_and_on_timeout():
    p = Pipeline(
        name="h", deadline=60.0,
        nodes=[Node(id="h1", kind="human", ask="approve?")],
        edges=[],
    )
    with pytest.raises(ValueError, match="human.*timeout"):
        p.validate()

def test_missing_deadline_rejected():
    p = Pipeline(name="x", deadline=None, nodes=[_agent("a")], edges=[])
    with pytest.raises(ValueError, match="deadline"):
        p.validate()

def test_undefined_input_rejected():
    p = Pipeline(
        name="u", deadline=60.0,
        nodes=[_agent("a"), _agent("b")],
        edges=[Edge(from_id="ghost", to_id="b", input_name="x")],
    )
    with pytest.raises(ValueError, match="ghost"):
        p.validate()
