import pytest
from gardener.journal import EventJournal
from gardener.pipeline.ir import Edge, Node, Pipeline
from gardener.pipeline.executor import PipelineExecutor

def _exec(tmp_path):
    return PipelineExecutor(journal=EventJournal(tmp_path))

def test_linear_runs_outputs_chain(tmp_path):
    p = Pipeline(name="lin", deadline=10.0,
        nodes=[Node(id="a", kind="agent", agent="echo",
                    params={"text": "hi"}, retry={"max": 1}),
               Node(id="b", kind="agent", agent="upper",
                    params={}, retry={"max": 1})],
        edges=[Edge(from_id="a", to_id="b", input_name="msg")])
    p.validate()
    runners = {
        "echo":  lambda inputs, params: params["text"],
        "upper": lambda inputs, params: inputs["msg"].upper(),
    }
    res = _exec(tmp_path).run(p, agent_runner=lambda n, ins: runners[n.agent](ins, n.params))
    assert res["state"] == "succeeded"
    assert res["outputs"]["b"] == "HI"

def test_retry_then_dlq_on_exhaustion(tmp_path):
    calls = []
    def flaky(node, inputs):
        calls.append(node.id); raise RuntimeError("nope")
    p = Pipeline(name="r", deadline=10.0,
        nodes=[Node(id="x", kind="agent", agent="flaky",
                    retry={"max": 3, "on": "fail"})],
        edges=[])
    p.validate()
    res = _exec(tmp_path).run(p, agent_runner=flaky)
    assert calls == ["x", "x", "x"]
    assert res["state"] == "completed_with_dead_letters"
    assert res["dead_letters"][0]["node"] == "x"

def test_deadline_terminates_timed_out(tmp_path):
    import time
    def slow(node, inputs):
        time.sleep(0.2); return "ok"
    p = Pipeline(name="d", deadline=0.05,
        nodes=[Node(id="s1", kind="agent", agent="slow", retry={"max": 1}),
               Node(id="s2", kind="agent", agent="slow", retry={"max": 1})],
        edges=[Edge(from_id="s1", to_id="s2", input_name="x")])
    p.validate()
    res = _exec(tmp_path).run(p, agent_runner=slow)
    assert res["state"] == "timed_out"

def test_human_node_dispatches_via_runner(tmp_path):
    p = Pipeline(name="h", deadline=10.0,
        nodes=[Node(id="ask", kind="human", ask="ok?",
                    timeout=5.0, on_timeout="dead-letter",
                    retry={"max": 1})],
        edges=[])
    p.validate()
    res = _exec(tmp_path).run(p,
        agent_runner=lambda n, i: None,
        human_runner=lambda n, i: "yes")
    assert res["outputs"]["ask"] == "yes" and res["state"] == "succeeded"
