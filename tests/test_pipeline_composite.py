"""Test composite node bounded sub-pipeline behavior with retry."""
import pytest
from gardener.journal import EventJournal
from gardener.pipeline.ir import Edge, Node, Pipeline
from gardener.pipeline.executor import PipelineExecutor


def test_composite_bounded_inner_retry_outer_still_acyclic(tmp_path):
    """Composite node with inner-pipeline retry, bounded by outer retry.max.

    Inner pipeline has a node with max=1 retry (fails on attempt 1).
    Outer composite node has max=3 retry (succeeds on attempt 3).
    Agent raises RuntimeError on attempts 1-2, returns 'ok' on attempt 3.

    Expected: outer retry kicks in after inner's 1 retry, succeeds on outer attempt 3.
    """
    attempts = {"n": 0}

    def flaky_then_ok(node, inputs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    # Inner pipeline: single node that will fail on first attempt, retry once
    inner = Pipeline(
        name="inner",
        deadline=10.0,
        nodes=[Node(id="i", kind="agent", agent="flaky", retry={"max": 1})],
        edges=[]
    )
    inner.validate()

    # Outer pipeline: composite node wrapping inner, with its own retry
    outer = Pipeline(
        name="outer",
        deadline=10.0,
        nodes=[Node(id="c", kind="composite", body=inner, retry={"max": 3, "on": "fail"})],
        edges=[]
    )
    outer.validate()

    # Execute: composite node should retry 3 times, succeeding on the 3rd attempt
    res = PipelineExecutor(EventJournal(tmp_path)).run(outer, agent_runner=flaky_then_ok)

    assert res["state"] == "succeeded", f"Pipeline state: {res['state']}"
    assert attempts["n"] == 3, f"Expected 3 attempts, got {attempts['n']}"
    # Outputs structure: composite node "c" returns the inner pipeline's outputs
    assert res["outputs"]["c"]["i"] == "ok", f"Unexpected output: {res['outputs']}"


def test_composite_nested_acyclic(tmp_path):
    """Verify that nested composites remain acyclic (no validation error)."""

    def ok_runner(node, inputs):
        return "ok"

    # Innermost pipeline
    inner = Pipeline(
        name="innermost",
        deadline=10.0,
        nodes=[Node(id="x", kind="agent", agent="ok", retry={"max": 1})],
        edges=[]
    )
    inner.validate()

    # Middle composite
    middle = Pipeline(
        name="middle",
        deadline=10.0,
        nodes=[Node(id="m", kind="composite", body=inner, retry={"max": 1})],
        edges=[]
    )
    middle.validate()

    # Outer composite
    outer = Pipeline(
        name="outer",
        deadline=10.0,
        nodes=[Node(id="o", kind="composite", body=middle, retry={"max": 1})],
        edges=[]
    )
    outer.validate()

    # Should execute without cycle errors
    res = PipelineExecutor(EventJournal(tmp_path)).run(outer, agent_runner=ok_runner)
    assert res["state"] == "succeeded"
    assert res["outputs"]["o"]["m"]["x"] == "ok"
