"""Tests for human-node runners (CLI + scripted)."""
from gardener.pipeline.ir import Node
from gardener.harness.cli_human import ScriptedHumanRunner


def test_scripted_human_returns_queued_response():
    r = ScriptedHumanRunner(responses={"approve": "yes", "name": "Alice"})
    node = Node(id="approve", kind="human", ask="ok?",
                timeout=10.0, on_timeout="dead-letter")
    assert r(node, {}) == "yes"
