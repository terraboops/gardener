import pytest
from gardener.harness.mlx_agent import MLXAgentRunner
from gardener.pipeline.ir import Node

pytestmark = pytest.mark.model


def test_mlx_agent_responds_to_simple_prompt(loaded_model):
    model, tok = loaded_model
    runner = MLXAgentRunner(model, tok)
    node = Node(id="a", kind="agent", agent="answerer",
                params={"system_prompt": "Answer in one word.",
                        "max_tokens": 6})
    out = runner(node, {"question": "What color is the sky on a clear day?"})
    assert isinstance(out, str) and len(out) > 0
