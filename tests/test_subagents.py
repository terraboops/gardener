import pytest

from gardener.subagents import AgentProfile, SubagentRegistry

pytestmark = pytest.mark.model


def test_register_creates_warm_cache(loaded_model, tmp_path):
    model, tok = loaded_model
    reg = SubagentRegistry(model, tok, root=tmp_path)
    p = AgentProfile(
        name="answerer",
        system_prompt="Answer in one short sentence.",
        warmup_text="You are a careful, terse answerer. Keep answers short.",
        params={"max_tokens": 30},
    )
    path = reg.register(p)
    assert path.exists() and path.stat().st_size > 0
    assert reg.has("answerer")


def test_runner_uses_warm_cache_and_responds(loaded_model, tmp_path):
    from gardener.pipeline.ir import Node

    model, tok = loaded_model
    reg = SubagentRegistry(model, tok, root=tmp_path)
    p = AgentProfile(
        name="answerer",
        system_prompt="Answer in one short sentence.",
        warmup_text="You are a careful, terse answerer.",
        params={"max_tokens": 30},
    )
    reg.register(p)
    runner = reg.runner("answerer")
    node = Node(id="a", kind="agent", agent="answerer",
                params={}, retry={"max": 1})
    out = runner(node, {"question": "What is 2 + 2?"})
    assert isinstance(out, str) and len(out) > 0


def test_register_is_idempotent_on_same_warmup(loaded_model, tmp_path):
    model, tok = loaded_model
    reg = SubagentRegistry(model, tok, root=tmp_path)
    p = AgentProfile(name="answerer", system_prompt="x",
                     warmup_text="hello world", params={})
    path1 = reg.register(p)
    mtime1 = path1.stat().st_mtime
    # Second call with identical warmup_text must not regenerate the cache.
    path2 = reg.register(p)
    assert path1 == path2
    # mtime should be unchanged (we skipped the rewrite)
    assert path2.stat().st_mtime == mtime1


def test_unknown_profile_raises(loaded_model, tmp_path):
    model, tok = loaded_model
    reg = SubagentRegistry(model, tok, root=tmp_path)
    with pytest.raises(KeyError):
        reg.runner("nope")
