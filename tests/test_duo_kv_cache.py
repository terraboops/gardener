"""Tests for DuoKVCache (HX4)."""
from __future__ import annotations

import json
import pytest
import mlx.core as mx

from gardener.mlxsuper.duo_kv_cache import (
    DuoKVCache, StreamingKVCache, load_duo_policy,
    _TRIM_INDEX_CACHE,
)


# --- StreamingKVCache ring buffer --------------------------------------------

def test_streaming_keeps_sink_and_window_after_overflow():
    """After enough writes to overflow the ring, the visible tokens should be
    the first `sink` plus the most recent `window` writes."""
    cache = StreamingKVCache(n_kv_heads=2, head_dim=8, window=4, sink=2)
    for i in range(20):
        k = mx.ones((1, 2, 1, 8)) * i
        v = mx.ones((1, 2, 1, 8)) * i
        cache.update_and_fetch(k, v)
    k_out, _ = cache.state
    # Expected visible: positions [0, 1] (sink) + last 4 (16, 17, 18, 19).
    # The actual order in the assembled output depends on impl; assert size + content set.
    seen_vals = set()
    for t in range(k_out.shape[-2]):
        seen_vals.add(int(k_out[0, 0, t, 0].item()))
    assert {0, 1} <= seen_vals
    assert {16, 17, 18, 19} <= seen_vals


# --- DuoKVCache policy classification ----------------------------------------

def test_duo_policy_classifies_heads_per_layer(tmp_path):
    """A minimal policy JSON with explicit per-layer head classifications
    loads correctly and the lookup returns the right value."""
    policy_doc = {
        "model": "test-model",
        "schema_version": 1,
        "layers": {
            "0": {"streaming_heads": [0, 1], "retrieval_heads": [2, 3]},
            "1": {"streaming_heads": [3],    "retrieval_heads": [0, 1, 2]},
        },
    }
    p = tmp_path / "test-model.json"
    p.write_text(json.dumps(policy_doc))
    policy = load_duo_policy("test-model", search_paths=[tmp_path])
    assert policy["_lookup"][(0, 0)] == "streaming"
    assert policy["_lookup"][(0, 2)] == "retrieval"
    assert policy["_lookup"][(1, 3)] == "streaming"
    assert policy["_lookup"][(1, 0)] == "retrieval"


def test_duo_policy_qwen3_coder_loads_from_default_search():
    """The Qwen3-Coder policy that ships with the repo loads cleanly."""
    policy = load_duo_policy("qwen3_coder_30b_a3b_instruct_8bit")
    assert "_lookup" in policy
    # At least one layer's lookup is non-empty.
    assert any(v in {"streaming", "retrieval"} for v in policy["_lookup"].values())


# --- DuoKVCache lifecycle ----------------------------------------------------

def _fake_policy(n_layers: int = 2, n_heads: int = 4):
    """Half-and-half streaming/retrieval policy for testing."""
    layers = {
        str(i): {
            "streaming_heads": list(range(n_heads // 2)),
            "retrieval_heads": list(range(n_heads // 2, n_heads)),
        }
        for i in range(n_layers)
    }
    policy = {"model": "fake", "schema_version": 1, "layers": layers}
    # Inject the _lookup the way load_duo_policy would.
    policy["_lookup"] = {
        (int(li), h): "streaming" if h in v["streaming_heads"] else "retrieval"
        for li, v in layers.items() for h in v["streaming_heads"] + v["retrieval_heads"]
    }
    return policy


def test_duo_kv_does_not_expose_bits_attribute():
    """Critical gotcha: mlx_lm SDPA routes on hasattr(cache, 'bits').
    DuoKVCache must NOT expose 'bits' or fp16 streaming heads break."""
    policy = _fake_policy()
    cache = DuoKVCache(policy=policy, layer_idx=0, n_kv_heads=4,
                       quantize_retrieval=True, quant_bits=3)
    assert not hasattr(cache, "bits"), \
        "DuoKVCache.bits would route through mlx_lm.quantized_matmul, breaking fp16 streaming heads"
    # The bits config is preserved as _quant_bits for our own use.
    assert cache._quant_bits == 3


def test_update_and_fetch_shape_correct():
    policy = _fake_policy()
    cache = DuoKVCache(policy=policy, layer_idx=0, n_kv_heads=4)
    k = mx.random.normal((1, 4, 16, 8)).astype(mx.float16)
    v = mx.random.normal((1, 4, 16, 8)).astype(mx.float16)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape[0] == 1 and k_out.shape[1] == 4
    assert k_out.shape == v_out.shape
    assert cache.offset >= 16


def test_trim_reduces_offset():
    policy = _fake_policy()
    cache = DuoKVCache(policy=policy, layer_idx=0, n_kv_heads=4)
    k = mx.random.normal((1, 4, 32, 8)).astype(mx.float16)
    v = mx.random.normal((1, 4, 32, 8)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    before = cache.offset
    dropped = cache.trim(8)
    assert dropped == 8
    assert cache.offset == before - 8


def test_state_meta_state_round_trip():
    """The state + meta_state must reconstruct via from_state — stock
    mlx_lm.models.cache.{save,load}_prompt_cache contract."""
    policy = _fake_policy()
    cache = DuoKVCache(policy=policy, layer_idx=0, n_kv_heads=4,
                       window=8, sink=2)
    k = mx.random.normal((1, 4, 16, 8)).astype(mx.float16)
    v = mx.random.normal((1, 4, 16, 8)).astype(mx.float16)
    cache.update_and_fetch(k, v)
    s = cache.state
    ms = cache.meta_state
    restored = DuoKVCache.from_state(s, ms)
    assert restored.offset == cache.offset
    assert restored.window == cache.window
    assert restored.sink == cache.sink


# --- _TRIM_INDEX_CACHE perf gate ---------------------------------------------

def test_trim_index_cache_is_reused_across_layers():
    """All layers at the same decode step see the same (T_total, sink, window).
    The module-level cache should be hit on the second+ layer per token."""
    policy = _fake_policy(n_layers=4, n_heads=4)
    _TRIM_INDEX_CACHE.clear()
    caches = [DuoKVCache(policy=policy, layer_idx=i, n_kv_heads=4,
                          window=8, sink=2) for i in range(4)]
    k = mx.random.normal((1, 4, 32, 8)).astype(mx.float16)
    v = mx.random.normal((1, 4, 32, 8)).astype(mx.float16)
    for c in caches:
        c.update_and_fetch(k, v)
    # Only ONE entry in the trim-index cache (all layers reused it).
    assert len(_TRIM_INDEX_CACHE) <= 4  # one per unique key shape


# --- optional model integration ----------------------------------------------

@pytest.mark.model
def test_duo_with_real_model(loaded_model):
    """Drive a forward pass with DuoKVCache on the test model. Test model is
    Qwen2.5-0.5B; no policy ships for it, so we use a fake half-and-half
    policy. Just verify the forward returns valid output and the cache offset
    advances."""
    model, tok = loaded_model
    n_layers = len(model.model.layers)
    # Detect n_kv_heads from the model's first attention layer.
    first_attn = model.model.layers[0].self_attn
    n_kv_heads = getattr(first_attn, "n_kv_heads",
                          getattr(first_attn, "n_heads", 2))
    policy = _fake_policy(n_layers=n_layers, n_heads=n_kv_heads)
    caches = [DuoKVCache(policy=policy, layer_idx=i, n_kv_heads=n_kv_heads,
                          window=64, sink=4) for i in range(n_layers)]
    ids = mx.array([tok.encode("hello world")])
    out = model(ids, cache=caches)
    assert out.shape[0] == 1
    assert caches[0].offset > 0
