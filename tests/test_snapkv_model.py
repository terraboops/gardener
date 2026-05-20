"""End-to-end SnapKV pipeline on the test model (Qwen2.5-0.5B-4bit).

These tests require a model download and run the full snapkv_select
pipeline including Q capture hooks and cache compaction.

Run with:
    .venv/bin/python -m pytest tests/test_snapkv_model.py -v -m model
"""
from __future__ import annotations

import pytest
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from gardener.mlxsuper.snapkv import (
    SnapKVOptions,
    snapkv_select,
    compact_cache,
)

pytestmark = pytest.mark.model


def _prefill(model, tokenizer, text: str, cache: list) -> mx.array:
    """Prefill model with text, return token id array."""
    ids = mx.array([tokenizer.encode(text)])
    model(ids, cache=cache)
    mx.eval(ids)
    return ids


# ---------------------------------------------------------------------------
# snapkv_select basic shape contract
# ---------------------------------------------------------------------------

def test_snapkv_select_returns_keep_indices_per_layer(loaded_model):
    """snapkv_select returns keep_indices for each layer, shape ≤ keep_ratio * total."""
    model, tok = loaded_model
    cache = make_prompt_cache(model)

    prompt_text = "A " * 200     # ~200 tokens
    ids = _prefill(model, tok, prompt_text, cache)

    total = cache[0].offset
    options = SnapKVOptions(keep_ratio=0.5, obs_window=32, enable_caote=False,
                             enable_freshness_decay=False, enable_buzz=False)
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)

    assert len(keep_indices) == len(cache), (
        f"Expected one keep_indices per layer ({len(cache)}), got {len(keep_indices)}"
    )
    # All layers share the same keep_indices (aggregated importance)
    # Each has length ≤ keep_ratio * total + small slack (GER widen may add up to 10%)
    max_allowed = int(total * options.keep_ratio * 1.2) + options.always_keep_last
    for i, ki in enumerate(keep_indices):
        assert ki.shape[0] <= max_allowed, (
            f"Layer {i}: {ki.shape[0]} indices > {max_allowed} (total={total})"
        )


def test_snapkv_select_indices_in_range(loaded_model):
    """All keep_indices are valid token positions (< total cached tokens)."""
    model, tok = loaded_model
    cache = make_prompt_cache(model)
    ids = _prefill(model, tok, "Hello world " * 50, cache)
    total = cache[0].offset

    options = SnapKVOptions(keep_ratio=0.5, obs_window=32, enable_caote=False,
                             enable_freshness_decay=False)
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)

    for i, ki in enumerate(keep_indices):
        ki_list = ki.tolist()
        for idx in ki_list:
            assert 0 <= idx < total, (
                f"Layer {i}: index {idx} out of range [0, {total})"
            )
        # No duplicates
        assert len(ki_list) == len(set(ki_list)), f"Layer {i}: duplicate indices"


def test_snapkv_select_with_all_options_enabled(loaded_model):
    """snapkv_select runs without error with all 8 layers enabled."""
    model, tok = loaded_model
    cache = make_prompt_cache(model)
    ids = _prefill(model, tok, "A " * 200, cache)
    total = cache[0].offset

    n_kv_heads = cache[0].state[0].shape[1]
    head_types = ["retrieval"] * (n_kv_heads // 2) + ["streaming"] * (n_kv_heads - n_kv_heads // 2)

    options = SnapKVOptions(
        keep_ratio=0.5,
        obs_window=32,
        enable_caote=True,
        enable_buzz=True,
        buzz_segment_size=32,
        enable_freshness_decay=True,
        enable_ger_safety=True,
        enable_fair_eviction=True,
        enable_head_rebalancing=True,
        head_types=head_types,
    )
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)
    assert len(keep_indices) == len(cache)
    # All layers non-empty
    for ki in keep_indices:
        assert ki.shape[0] > 0


# ---------------------------------------------------------------------------
# compact_cache basic correctness
# ---------------------------------------------------------------------------

def test_compact_cache_reduces_offset(loaded_model):
    """After compact_cache, cache.offset equals len(keep_indices)."""
    model, tok = loaded_model
    cache = make_prompt_cache(model)
    ids = _prefill(model, tok, "A " * 150, cache)
    total = cache[0].offset

    options = SnapKVOptions(keep_ratio=0.5, obs_window=32, enable_caote=False,
                             enable_freshness_decay=False, enable_buzz=False)
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)
    ki_list = keep_indices[0].tolist()
    n_keep = len(ki_list)

    # Compact all layers with the same indices
    compact_cache(cache, ki_list, model=model, skip_rerope=True)

    # After compaction, offset should equal the number of kept tokens
    new_offset = cache[0].offset
    assert new_offset == n_keep, (
        f"Expected offset={n_keep} after compaction, got {new_offset}"
    )
    assert new_offset < total, "Compaction should reduce cache size"


def test_compact_then_decode_still_works(loaded_model):
    """After compact_cache, the model continues to decode without error."""
    model, tok = loaded_model
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    cache = make_prompt_cache(model)
    prompt_text = "A " * 200
    ids = _prefill(model, tok, prompt_text, cache)

    options = SnapKVOptions(keep_ratio=0.5, obs_window=32, enable_caote=False,
                             enable_freshness_decay=False, enable_buzz=False)
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)
    ki_list = keep_indices[0].tolist()

    # Apply compaction to all layers
    compact_cache(cache, ki_list, model=model, skip_rerope=True)

    # Decode a few tokens — should not raise
    out = generate(
        model, tok,
        prompt="Continue:",
        max_tokens=4,
        sampler=make_sampler(temp=0.0),
        prompt_cache=cache,
        verbose=False,
    )
    assert isinstance(out, str), f"Expected str output, got {type(out)}"


def test_compact_with_rerope_doesnt_crash(loaded_model):
    """compact_cache with skip_rerope=False (full Re-RoPE) runs without error."""
    model, tok = loaded_model
    cache = make_prompt_cache(model)
    ids = _prefill(model, tok, "Hello " * 100, cache)
    total = cache[0].offset

    options = SnapKVOptions(keep_ratio=0.5, obs_window=32, enable_caote=False,
                             enable_freshness_decay=False, enable_buzz=False,
                             skip_rerope=False)
    keep_indices = snapkv_select(model, cache, tok, ids, options=options)
    ki_list = keep_indices[0].tolist()

    # Should not raise; Re-RoPE applies positional correction
    compact_cache(cache, ki_list, model=model, skip_rerope=False)
    assert cache[0].offset == len(ki_list)
