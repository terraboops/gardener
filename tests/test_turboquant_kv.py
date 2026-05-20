"""Tests for TurboQuantKVCache (HX3).

Pure-logic tests cover the codec round-trip + agentic API (fork/rewind/
save/load). The model integration test confirms it interoperates with
stock mlx_lm.models.cache helpers.
"""
from __future__ import annotations

import pytest
import mlx.core as mx

from gardener.mlxsuper.turboquant_kv import (
    TurboQuantKVCache,
    _wht,
    _apply_wht_rotation,
    _codebook,
)


# --- WHT + codebook math ------------------------------------------------------


def test_wht_is_its_own_inverse_up_to_scale():
    """WHT is its own inverse when normalized — applying twice returns the original.

    The implementation normalizes by 1/sqrt(D) each pass so:
       (1/sqrt(D) * H) @ (1/sqrt(D) * H) = (1/D) * H^2 = (1/D) * D*I = I
    i.e. normalized WHT applied twice is the identity, not a scale.
    """
    mx.random.seed(0)
    x = mx.random.normal((16,))
    y = _wht(_wht(x))
    rel = float(mx.max(mx.abs(y - x)))
    assert rel < 1e-4


def test_apply_wht_rotation_preserves_magnitude():
    mx.random.seed(0)
    x = mx.random.normal((8, 16))
    y = _apply_wht_rotation(x, seed=42)
    # Rotation is norm-preserving (up to numeric).
    nx = float(mx.sqrt(mx.sum(x ** 2)).item())
    ny = float(mx.sqrt(mx.sum(y ** 2)).item())
    assert abs(nx - ny) / nx < 1e-3


def test_codebook_is_cached_per_dim_bits():
    a = _codebook(dim=16, bits=3)
    b = _codebook(dim=16, bits=3)
    # lru_cache: should be the same exact array object (not just equal).
    assert a is b


# --- cache class lifecycle ----------------------------------------------------


def _make_kv(B=1, H=2, T=128, D=16, dtype=mx.float16):
    mx.random.seed(1)
    k = mx.random.normal((B, H, T, D)).astype(dtype)
    v = mx.random.normal((B, H, T, D)).astype(dtype)
    return k, v


def test_update_and_fetch_round_trip_returns_correct_shape():
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=64, D=16)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert cache.offset == 64


def test_quantization_reduces_storage_significantly():
    """3-bit quantized state must be smaller than fp16 input."""
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=256, D=16)
    cache.update_and_fetch(k, v)
    fp16_bytes = k.size * 2 * 2  # k+v in fp16
    # Inspect state tree size — quantized state should be much smaller.
    state = cache.state
    total_bytes = sum(a.nbytes for a in _flatten_arrays(state))
    assert total_bytes < fp16_bytes * 0.5


def _flatten_arrays(tree):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _flatten_arrays(v)
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _flatten_arrays(v)


# --- agentic API: fork / rewind / save / load --------------------------------


def test_fork_is_independent_branch():
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=64, D=16)
    cache.update_and_fetch(k, v)
    branch = cache.fork()
    # Append to branch; original offset should be unchanged.
    k2, v2 = _make_kv(T=8, D=16)
    branch.update_and_fetch(k2, v2)
    assert cache.offset == 64
    assert branch.offset == 72


def test_rewind_to_drops_tail():
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=64, D=16)
    cache.update_and_fetch(k, v)
    cache.rewind_to(target_offset=32)
    assert cache.offset == 32


def test_save_load_round_trip(tmp_path):
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=64, D=16)
    cache.update_and_fetch(k, v)
    meta = cache.save_to_disk(str(tmp_path / "c.npz"))
    assert meta["tokens"] == 64
    assert meta["bits"] == 3
    cache2 = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    n_loaded = cache2.load_from_disk(str(tmp_path / "c.npz"))
    assert n_loaded == 64
    assert cache2.offset == 64


# --- stock mlx_lm interop -----------------------------------------------------


def test_state_meta_state_round_trip():
    """state + meta_state must be reconstructible via from_state — this is the
    contract mlx_lm.models.cache.{save,load}_prompt_cache depends on."""
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=32, D=16)
    cache.update_and_fetch(k, v)
    s = cache.state
    ms = cache.meta_state
    restored = TurboQuantKVCache.from_state(s, ms)
    assert restored.offset == cache.offset
    assert restored.bits == cache.bits


def test_trim_returns_actual_dropped_count():
    cache = TurboQuantKVCache(bits=3, group_size=16, min_quant_tokens=0)
    k, v = _make_kv(T=64, D=16)
    cache.update_and_fetch(k, v)
    dropped = cache.trim(20)
    assert dropped == 20
    assert cache.offset == 44


# --- optional model test ------------------------------------------------------


@pytest.mark.model
@pytest.mark.xfail(
    reason=(
        "mlx / mlx_lm version skew: mlx_lm/models/base.py:84 calls "
        "mx.quantized_matmul(queries, *q_keys, transpose=..., group_size=..., bits=...) "
        "but the installed mlx (0.31.2) requires scales+biases as positional args "
        "after w. mlx_lm 0.31.3 was built against a different mlx signature. This "
        "trips for TurboQuantKVCache because it (correctly) exposes `bits` so "
        "mlx_lm's SDPA routes through the quantized path — and that path is "
        "currently broken upstream. The codec itself is correct (10 pure-logic "
        "tests pass); fix requires either a compatible mlx version or an mlx_lm "
        "patch. DuoKVCache sidesteps this by deliberately not exposing `bits`."
    ),
    strict=False,
)
def test_tq3_cache_works_with_real_model(loaded_model):
    """A TurboQuantKVCache populated with one layer's worth of real K/V from
    the test model produces shape-correct quantized state and can be
    save/loaded via mlx_lm prompt-cache helpers."""
    from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
    import tempfile
    import os

    model, tok = loaded_model
    # Build a TQ3 cache list (one per layer).
    n_layers = len(model.model.layers)
    caches = [
        TurboQuantKVCache(bits=3, group_size=64, min_quant_tokens=0)
        for _ in range(n_layers)
    ]
    # Drive a forward pass to populate the caches.
    ids = mx.array([tok.encode("hello world")])
    model(ids, cache=caches)
    assert caches[0].offset > 0
    # Save + load via stock helpers.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tq3.safetensors")
        save_prompt_cache(path, caches, {})
        restored = load_prompt_cache(path)
        assert len(restored) == n_layers
        assert restored[0].offset == caches[0].offset
