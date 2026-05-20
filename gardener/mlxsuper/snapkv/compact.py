"""SnapKV compact_cache: physical token removal with Re-RoPE correction.

Ported (not imported) from omlx/patches/snapkv.py.

Coupling note:
    Re-RoPE reads rope_dims and rope_base from:
        model.layers[0].self_attn.rope.dims
        model.layers[0].self_attn.rope.base
    Defaults to Qwen3-family values (dims=64, base=1_000_000) when the
    attribute path is absent. This is a deliberate mlx_lm coupling — the
    same as the hypercar reference implementation.

    compact_cache also duck-types for a `_logical_positions` attribute
    to detect SparseKVCache-style caches that carry their own compact()
    method, avoiding the gather path.
"""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoPE correction
# ---------------------------------------------------------------------------

def _rerope_keys(
    keys: mx.array,
    old_positions: list[int],
    rope_dims: int,
    rope_base: float = 1_000_000.0,
) -> mx.array:
    """Re-encode RoPE on compacted keys from original to sequential positions.

    After physical compaction, keys have RoPE for their original positions
    but sit at new sequential positions [0, 1, ..., N-1]. Applies a
    per-token rotation shift to correct the encoding.

    RoPE rotations compose additively: rope(rope(x, a), b) = rope(x, a+b).
    To shift from old_pos to new_pos: apply rope(x, new_pos - old_pos).

    Args:
        keys: (B, H, N, D) — keys with RoPE at original positions
        old_positions: list of original position indices (len = N)
        rope_dims: number of dimensions that have RoPE applied
        rope_base: RoPE base frequency (Qwen3-family uses 1_000_000)

    Returns:
        keys with RoPE corrected to sequential positions [0, 1, ..., N-1]
    """
    B, H, N, D = keys.shape
    if N == 0:
        return keys

    new_positions = mx.arange(N)
    old_pos_arr = mx.array(old_positions[:N])
    shifts = new_positions - old_pos_arr    # (N,) mostly negative

    half_d = rope_dims // 2
    freqs = 1.0 / (rope_base ** (
        mx.arange(0, half_d).astype(mx.float32) * 2 / rope_dims
    ))

    angles = shifts[:, None].astype(mx.float32) * freqs[None, :]  # (N, half_d)
    cos_a = mx.cos(angles).astype(keys.dtype)   # (N, half_d)
    sin_a = mx.sin(angles).astype(keys.dtype)

    k1 = keys[:, :, :, :half_d]                 # (B, H, N, half_d)
    k2 = keys[:, :, :, half_d:rope_dims]

    cos_a = cos_a[None, None, :, :]              # (1, 1, N, half_d)
    sin_a = sin_a[None, None, :, :]

    k1_new = k1 * cos_a - k2 * sin_a
    k2_new = k2 * cos_a + k1 * sin_a

    if rope_dims < D:
        result = mx.concatenate([k1_new, k2_new, keys[:, :, :, rope_dims:]], axis=-1)
    else:
        result = mx.concatenate([k1_new, k2_new], axis=-1)

    return result


# ---------------------------------------------------------------------------
# compact_cache
# ---------------------------------------------------------------------------

def compact_cache(
    cache: list,
    keep_indices: list[int],
    *,
    model=None,
    captured_kv: dict | None = None,
    skip_rerope: bool = True,
) -> None:
    """Physical token removal with optional Re-RoPE correction.

    Applies to stock mlx_lm KVCache and QuantizedKVCache. Also handles
    DuoKVCache / TurboQuantKVCache when the caller passes them in `cache`
    (duck-typed on .state attribute).

    For QuantizedKVCache: when `captured_kv` is provided (fp16 K/V captured
    before quantization by Q hooks), uses them to avoid double-quantization
    noise that corrupts at 64K+. Falls back to dequant path otherwise.

    Note: skip_rerope=True (default) skips Re-RoPE for 4.7× faster decode.
    Keys keep original RoPE positions (with gaps). Set False only when
    position-perfect attention is required.

    Args:
        cache: list of KVCache objects (one per layer)
        keep_indices: sorted list of token positions to keep
        model: model object used to extract RoPE config (rope_dims, rope_base)
        captured_kv: dict mapping layer_idx → (Q, K_fp16, V_fp16) or (K_fp16, V_fp16)
            produced by Q capture hooks during prefill
        skip_rerope: skip Re-RoPE position correction (default True)
    """
    if not cache:
        return

    # SparseKVCache duck-type: if the cache carries its own compact() that
    # takes keep_indices, delegate entirely (no gather, no Re-RoPE).
    if hasattr(cache[0], "compact") and hasattr(cache[0], "_logical_positions"):
        for c in cache:
            c.compact(list(keep_indices))
        return

    # Extract RoPE config from model (Qwen3-family defaults)
    rope_dims = 64
    rope_base = 1_000_000.0
    if model is not None:
        try:
            rope = model.layers[0].self_attn.rope
            rope_dims = rope.dims
            rope_base = float(rope.base)
        except AttributeError:
            pass

    idx = mx.array(keep_indices)
    new_len = len(keep_indices)

    for layer_i, c in enumerate(cache):
        keys_raw = c.state[0]
        values_raw = c.state[1]

        if isinstance(keys_raw, (tuple, list)):
            # QuantizedKVCache path
            cap_k: mx.array | None = None
            cap_v: mx.array | None = None
            if captured_kv and layer_i in captured_kv:
                entry = captured_kv[layer_i]
                if isinstance(entry, tuple):
                    if len(entry) == 3:
                        _, cap_k, cap_v = entry   # (Q, K, V) from Q hooks
                    elif len(entry) == 2:
                        cap_k, cap_v = entry       # (K, V) from lightweight hooks

            if cap_k is not None and cap_v is not None:
                keys_fp = cap_k
                values_fp = cap_v
            else:
                keys_fp = mx.dequantize(
                    *keys_raw, group_size=c.group_size, bits=c.bits
                )
                values_fp = mx.dequantize(
                    *values_raw, group_size=c.group_size, bits=c.bits
                )

            keys_compact = keys_fp[:, :, idx, :]
            values_compact = values_fp[:, :, idx, :]

            if not skip_rerope:
                keys_compact = _rerope_keys(keys_compact, keep_indices,
                                            rope_dims, rope_base)

            c.keys = mx.quantize(
                keys_compact, group_size=c.group_size, bits=c.bits
            )
            c.values = mx.quantize(
                values_compact, group_size=c.group_size, bits=c.bits
            )
            c.offset = new_len
        else:
            # KVCache (fp16) or DuoKVCache — direct gather
            keys_compact = keys_raw[:, :, idx, :]
            values_compact = values_raw[:, :, idx, :]

            if not skip_rerope:
                keys_compact = _rerope_keys(keys_compact, keep_indices,
                                            rope_dims, rope_base)

            c.state = (keys_compact, values_compact)
            # Note: offset is set to new_len by state setter; subsequent
            # tokens get RoPE at new_len, new_len+1, etc.

    # Force evaluation of compacted state
    to_eval = []
    for c in cache:
        s = c.state
        if isinstance(s[0], (tuple, list)):
            to_eval.extend(s[0])
            to_eval.extend(s[1])
        else:
            to_eval.append(s[0])
            to_eval.append(s[1])
    mx.eval(*to_eval)
