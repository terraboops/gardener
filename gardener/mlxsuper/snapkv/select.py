"""SnapKV select: real-Q attention importance scoring + top-K selection.

Ported (not imported) from omlx/patches/snapkv.py.

Coupling note:
    Real-Q capture hooks monkey-patch model.layers[i] to install a subclass
    whose __call__ records query projections during the forward pass. The
    hook replicates the Qwen3/Qwen2.5 attention layer's internal structure
    (q_proj, k_proj, v_proj, q_norm, k_norm, rope, o_proj, mlp, layernorms).

    The hook is fragile to mlx_lm model changes. The clean-up restores the
    original class by walking MRO — this relies on the assumption that the
    hooked class is a direct subclass of the original (true for our dynamic
    type(..) approach). Document this as a known brittleness point.

    Supported model families: Qwen2.5 / Qwen3 (both MoE and dense).
    Other families (Llama, Mistral, etc.) will fail silently in the hook if
    the attribute names differ — the fallback is K-as-Q proxy importance.
"""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q capture hook
# ---------------------------------------------------------------------------

def install_q_capture_hook(
    model,
    target_layers: list[int] | None = None,
) -> tuple[dict, callable]:
    """Install hooks on decoder layers to capture Q, K, V after RoPE.

    Monkey-patches each target layer with a dynamically-created subclass
    whose __call__ mirrors the Qwen3/Qwen2.5 attention+MLP forward path,
    capturing (Q, K_fp16, V_fp16) before cache.update_and_fetch quantizes them.

    Args:
        model: loaded model with .layers[i].self_attn
        target_layers: layer indices to hook (default: last 4)

    Returns:
        captured: dict mapping layer_idx → (Q, K_fp16, V_fp16) arrays
            Built up incrementally during the forward pass.
        cleanup: callable that removes all hooks and restores original types.
    """
    n_layers = len(model.layers)
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 4), n_layers))

    captured: dict[int, tuple[mx.array, mx.array, mx.array]] = {}

    def _make_layer_hook(lid: int):
        def hooked_layer(self, x, mask=None, cache=None):
            # Replicate the TransformerBlock forward, capturing Q/K/V.
            # Supports both Qwen3 (has q_norm/k_norm) and Qwen2 (no norms).
            normed = self.input_layernorm(x)
            B, L, D = normed.shape
            sa = self.self_attn

            queries = sa.q_proj(normed)
            keys = sa.k_proj(normed)
            values = sa.v_proj(normed)

            # Reshape to (B, H, L, head_dim)
            queries = queries.reshape(B, L, sa.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, sa.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, sa.n_kv_heads, -1).transpose(0, 2, 1, 3)

            # Qwen3 adds QK norms; Qwen2 does not — check gracefully
            if hasattr(sa, "q_norm"):
                # q_norm/k_norm operate on (..., head_dim) — transpose to (B, L, H, D)
                queries = sa.q_norm(queries.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            if hasattr(sa, "k_norm"):
                keys = sa.k_norm(keys.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)

            if cache is not None:
                queries_rope = sa.rope(queries, offset=cache.offset)
                keys_rope = sa.rope(keys, offset=cache.offset)
                # Capture fp16 Q/K/V BEFORE update_and_fetch may quantize
                if lid not in captured:
                    captured[lid] = (queries_rope, keys_rope, values)
                else:
                    prev_q, prev_k, prev_v = captured[lid]
                    captured[lid] = (
                        mx.concatenate([prev_q, queries_rope], axis=2),
                        mx.concatenate([prev_k, keys_rope], axis=2),
                        mx.concatenate([prev_v, values], axis=2),
                    )
                keys_ret, values_ret = cache.update_and_fetch(keys_rope, values)
                keys_for_attn = keys_ret
                values_for_attn = values_ret
            else:
                queries_rope = sa.rope(queries)
                keys_rope = sa.rope(keys)
                captured[lid] = (queries_rope, keys_rope, values)
                keys_for_attn = keys_rope
                values_for_attn = values

            from mlx_lm.models.base import scaled_dot_product_attention
            attn_out = scaled_dot_product_attention(
                queries_rope, keys_for_attn, values_for_attn,
                cache=cache, scale=sa.scale, mask=mask,
            )
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            h = x + sa.o_proj(attn_out)
            r = self.mlp(self.post_attention_layernorm(h))
            return h + r

        return hooked_layer

    for layer_idx in target_layers:
        layer = model.layers[layer_idx]
        hooked_fn = _make_layer_hook(layer_idx)
        orig_cls = layer.__class__

        # Create a dynamic subclass with the hooked __call__
        new_cls = type(
            f"_SnapKVHooked_{orig_cls.__name__}_{layer_idx}",
            (orig_cls,),
            {"__call__": lambda self, *a, _fn=hooked_fn, **kw: _fn(self, *a, **kw)},
        )
        layer.__class__ = new_cls

    def cleanup():
        for layer_idx in target_layers:
            layer = model.layers[layer_idx]
            # Restore original class by walking MRO past the hooked subclass
            orig_cls = layer.__class__.__mro__[1]
            layer.__class__ = orig_cls

    return captured, cleanup


# ---------------------------------------------------------------------------
# Importance computation from captured Q
# ---------------------------------------------------------------------------

def compute_importance_from_real_q(
    captured: dict,
    cache: list,
    obs_window: int = 64,
) -> mx.array:
    """Compute per-token importance from real Q projections captured by hooks.

    Uses fp16 K from the capture hook to avoid quantization noise in
    native 3-bit mode. Falls back to dequantized K when not captured.

    Args:
        captured: dict from install_q_capture_hook
            (layer_idx → (Q, K_fp16, V_fp16) or bare Q)
        cache: KVCache list (fallback for K values)
        obs_window: number of trailing query positions to use

    Returns:
        importance: (B, H_kv, T) — aggregated max importance across layers
    """
    from .scoring import _get_fp16_keys  # keep circular-import safe

    all_importance = []

    for layer_idx, entry in captured.items():
        # Unpack captured entry
        if isinstance(entry, tuple):
            queries = entry[0]
            keys = entry[1] if len(entry) > 1 else _get_fp16_keys(cache[layer_idx])
        else:
            queries = entry
            keys = _get_fp16_keys(cache[layer_idx])

        B, H_kv, T, D = keys.shape
        H_q = queries.shape[1]
        gqa_ratio = H_q // H_kv
        scale = D ** -0.5

        obs_start = max(0, queries.shape[2] - obs_window)
        Q_obs = queries[:, :, obs_start:, :]        # (B, H_q, obs_len, D)
        obs_len = Q_obs.shape[2]

        # GQA-aware: reshape to (B, H_kv, gqa_ratio, obs_len, D)
        Q_grouped = Q_obs.reshape(B, H_kv, gqa_ratio, obs_len, D)

        # Causal mask
        q_pos = mx.arange(obs_start, obs_start + obs_len).reshape(1, 1, obs_len, 1)
        k_pos = mx.arange(T).reshape(1, 1, 1, T)
        causal_mask = k_pos <= q_pos

        # Scores: (B, H_kv, gqa_ratio, obs_len, T)
        scores = (Q_grouped @ keys[:, :, None, :, :].swapaxes(-1, -2)) * scale
        scores = mx.where(
            causal_mask[:, :, None, :, :], scores, mx.array(float("-inf"))
        )
        weights = mx.softmax(scores, axis=-1)

        # Pool: max over obs window and GQA group
        max_weights = mx.max(weights, axis=3)   # (B, H_kv, gqa_ratio, T)
        importance = mx.max(max_weights, axis=2)  # (B, H_kv, T)
        mx.eval(importance)
        all_importance.append(importance)

    stacked = mx.stack(all_importance, axis=0)
    result = mx.max(stacked, axis=0)
    mx.eval(result)
    return result


# ---------------------------------------------------------------------------
# Top-K selection helpers
# ---------------------------------------------------------------------------

def _select_global(pooled: mx.array, k: int) -> set[int]:
    """Global top-K selection (original SnapKV behavior)."""
    top_k = mx.argpartition(-pooled[0], kth=k)[:k]
    mx.eval(top_k)
    return set(top_k.tolist())
