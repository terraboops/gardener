"""SnapKV scoring helpers: CAOTE, BUZZ, freshness decay.

Ported (not imported) from omlx/patches/snapkv.py.
Uses the ORIGINAL loop variants — NOT the Task 140/147/168 vectorizations
which regressed in hypercar Run 84 (needle lost at 49% keep).

Functions here are pure score-modifier passes on a (B, H_kv, T) importance
tensor produced by select.py's real-Q attention computation.
"""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for extracting fp16 K/V from various cache types
# ---------------------------------------------------------------------------

def _get_fp16_keys(cache_entry) -> mx.array:
    """Extract fp16 key tensor from any stock mlx_lm cache type.

    Handles KVCache (raw arrays) and QuantizedKVCache (dequantize).

    Returns:
        keys: (B, H_kv, T, D) fp16/bfloat16 array
    """
    state = cache_entry.state
    keys_raw = state[0]
    if isinstance(keys_raw, tuple):
        # QuantizedKVCache — state[0] is (data, scales, biases)
        return mx.dequantize(
            *keys_raw,
            group_size=cache_entry.group_size,
            bits=cache_entry.bits,
        )
    return keys_raw


def _get_fp16_values(cache_entry) -> mx.array:
    """Extract fp16 value tensor from any stock mlx_lm cache type."""
    state = cache_entry.state
    values_raw = state[1]
    if isinstance(values_raw, tuple):
        return mx.dequantize(
            *values_raw,
            group_size=cache_entry.group_size,
            bits=cache_entry.bits,
        )
    return values_raw


# ---------------------------------------------------------------------------
# CAOTE scoring (Task 100) — value-aware importance
# ---------------------------------------------------------------------------

def caote_score(
    importance_attn: mx.array,
    cache_entry,
    obs_window: int = 64,
) -> mx.array:
    """Blend attention importance with value distinctiveness (CAOTE).

    From arXiv:2504.14051 Theorem 3.2: the MSE cost of evicting token j
    equals (alpha_j / (1 - alpha_j)) * ||V_mean - v_j||_2.

    FastCAOTE approximation: uses unweighted mean(V) instead of the
    weighted attention-mean. O(n*d) per head, not O(n^2*d).

    Args:
        importance_attn: (B, H_kv, T) — attention-only importance (alpha)
        cache_entry: single layer's KVCache (or QuantizedKVCache)
        obs_window: number of trailing queries used for importance (used
            only to align obs-window pooling with the attention scores)

    Returns:
        caote: (B, H_kv, T) — CAOTE eviction cost (higher = more important)
    """
    values = _get_fp16_values(cache_entry)  # (B, H_kv, T, D)

    # Pool attention weights as alpha per token
    alpha = importance_attn  # (B, H_kv, T)

    # Value distinctiveness: ||V_mean - v_j||_2
    V_mean = mx.mean(values, axis=2, keepdims=True)  # (B, H_kv, 1, D)
    v_diff = values - V_mean                          # (B, H_kv, T, D)
    v_dist = mx.sqrt(mx.sum(v_diff * v_diff, axis=-1) + 1e-8)  # (B, H_kv, T)

    # CAOTE: (alpha / (1 - alpha)) * ||V_mean - v_j||
    alpha_clamped = mx.clip(alpha, 1e-6, 1.0 - 1e-6)
    caote = (alpha_clamped / (1.0 - alpha_clamped)) * v_dist
    mx.eval(caote)
    return caote


# ---------------------------------------------------------------------------
# BUZZ segmented selection (Task 98) — original loop variant
# ---------------------------------------------------------------------------

def buzz_segment_select(pooled: mx.array, k: int, segment_size: int) -> set[int]:
    """BUZZ-style per-segment top-K selection.

    Divides the selectable range into segments of `segment_size` tokens.
    Within each segment, selects the top-k_local tokens proportional to
    the segment's share of the total budget.

    This is the ORIGINAL LOOP VARIANT from Task 98 (not Task 147's
    vectorized scatter which regressed in Run 84).

    From arXiv:2410.23079: segmented selection outperforms global H2O
    by 7.69% on multi-document QA at 2.5x cache reduction.

    Args:
        pooled: (B, S) pooled importance scores (B=1 typical)
        k: total tokens to select from selectable range
        segment_size: tokens per BUZZ segment

    Returns:
        set of local indices into `pooled` that are selected
    """
    B, S = pooled.shape
    n_segments = max(1, (S + segment_size - 1) // segment_size)

    base_k = max(1, k // n_segments)
    remainder = k - base_k * n_segments

    all_indices: set[int] = set()

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, S)
        seg_len = seg_end - seg_start

        seg_k = base_k + (1 if seg_idx < remainder else 0)
        seg_k = min(seg_k, seg_len)

        if seg_k <= 0:
            continue

        seg_scores = pooled[0, seg_start:seg_end]           # (seg_len,)
        top_k_idx = mx.argpartition(-seg_scores, kth=seg_k)[:seg_k]
        mx.eval(top_k_idx)
        for i in top_k_idx.tolist():
            all_indices.add(seg_start + i)

    return all_indices


# ---------------------------------------------------------------------------
# Freshness decay (Task 97) — cosine-similarity conflict detection
# ---------------------------------------------------------------------------

def freshness_mask(
    cache: list,
    target_layers: list[int] | None = None,
    conflict_threshold: float = 0.85,
    decay_factor: float = 0.1,
    window: int = 10,
) -> mx.array:
    """Compute per-token freshness scores via conflict detection.

    Tokens whose K vectors are highly similar to LATER tokens are
    "superseded" — their information has been updated. Multiply-superseded
    tokens get exponentially decayed freshness scores.

    This is the ORIGINAL loop variant (not Task 140's vectorized scatter).

    Args:
        cache: KVCache list (one per layer)
        target_layers: which layers to check (default: last 4)
        conflict_threshold: cosine similarity above this = superseded (0.85)
        decay_factor: freshness = decay^n_supersessions per superseded token
        window: check each token against the next `window` tokens

    Returns:
        freshness: (B, H_kv, T) — freshness score per token (1.0 = fresh,
            decay^n = stale). Multiply with importance scores.
    """
    n_layers = len(cache)
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 4), n_layers))

    all_counts = []

    for layer_idx in target_layers:
        keys = _get_fp16_keys(cache[layer_idx])   # (B, H_kv, T, D)
        B, H_kv, T, D = keys.shape

        # Normalize for cosine similarity
        norms = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True) + 1e-8)
        keys_normed = keys / norms                # (B, H_kv, T, D)

        supersession_count = mx.zeros((B, H_kv, T))

        # ORIGINAL LOOP VARIANT — processes chunks to avoid O(T²) memory
        for i_start in range(0, T - 1, 256):
            i_end = min(i_start + 256, T - 1)
            q = keys_normed[:, :, i_start:i_end, :]   # (B, H_kv, chunk, D)

            j_start = i_start + 1
            j_end = min(i_end + window, T)
            k = keys_normed[:, :, j_start:j_end, :]   # (B, H_kv, j_len, D)

            sims = q @ k.swapaxes(-1, -2)              # (B, H_kv, chunk, j_len)

            chunk_len = i_end - i_start
            j_len = j_end - j_start
            i_idx = mx.arange(chunk_len)[:, None]      # (chunk, 1)
            j_idx = mx.arange(j_len)[None, :]          # (1, j_len)
            valid = (j_idx >= i_idx) & (j_idx < i_idx + window)

            above_thresh = sims > conflict_threshold   # (B, H_kv, chunk, j_len)
            masked = above_thresh & valid[None, None, :, :]
            n_conflicts = mx.sum(masked, axis=-1)      # (B, H_kv, chunk)
            supersession_count[:, :, i_start:i_end] = n_conflicts

        mx.eval(supersession_count)
        all_counts.append(supersession_count)

    stacked = mx.stack(all_counts, axis=0)
    avg_counts = mx.mean(stacked, axis=0)   # (B, H_kv, T)

    # Freshness = decay_factor ^ n_supersessions
    freshness = mx.power(mx.array(decay_factor), avg_counts)
    mx.eval(freshness)
    return freshness
