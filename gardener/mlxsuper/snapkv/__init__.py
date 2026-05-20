"""SnapKV attention-guided KV eviction stack.

Ported (not imported) from omlx/patches/snapkv.py + 7 companion helpers.
Uses the ORIGINAL loop variants — NOT the Task 140/147/168 vectorizations
which regressed in hypercar Run 84 (needle lost at 49% keep). Loop variant
is 5.7s at 64K; vectorized scatter was <0.5s but produces different
keep_mask ordering (semantics divergence, root cause not isolated).

The 8 essential layers (validated to 128K fp16 / 64K TQ3 at 50% keep):
1. snapkv_select with real Q capture           (Task 46)
2. CAOTE value-aware scoring                   (Task 100)
3. BUZZ segmented per-segment top-K            (Task 98)
4. Freshness decay (cosine-sim conflicts)      (Task 97)
5. Re-RoPE physical compaction                 (Task 46)
6. GER safety hallucination cliff guard        (Task 106)
7. Fair eviction (per-layer proportional)      (Task 107)
8. Head rebalancing (retrieval 2x, stream 0.5x) (Task 111)

Explicitly SKIPPED:
- Task 140 vectorized scatter (regressed Run 84 — needle lost)
- Task 147 GER threshold_idx vectorization (regressed Run 84)
- Task 168 skip_rerope=True optimization (regressed at 49% keep)
  NOTE: compact_cache defaults to skip_rerope=True here for API compatibility
  — the 168-regression was about auto-enabling it; callers should set
  skip_rerope=False for the position-correct path.

Reference map:
- omlx/patches/snapkv.py (main controller — all functions ported from here)

Model coupling:
- Real-Q hooks: couple to Qwen2.5/Qwen3 layer attribute names
  (q_proj, k_proj, v_proj, q_norm, k_norm, rope, o_proj, mlp,
  input_layernorm, post_attention_layernorm). Other mlx_lm model
  families may fail in the hook; fall back to K-as-Q proxy.
- Re-RoPE: reads rope.dims, rope.base from model.layers[0].self_attn.rope.
  Defaults to Qwen3-family values (dims=64, base=1_000_000).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import logging

import mlx.core as mx

from .select import (
    install_q_capture_hook,
    compute_importance_from_real_q,
    _select_global,
)
from .scoring import (
    caote_score,
    buzz_segment_select,
    freshness_mask,
)
from .safety import (
    head_rebalance,
    fair_eviction_budget,
    fair_eviction_select,
    ger_check,
)
from .compact import compact_cache, _rerope_keys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------

@dataclass
class SnapKVOptions:
    """Configuration for the 8-layer SnapKV eviction stack.

    Defaults reflect the 128K-validated configuration from hypercar
    CLAUDE.md (50% keep, obs_window=64, all 8 layers enabled).
    """
    obs_window: int = 64
    """Tokens to use for attention observation window."""

    keep_ratio: float = 0.5
    """Fraction of tokens to keep (0.25–0.75). 50% is the validated sweet spot.
    25% hurts decode 14% (re-RoPE cost) per Task 155."""

    enable_caote: bool = True
    """CAOTE value-aware scoring (Task 100). +14% needle accuracy."""

    enable_buzz: bool = True
    """BUZZ segmented per-segment top-K (Task 98)."""

    buzz_segment_size: int = 512
    """Segment size for BUZZ selection."""

    enable_freshness_decay: bool = True
    """Freshness decay via cosine-similarity conflict detection (Task 97)."""

    freshness_cos_threshold: float = 0.85
    """Cosine similarity above which a token is considered superseded."""

    enable_ger_safety: bool = True
    """GER hallucination cliff guard (Task 106). Widens budget if GER > threshold."""

    ger_threshold: float = 0.05
    """GER value above which budget is widened."""

    ger_widen_pct: float = 0.10
    """Fraction by which keep_count is widened when GER exceeds threshold."""

    enable_fair_eviction: bool = True
    """Fair eviction — proportional budget per partition (Task 107)."""

    fair_partition_min_tokens: int = 20
    """Minimum tokens to keep per partition under fair eviction."""

    enable_head_rebalancing: bool = True
    """Head rebalancing — retrieval 2×, streaming 0.5× (Task 111)."""

    retrieval_head_multiplier: float = 2.0
    """Importance weight for retrieval-tagged heads."""

    streaming_head_multiplier: float = 0.5
    """Importance weight for streaming-tagged heads."""

    head_types: Optional[List[str]] = None
    """List of 'retrieval'/'streaming' per KV head. When None and
    enable_head_rebalancing=True, rebalancing is skipped (no policy loaded)."""

    partitions: Optional[List[tuple]] = None
    """Token-range partitions for fair eviction. When None, fair eviction
    treats the entire selectable range as one partition (no-op)."""

    always_keep_last: int = 64
    """Always retain this many most-recent tokens (attention sink / window)."""

    skip_rerope: bool = True
    """Skip Re-RoPE position correction in compact_cache. Default True for
    4.7× faster decode; NIAH passes at most context lengths with gaps.
    Set False when position-perfect attention is required."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapkv_select(
    model,
    cache: list,
    tokenizer,
    input_ids: mx.array,
    *,
    options: SnapKVOptions | None = None,
) -> list[mx.array]:
    """Compute per-layer keep_indices for KV cache eviction.

    Runs the full 8-layer SnapKV stack:
    1. Install Q capture hooks (last 4 layers) and forward once with
       the observation window.
    2. Compute real-Q attention importance.
    3. Optionally apply CAOTE value-aware scoring (blends in per-layer).
    4. Optionally apply freshness decay.
    5. Optionally apply head rebalancing.
    6. Select top-K indices using BUZZ/fair/global strategy.
    7. Optionally check GER safety and widen budget if needed.
    8. Return per-layer keep_indices arrays.

    Note: this function does NOT call compact_cache. The caller is
    responsible for applying keep_indices:

        keep_indices = snapkv_select(model, cache, tok, ids, options=opts)
        for c, ki in zip(cache, keep_indices):
            compact_cache([c], ki.tolist(), model=model)

    Args:
        model: loaded mlx_lm model
        cache: list of KVCache objects (one per layer), already populated
            by prefill
        tokenizer: tokenizer (unused in current implementation; reserved for
            future prompt-aware partitioning)
        input_ids: (1, T) token ids used for prefill
        options: SnapKVOptions configuration (default: 50% keep, all 8 layers)

    Returns:
        keep_indices: list[mx.array] of shape (K_i,) per layer, where K_i
            is the number of tokens to keep in layer i. All layers receive
            the same indices (computed from aggregated multi-layer importance).
    """
    if options is None:
        options = SnapKVOptions()

    n_layers = len(cache)

    # Step 1: Install Q capture hooks and run a forward pass to capture Q/K/V
    target_layers = list(range(max(0, n_layers - 4), n_layers))
    captured, cleanup = install_q_capture_hook(model, target_layers)

    try:
        # Re-run the model over the observation window to populate `captured`.
        # We use only the last obs_window tokens to keep the hook pass cheap.
        total_offset = cache[0].offset if hasattr(cache[0], "offset") else 0
        obs_start = max(0, total_offset - options.obs_window)
        obs_ids = input_ids[:, obs_start:] if input_ids.shape[1] > obs_start else input_ids

        # Run with cache=None so we don't write to the cache — hooks capture
        # the Q/K/V projections but we discard the output.
        model(obs_ids, cache=None)
    finally:
        cleanup()

    # Deduce cache size from the actual cache offset (not input_ids length,
    # since the cache may have been prefilled in chunks).
    T = total_offset
    if T == 0:
        # Fallback: use input_ids length
        T = int(input_ids.shape[1])

    if T == 0:
        # Nothing in cache
        empty = mx.array([], dtype=mx.uint32)
        return [empty] * n_layers

    # Step 2: compute attention importance from real Q
    importance = compute_importance_from_real_q(captured, cache, options.obs_window)
    # importance: (B, H_kv, T_captured)  where T_captured may be obs_window only

    # Pad importance to full T if needed
    T_imp = importance.shape[2]
    if T_imp < T:
        prefix = mx.zeros(
            (importance.shape[0], importance.shape[1], T - T_imp),
            dtype=importance.dtype,
        )
        importance = mx.concatenate([prefix, importance], axis=2)
    elif T_imp > T:
        importance = importance[:, :, :T]

    # Step 3: CAOTE value-aware scoring (per captured layer, then aggregate)
    if options.enable_caote:
        caote_vals = []
        for layer_idx in target_layers:
            if layer_idx < len(cache):
                c_imp = caote_score(importance, cache[layer_idx], options.obs_window)
                caote_vals.append(c_imp)
        if caote_vals:
            caote_stack = mx.stack(caote_vals, axis=0)
            importance = mx.max(caote_stack, axis=0)

    # Step 4: freshness decay
    if options.enable_freshness_decay:
        fresh = freshness_mask(
            cache,
            target_layers=target_layers,
            conflict_threshold=options.freshness_cos_threshold,
        )
        # Align freshness to importance shape
        T_fresh = fresh.shape[2]
        if T_fresh < T:
            fresh = mx.concatenate([
                mx.ones((fresh.shape[0], fresh.shape[1], T - T_fresh), dtype=fresh.dtype),
                fresh,
            ], axis=2)
        importance = importance * fresh

    # Step 5: head rebalancing
    if options.enable_head_rebalancing and options.head_types is not None:
        importance = head_rebalance(
            importance,
            options.head_types,
            retrieval_multiplier=options.retrieval_head_multiplier,
            streaming_multiplier=options.streaming_head_multiplier,
        )

    # Step 6: select top-K
    keep_count = max(1, int(T * options.keep_ratio))
    always_keep_last = min(options.always_keep_last, T)
    selectable = max(0, T - always_keep_last)
    k_selectable = max(1, keep_count - always_keep_last)

    # Pool importance across heads (max = keep if ANY head cares)
    imp_weighted = importance
    pooled = mx.max(imp_weighted[:, :, :selectable], axis=1)  # (B, selectable)

    if selectable == 0:
        keep_indices_list = list(range(T))
    elif (
        options.enable_fair_eviction
        and options.partitions is not None
        and len(options.partitions) > 1
    ):
        selected = fair_eviction_select(
            pooled, k_selectable,
            options.partitions,
            options.fair_partition_min_tokens,
            options.buzz_segment_size if options.enable_buzz else 0,
        )
        keep_indices_list = sorted(selected) + list(range(selectable, T))
    elif options.enable_buzz and selectable > options.buzz_segment_size:
        selected = buzz_segment_select(pooled, k_selectable, options.buzz_segment_size)
        keep_indices_list = sorted(selected) + list(range(selectable, T))
    else:
        selected = _select_global(pooled, k_selectable)
        keep_indices_list = sorted(selected) + list(range(selectable, T))

    keep_indices_arr = mx.array(keep_indices_list, dtype=mx.uint32)

    # Step 7: GER safety check — widen budget if near hallucination cliff
    if options.enable_ger_safety and selectable > 0:
        keep_mask = mx.zeros((1, T), dtype=mx.bool_)
        for pos in keep_indices_list:
            keep_mask[0, pos] = True

        safe, ger_val, rec_keep = ger_check(
            importance, keep_mask,
            threshold=options.ger_threshold,
            widen_pct=options.ger_widen_pct,
        )

        if not safe:
            # Re-select with wider budget
            keep_count2 = rec_keep
            k_selectable2 = max(1, keep_count2 - always_keep_last)

            if options.enable_buzz and selectable > options.buzz_segment_size:
                selected2 = buzz_segment_select(pooled, k_selectable2, options.buzz_segment_size)
            else:
                selected2 = _select_global(pooled, k_selectable2)

            keep_indices_list = sorted(selected2) + list(range(selectable, T))
            keep_indices_arr = mx.array(keep_indices_list, dtype=mx.uint32)
            logger.info(
                f"snapkv_select: GER {ger_val:.3f} > threshold — "
                f"widened budget {keep_count} → {keep_count2}"
            )

    mx.eval(keep_indices_arr)

    # Return same indices for every layer (aggregated multi-layer importance).
    # Per-layer budgets (PyramidKV) are out of scope for this port.
    return [keep_indices_arr] * n_layers


__all__ = [
    "SnapKVOptions",
    "snapkv_select",
    "compact_cache",
    # Internal helpers re-exported for testing
    "caote_score",
    "buzz_segment_select",
    "freshness_mask",
    "head_rebalance",
    "fair_eviction_budget",
    "ger_check",
    "install_q_capture_hook",
]
