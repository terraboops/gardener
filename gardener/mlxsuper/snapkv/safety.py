"""SnapKV safety helpers: GER guard, fair eviction, head rebalancing.

Ported (not imported) from omlx/patches/snapkv.py.
These are pure functions over importance tensors and budgets — no model
coupling, fully unit-testable without model load.
"""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Head rebalancing (Task 111)
# ---------------------------------------------------------------------------

def head_rebalance(
    importance: mx.array,
    head_types: list[str],
    retrieval_multiplier: float = 2.0,
    streaming_multiplier: float = 0.5,
) -> mx.array:
    """Reweight importance by head type: retrieval 2×, streaming 0.5×.

    Shifts the pooled selection budget toward retrieval heads where
    it matters most for quality. Streaming heads (ring-buffer short-term
    context) can tolerate higher eviction.

    Args:
        importance: (B, H_kv, T) — raw importance scores
        head_types: list of "retrieval" or "streaming" per KV head (len=H_kv)
        retrieval_multiplier: weight for retrieval-tagged heads (default 2.0)
        streaming_multiplier: weight for streaming-tagged heads (default 0.5)

    Returns:
        importance_reweighted: (B, H_kv, T) — reweighted importance
    """
    B, H_kv, T = importance.shape
    if len(head_types) != H_kv:
        logger.warning(
            f"head_rebalance: head_types len {len(head_types)} != H_kv {H_kv}, skipping"
        )
        return importance

    weights = mx.array([
        retrieval_multiplier if t == "retrieval" else streaming_multiplier
        for t in head_types
    ]).reshape(1, H_kv, 1)

    return importance * weights


# ---------------------------------------------------------------------------
# Fair eviction (Task 107) — proportional partition budgets
# ---------------------------------------------------------------------------

def fair_eviction_budget(
    total_tokens: int,
    partitions: list[tuple[int, int]],
    k: int,
    min_tokens: int = 20,
) -> list[int]:
    """Allocate keep budget proportionally across token-range partitions.

    From "Pitfalls of KV Cache Compression" (arXiv:2510.00231): allocates
    budget proportionally to each partition's size, with a floor to protect
    small partitions from being fully evicted.

    Args:
        total_tokens: total selectable token count
        partitions: list of (start, end) token ranges (non-overlapping)
        k: total tokens to select across all partitions
        min_tokens: minimum budget per partition (floor)

    Returns:
        budgets: per-partition keep counts (same len as partitions)
    """
    n_parts = len(partitions)
    floor_total = min_tokens * n_parts

    if floor_total >= k:
        # Not enough budget for floors — distribute evenly
        per = max(1, k // n_parts)
        return [per] * n_parts

    part_lens = [max(0, min(e, total_tokens) - s) for s, e in partitions]
    total_len = sum(part_lens) or 1
    remaining = k - floor_total

    budgets = [
        min_tokens + int(remaining * pl / total_len)
        for pl in part_lens
    ]

    # Correct for rounding error — loop until exact
    while sum(budgets) < k:
        for idx in range(n_parts):
            if sum(budgets) >= k:
                break
            s, e = partitions[idx]
            pl = max(0, min(e, total_tokens) - s)
            if budgets[idx] < pl:
                budgets[idx] += 1
    while sum(budgets) > k:
        for idx in range(n_parts):
            if sum(budgets) <= k:
                break
            if budgets[idx] > 0:
                budgets[idx] -= 1

    return budgets


def _select_top_k_indices(scores: mx.array, k: int) -> list[int]:
    """Top-K indices from a 1-D score array (loop variant, not vectorized)."""
    if k <= 0:
        return []
    k = min(k, scores.shape[0])
    top_k = mx.argpartition(-scores, kth=k)[:k]
    mx.eval(top_k)
    return top_k.tolist()


def fair_eviction_select(
    pooled: mx.array,
    k: int,
    partitions: list[tuple[int, int]],
    min_tokens: int = 20,
    segment_size: int = 0,
) -> set[int]:
    """Select top-K indices using per-partition proportional budgets.

    Args:
        pooled: (B, S) — pooled importance scores for selectable range
        k: total indices to select
        partitions: (start, end) ranges in selectable-range coordinates
        min_tokens: per-partition minimum
        segment_size: if > 0, apply BUZZ within each partition

    Returns:
        set of integer indices into the selectable range
    """
    from .scoring import buzz_segment_select  # avoid circular at module level

    B, S = pooled.shape
    budgets = fair_eviction_budget(S, partitions, k, min_tokens)

    all_indices: set[int] = set()

    for (start, end), budget in zip(partitions, budgets):
        part_start = start
        part_end = min(end, S)
        part_len = part_end - part_start
        part_k = min(budget, part_len)

        if part_k <= 0:
            continue

        part_pooled = pooled[:, part_start:part_end]  # (B, part_len)

        if segment_size > 0 and part_len > segment_size:
            seg_indices = buzz_segment_select(part_pooled, part_k, segment_size)
            for idx in seg_indices:
                all_indices.add(part_start + idx)
        else:
            local_idxs = _select_top_k_indices(part_pooled[0], part_k)
            for idx in local_idxs:
                all_indices.add(part_start + idx)

    return all_indices


# ---------------------------------------------------------------------------
# GER safety guard (Task 106)
# ---------------------------------------------------------------------------

def ger_check(
    importance: mx.array,
    keep_mask: mx.array,
    top_pct: float = 0.1,
    threshold: float = 0.05,
    widen_pct: float = 0.10,
) -> tuple[bool, float, int]:
    """Check GER safety and recommend budget adjustment if needed.

    Global Eviction Ratio (GER) measures what fraction of the most-important
    tokens are being evicted across ALL heads simultaneously. A spike above
    ~5% correlates with the hallucination cliff.

    From "Physics of KV Cache Compression" (arXiv:2603.01426).

    Args:
        importance: (B, H_kv, T) — per-token importance per head
        keep_mask: (B, T) — boolean mask of kept tokens
        top_pct: fraction of tokens considered important for GER (default 10%)
        threshold: GER above this triggers budget widening (default 0.05)
        widen_pct: how much to widen keep_count when GER exceeds threshold

    Returns:
        (safe, ger_value, recommended_keep_count)
        safe=True when GER ≤ threshold.
    """
    B, H_kv, T = importance.shape
    evicted = ~keep_mask[0]              # (T,) True = evicted

    pooled = mx.max(importance[0], axis=0)  # (T,) max across heads
    n_important = max(1, int(T * top_pct))

    # ORIGINAL LOOP VARIANT for GER — not Task 147's threshold_idx scatter
    top_k = mx.argpartition(-pooled, kth=n_important)[:n_important]
    mx.eval(top_k)
    top_k_list = top_k.tolist()

    important_mask = mx.zeros(T, dtype=mx.bool_)
    for idx in top_k_list:
        important_mask[idx] = True

    important_evicted = mx.sum(important_mask & evicted)
    ger = float(important_evicted.item()) / max(n_important, 1)

    current_kept = int(mx.sum(keep_mask).item())
    recommended = current_kept

    if ger > threshold:
        recommended = min(T, int(current_kept * (1 + widen_pct)))
        logger.warning(
            f"GER safety: {ger:.3f} > {threshold} — widening budget "
            f"{current_kept} → {recommended}"
        )

    return ger <= threshold, ger, recommended
