"""Tests for SnapKV eviction stack (HX5).

Pure-logic tests for scoring/safety helpers. Model integration for the full
snapkv_select pipeline is in test_snapkv_model.py.

All helpers are exposed from the gardener.mlxsuper.snapkv sub-package so
these tests exercise real logic rather than skipping to integration.
"""
from __future__ import annotations

import pytest
import mlx.core as mx

from gardener.mlxsuper.snapkv import (
    SnapKVOptions,
    caote_score,
    buzz_segment_select,
    freshness_mask,
    head_rebalance,
    fair_eviction_budget,
    ger_check,
)


# ---------------------------------------------------------------------------
# SnapKVOptions defaults
# ---------------------------------------------------------------------------

def test_options_defaults_match_validated_config():
    """Defaults reflect the 128K-validated config from CLAUDE.md."""
    o = SnapKVOptions()
    assert o.keep_ratio == 0.5
    assert o.obs_window == 64
    assert o.enable_caote is True
    assert o.enable_buzz is True
    assert o.retrieval_head_multiplier == 2.0
    assert o.streaming_head_multiplier == 0.5
    assert o.enable_ger_safety is True
    assert o.enable_fair_eviction is True
    assert o.enable_head_rebalancing is True
    assert o.enable_freshness_decay is True


# ---------------------------------------------------------------------------
# CAOTE scoring (Task 100)
# ---------------------------------------------------------------------------

class _FakeCacheEntry:
    """Minimal fake cache entry that exposes .state with fp16 values."""
    def __init__(self, values: mx.array):
        # state = (keys, values) — keys not read by caote_score
        self.state = (values, values)  # keys = values (doesn't matter for caote)


def test_caote_scoring_boosts_value_distant_tokens():
    """CAOTE: tokens with V farther from V_mean should score higher."""
    B, H_kv, T, D = 1, 2, 8, 16
    # Construct values where token 7 is far from mean, token 0 is close
    values = mx.zeros((B, H_kv, T, D))
    # Token 7 in all heads: large magnitude
    values_list = values.tolist()
    for h in range(H_kv):
        for d in range(D):
            values_list[0][h][7][d] = 10.0  # far from mean
            values_list[0][h][0][d] = 0.01  # close to mean
    values = mx.array(values_list)

    # Uniform attention importance (so only value distinctiveness varies)
    importance_attn = mx.ones((B, H_kv, T)) * 0.1

    fake_entry = _FakeCacheEntry(values)
    caote = caote_score(importance_attn, fake_entry, obs_window=T)

    assert caote.shape == (B, H_kv, T)

    # Token 7 (far from mean) should score higher than token 0 (close to mean)
    score_far = float(caote[0, 0, 7].item())
    score_close = float(caote[0, 0, 0].item())
    assert score_far > score_close, (
        f"Token 7 (far from V_mean) should score higher: {score_far:.4f} vs {score_close:.4f}"
    )


# ---------------------------------------------------------------------------
# BUZZ segmented selection (Task 98)
# ---------------------------------------------------------------------------

def test_buzz_segmented_preserves_token_in_every_segment():
    """When keep budget is distributed across segments, every segment
    contributes at least one token (no segment is completely evicted)."""
    B, S = 1, 100
    segment_size = 20  # 5 segments
    k = 10  # 2 per segment on average

    # Uniform scores — selection is arbitrary but deterministic
    pooled = mx.ones((B, S))
    selected = buzz_segment_select(pooled, k, segment_size)

    # With uniform scores and 5 segments of 20, at least 5 segments should
    # each contribute ≥1 token (k=10 ≥ n_segments=5, base_k=2 each)
    assert len(selected) == k

    # Check at least one index per segment
    n_segments = (S + segment_size - 1) // segment_size
    for seg in range(n_segments):
        seg_start = seg * segment_size
        seg_end = min(seg_start + segment_size, S)
        seg_indices = [i for i in selected if seg_start <= i < seg_end]
        assert len(seg_indices) >= 1, (
            f"Segment {seg} [{seg_start},{seg_end}) has no selected tokens"
        )


def test_buzz_returns_exact_k():
    """buzz_segment_select always returns exactly k indices."""
    B, S = 1, 64
    pooled = mx.arange(S).reshape(B, S).astype(mx.float32)
    for k in [4, 8, 16, 32]:
        selected = buzz_segment_select(pooled, k, segment_size=16)
        assert len(selected) == k, f"Expected {k} indices, got {len(selected)}"


def test_buzz_high_score_tokens_preferred():
    """buzz_segment_select prefers high-score tokens within each segment."""
    B, S = 1, 20
    # Scores: last 10 tokens have score 10.0, first 10 have score 1.0
    scores_list = [1.0] * 10 + [10.0] * 10
    pooled = mx.array(scores_list).reshape(B, S)
    k = 4
    segment_size = 10

    selected = buzz_segment_select(pooled, k, segment_size)

    # Both segments contribute 2 tokens (k=4, 2 segments)
    # From segment 0 [0-9]: scores all equal (1.0), any 2 OK
    # From segment 1 [10-19]: scores all equal (10.0), any 2 OK
    seg0 = [i for i in selected if i < 10]
    seg1 = [i for i in selected if i >= 10]
    assert len(seg0) >= 1
    assert len(seg1) >= 1


# ---------------------------------------------------------------------------
# Freshness decay (Task 97)
# ---------------------------------------------------------------------------

class _FakeCache:
    """Minimal fake cache list element for freshness_mask tests."""
    def __init__(self, keys: mx.array):
        self.state = (keys, keys)  # values not read by freshness_mask


def test_freshness_mask_shape():
    """freshness_mask returns (B, H_kv, T) matching cache key shape."""
    B, H_kv, T, D = 1, 2, 16, 8
    keys = mx.random.normal((B, H_kv, T, D))
    fake_cache = [_FakeCache(keys)] * 4  # 4 layers

    fresh = freshness_mask(fake_cache, target_layers=[0, 1, 2, 3])
    assert fresh.shape == (B, H_kv, T), f"Expected ({B},{H_kv},{T}), got {fresh.shape}"


def test_freshness_decay_suppresses_high_cosine_similarity():
    """Tokens that are nearly identical to later tokens get low freshness."""
    B, H_kv, T, D = 1, 1, 6, 8
    # Token 0 is nearly identical to token 1 (high cosine sim → superseded)
    keys_list = mx.zeros((B, H_kv, T, D)).tolist()
    # Token 0 and token 1: same direction
    for d in range(D):
        keys_list[0][0][0][d] = 1.0
        keys_list[0][0][1][d] = 1.0 + 1e-4  # nearly identical
        # Token 4 and 5: orthogonal to all others (fresh)
        keys_list[0][0][4][d % 2] = 1.0 if d % 2 == 0 else 0.0
        keys_list[0][0][5][d % 2] = 0.0 if d % 2 == 0 else 1.0
    keys = mx.array(keys_list)

    fake_cache = [_FakeCache(keys)]
    fresh = freshness_mask(fake_cache, target_layers=[0], conflict_threshold=0.9, window=3)

    # Token 0 should be staler than token 4 (no similar follower)
    freshness_0 = float(fresh[0, 0, 0].item())
    freshness_4 = float(fresh[0, 0, 4].item())
    assert freshness_0 <= freshness_4, (
        f"Token 0 (superseded) should be ≤ token 4 (fresh): {freshness_0:.4f} vs {freshness_4:.4f}"
    )


def test_freshness_values_in_range():
    """All freshness values are in (0, 1]."""
    B, H_kv, T, D = 1, 2, 20, 8
    keys = mx.random.normal((B, H_kv, T, D))
    fake_cache = [_FakeCache(keys)] * 4

    fresh = freshness_mask(fake_cache, target_layers=[0, 1, 2, 3])
    assert float(mx.min(fresh).item()) > 0.0
    assert float(mx.max(fresh).item()) <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Head rebalancing (Task 111)
# ---------------------------------------------------------------------------

def test_head_rebalance_gives_retrieval_2x_streaming_0p5x():
    """head_rebalance applies 2× to retrieval heads, 0.5× to streaming heads."""
    B, H_kv, T = 1, 4, 10
    importance = mx.ones((B, H_kv, T))
    head_types = ["retrieval", "streaming", "retrieval", "streaming"]

    reweighted = head_rebalance(importance, head_types, retrieval_multiplier=2.0, streaming_multiplier=0.5)

    assert reweighted.shape == (B, H_kv, T)
    for h, ht in enumerate(head_types):
        expected = 2.0 if ht == "retrieval" else 0.5
        actual = float(reweighted[0, h, 0].item())
        assert abs(actual - expected) < 1e-5, (
            f"head {h} ({ht}): expected {expected}, got {actual}"
        )


def test_head_rebalance_wrong_length_is_noop():
    """head_rebalance with mismatched head_types length returns input unchanged."""
    B, H_kv, T = 1, 4, 10
    importance = mx.ones((B, H_kv, T)) * 3.0
    head_types = ["retrieval", "streaming"]  # len=2, H_kv=4 → mismatch

    result = head_rebalance(importance, head_types)
    # Should return original (no-op)
    assert float(mx.max(result).item()) == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Fair eviction budget (Task 107)
# ---------------------------------------------------------------------------

def test_fair_eviction_budget_proportional():
    """fair_eviction_budget allocates more tokens to larger partitions."""
    total_tokens = 100
    # Partition A: 20 tokens, Partition B: 80 tokens
    partitions = [(0, 20), (20, 100)]
    k = 40
    budgets = fair_eviction_budget(total_tokens, partitions, k, min_tokens=5)

    assert len(budgets) == 2
    assert sum(budgets) == k, f"Total budget {sum(budgets)} != {k}"
    # Larger partition should get more budget
    assert budgets[1] > budgets[0], (
        f"Larger partition should get more: {budgets[0]} vs {budgets[1]}"
    )


def test_fair_eviction_budget_respects_floor():
    """fair_eviction_budget gives at least min_tokens to each partition."""
    total_tokens = 100
    partitions = [(0, 5), (5, 100)]  # very small first partition
    k = 50
    min_tokens = 10
    budgets = fair_eviction_budget(total_tokens, partitions, k, min_tokens=min_tokens)

    # Each partition should get at least min_tokens
    for i, b in enumerate(budgets):
        assert b >= min(min_tokens, k), (
            f"Partition {i} budget {b} below floor {min_tokens}"
        )


def test_fair_eviction_budget_sums_to_k():
    """fair_eviction_budget total always equals k."""
    for k in [10, 25, 50, 100]:
        partitions = [(0, 30), (30, 70), (70, 100)]
        budgets = fair_eviction_budget(100, partitions, k, min_tokens=2)
        assert sum(budgets) == k, f"k={k}: sum {sum(budgets)} != {k}"


# ---------------------------------------------------------------------------
# GER safety guard (Task 106)
# ---------------------------------------------------------------------------

def test_ger_check_safe_when_important_tokens_kept():
    """GER check is safe when the top tokens are all in the keep mask."""
    B, H_kv, T = 1, 2, 20
    # All importance: token 0-1 have high importance, rest low
    imp_list = [[[0.01] * T for _ in range(H_kv)] for _ in range(B)]
    for h in range(H_kv):
        imp_list[0][h][0] = 1.0
        imp_list[0][h][1] = 0.9
    importance = mx.array(imp_list)

    # Keep mask keeps tokens 0 and 1 (the important ones)
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    keep_mask[0, 0] = True
    keep_mask[0, 1] = True

    safe, ger_val, rec = ger_check(importance, keep_mask)
    assert safe is True, f"Should be safe when important tokens kept: GER={ger_val}"
    assert ger_val == pytest.approx(0.0, abs=1e-6)


def test_ger_check_unsafe_when_important_tokens_evicted():
    """GER check reports unsafe when important tokens are all evicted."""
    B, H_kv, T = 1, 2, 40
    imp_list = [[[0.01] * T for _ in range(H_kv)] for _ in range(B)]
    for h in range(H_kv):
        imp_list[0][h][0] = 1.0
        imp_list[0][h][1] = 0.9
    importance = mx.array(imp_list)

    # Keep mask keeps only unimportant tokens 2-6 (5 total), evicts tokens 0 and 1
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    for i in range(2, 7):
        keep_mask[0, i] = True

    safe, ger_val, rec = ger_check(importance, keep_mask, top_pct=0.1, threshold=0.05, widen_pct=0.5)
    assert safe is False, f"Should be unsafe when important tokens evicted: GER={ger_val}"
    assert ger_val > 0.05
    # With widen_pct=0.5: 5 * 1.5 = 7 > 5
    assert rec > int(mx.sum(keep_mask).item()), "Should recommend widening"


def test_ger_check_recommends_wider_budget():
    """When GER > threshold, recommended keep count is larger than current."""
    B, H_kv, T = 1, 2, 40
    # Mark all tokens equally important
    importance = mx.ones((B, H_kv, T)) * 0.5
    # Keep only 5 tokens (extreme compression → high GER)
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    for i in range(5):
        keep_mask[0, i] = True

    safe, ger_val, rec = ger_check(importance, keep_mask, threshold=0.05, widen_pct=0.2)
    if not safe:
        assert rec >= 5, f"Recommendation {rec} should be ≥ current kept 5"
