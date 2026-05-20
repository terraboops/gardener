"""Tests for the HX2 adaptive prefill chunk controller.

Pure-logic tests — no model load required (unless marked @pytest.mark.model).
"""
import pytest
from gardener.mlxsuper.patches.adaptive_prefill import (
    AdaptivePrefillController,
    apply_adaptive_prefill,
    is_patched,
    DEFAULTS,
)


def test_controller_defaults_apply():
    c = AdaptivePrefillController()
    assert c.min_chunk == 512
    assert c.max_chunk == 16384
    assert 0.0 < c.target_metal_pct < 1.0
    assert c.throughput_floor > 0


def test_first_chunk_is_max_chunk_when_remaining_is_large():
    c = AdaptivePrefillController(max_chunk=8192, min_chunk=512)
    assert c.next_chunk_size(remaining=100_000) == 8192


def test_chunk_never_exceeds_remaining():
    c = AdaptivePrefillController(max_chunk=8192, min_chunk=512)
    assert c.next_chunk_size(remaining=300) <= 300


def test_feedback_above_throughput_floor_keeps_or_grows_chunk():
    c = AdaptivePrefillController(
        max_chunk=8192, min_chunk=512,
        throughput_floor=200.0, target_metal_pct=0.65
    )
    first = c.next_chunk_size(remaining=100_000)
    # Healthy: 50% Metal, 1000 tok/s throughput.
    c.feedback(metal_gb=24.0, tok_per_sec=1000.0)
    second = c.next_chunk_size(remaining=100_000)
    assert second >= first  # may grow or hold steady


def test_feedback_below_throughput_floor_shrinks_chunk():
    c = AdaptivePrefillController(
        max_chunk=8192, min_chunk=512,
        throughput_floor=200.0, target_metal_pct=0.65
    )
    first = c.next_chunk_size(remaining=100_000)
    # Cliff detected: throughput 50 tok/s (well below 200 floor).
    c.feedback(metal_gb=24.0, tok_per_sec=50.0)
    second = c.next_chunk_size(remaining=100_000)
    assert second < first


def test_feedback_above_metal_target_shrinks_proportionally():
    c = AdaptivePrefillController(
        max_chunk=8192, min_chunk=512,
        throughput_floor=200.0, target_metal_pct=0.65
    )
    first = c.next_chunk_size(remaining=100_000)
    # Metal pressure: 90% of 48 GB.
    c.feedback(metal_gb=43.0, tok_per_sec=1000.0)
    second = c.next_chunk_size(remaining=100_000)
    assert second < first


def test_chunk_floors_at_min_chunk():
    c = AdaptivePrefillController(
        max_chunk=8192, min_chunk=512,
        throughput_floor=200.0, target_metal_pct=0.65
    )
    # Repeatedly slam the controller with cliff signals.
    for _ in range(20):
        c.next_chunk_size(remaining=100_000)
        c.feedback(metal_gb=46.0, tok_per_sec=20.0)
    assert c.next_chunk_size(remaining=100_000) >= 512


def test_reset_restores_starting_chunk():
    c = AdaptivePrefillController(max_chunk=8192, min_chunk=512)
    c.next_chunk_size(remaining=100_000)
    c.feedback(metal_gb=43.0, tok_per_sec=50.0)
    c.reset()
    assert c.next_chunk_size(remaining=100_000) == 8192


def test_summary_returns_expected_keys():
    c = AdaptivePrefillController()
    c.next_chunk_size(remaining=100_000)
    c.feedback(metal_gb=20.0, tok_per_sec=500.0)
    s = c.summary()
    expected_keys = {
        "current_chunk", "chunks_seen", "min_chunk", "max_chunk",
        "target_metal_pct", "throughput_floor",
    }
    assert expected_keys <= set(s)


def test_apply_adaptive_prefill_is_idempotent():
    assert apply_adaptive_prefill() in (True, False)  # depends on test ordering
    assert is_patched() is True
    assert apply_adaptive_prefill() is False  # second call: already patched


# ---------------------------------------------------------------------------
# Optional model tests
# ---------------------------------------------------------------------------

@pytest.mark.model
def test_short_prompts_pass_through(loaded_model):
    """Below min_prompt_tokens, the controller is a no-op — generation
    proceeds normally."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    apply_adaptive_prefill(min_prompt_tokens=1024)
    model, tok = loaded_model
    out = generate(
        model, tok, prompt="Hello", max_tokens=4,
        sampler=make_sampler(temp=0.0), verbose=False,
    )
    assert isinstance(out, str) and len(out) > 0
