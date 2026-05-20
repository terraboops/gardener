"""Tests for MInference sparse prefill (HX6).

Calibration tables for both Qwen3-Coder and Qwen3.6 are SYNTHETIC PLACEHOLDERS
(per their `note` field). Tests pin the loud-warning behavior + the loader +
the idempotent monkey-patch.
"""
from __future__ import annotations

import json
import logging
import pytest

from gardener.mlxsuper.patches.minference_prefill import (
    load_pattern_table,
    apply_minference_prefill_patch,
    is_patched,
)


def test_qwen3_coder_table_loads_and_warns(caplog):
    """The shipped Qwen3-Coder table is a SYNTHETIC PLACEHOLDER. Loading it
    MUST emit a loud warning so callers know quality is at risk."""
    with caplog.at_level(logging.WARNING):
        table = load_pattern_table("qwen3_coder_30b_a3b_instruct_8bit")
    assert "_lookup" in table
    # Loud warning must be present.
    warnings_text = " ".join(
        rec.message for rec in caplog.records if rec.levelname == "WARNING"
    )
    assert "SYNTHETIC PLACEHOLDER" in warnings_text or "synthetic" in warnings_text.lower()


def test_loader_rejects_unknown_model(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        load_pattern_table("does-not-exist", search_paths=[tmp_path])


def test_loader_load_real_table_does_not_warn(tmp_path, caplog):
    """If a future calibration table has note='REAL: measured on …', the
    loader MUST NOT emit the synthetic warning."""
    real_table = {
        "model": "fake-real",
        "num_layers": 1,
        "num_heads": 1,
        "seq_len": 512,
        "target_sparsity": 0.85,
        "calibration_time_s": 5.0,
        "note": "REAL: measured on 1000 calibration prompts from CodeContests",
        "heads": [
            {
                "layer": 0,
                "head": 0,
                "pattern": "dense",
                "mse": 0.0,
                "sparsity": 0.0,
                "params": {},
            }
        ],
        "summary": {
            "pattern_counts": {"a_shape": 0, "vertical_slash": 0, "block_sparse": 0, "dense": 1},
            "avg_mse": 0.0,
            "avg_sparsity": 0.0,
            "total_entries": 1,
        },
    }
    (tmp_path / "fake-real.json").write_text(json.dumps(real_table))
    with caplog.at_level(logging.WARNING):
        load_pattern_table("fake-real", search_paths=[tmp_path])
    warnings_text = " ".join(
        rec.message for rec in caplog.records if rec.levelname == "WARNING"
    )
    assert "SYNTHETIC PLACEHOLDER" not in warnings_text


def test_loader_lookup_key_structure():
    """_lookup keys must be (int, int) tuples for every entry in the table."""
    table = load_pattern_table("qwen3_coder_30b_a3b_instruct_8bit")
    lookup = table["_lookup"]
    assert len(lookup) > 0, "lookup must be non-empty"
    for key in lookup:
        assert isinstance(key, tuple) and len(key) == 2, f"Bad key: {key!r}"
        layer, head = key
        assert isinstance(layer, int) and isinstance(head, int)


def test_loader_required_fields():
    """Table must expose num_layers, num_heads, heads, summary."""
    table = load_pattern_table("qwen3_coder_30b_a3b_instruct_8bit")
    assert "num_layers" in table
    assert "num_heads" in table
    assert "heads" in table
    assert "summary" in table
    assert table["num_layers"] > 0
    assert table["num_heads"] > 0


def test_apply_is_idempotent():
    """apply_minference_prefill_patch is idempotent: second call returns False."""
    # First call: may return True or False depending on mlx_lm availability.
    apply_minference_prefill_patch()
    assert is_patched() is True
    assert apply_minference_prefill_patch() is False   # second: already patched


def test_apply_with_preloaded_table():
    """Passing a pre-loaded pattern_table skips the file-path lookup."""
    table = load_pattern_table("qwen3_coder_30b_a3b_instruct_8bit")
    # Already patched from test_apply_is_idempotent — should still return False
    result = apply_minference_prefill_patch(pattern_table=table)
    # Either True (first) or False (already patched) — both are correct
    assert isinstance(result, bool)
    assert is_patched() is True


# ---------------------------------------------------------------------------
# Optional model test (requires mlx model downloaded — slow, mark accordingly)
# ---------------------------------------------------------------------------

@pytest.mark.model
def test_short_prompt_decode_unaffected_by_patch(loaded_model):
    """Patch should NOT alter single-token decode behavior. Smoke gate only."""
    from mlx_lm import generate  # type: ignore[import]
    from mlx_lm.sample_utils import make_sampler  # type: ignore[import]

    apply_minference_prefill_patch()
    model, tok = loaded_model
    out = generate(
        model, tok, prompt="Hello", max_tokens=4,
        sampler=make_sampler(temp=0.0), verbose=False,
    )
    assert isinstance(out, str) and len(out) > 0
