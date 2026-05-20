"""Tests for the prefill_last_logit patch.

Mostly model-bearing because the patch's value is in shape semantics during
real forward passes. Pure unit tests cover the idempotence + missing-attrs
paths.
"""
from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from gardener.mlxsuper.patches.prefill_last_logit import (
    apply_prefill_last_logit_patch, is_patched,
)


# --- pure-logic tests (no model) ----------------------------------------------

class _FakeBareModel(nn.Module):
    """Minimal model with backbone + lm_head — exercises the happy patch path
    without loading a real LM."""
    def __init__(self, vocab: int = 8, hidden: int = 4, n_layers: int = 1):
        super().__init__()
        # nn.Module won't tolerate arbitrary attrs without registration; we
        # define small stand-ins.
        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab, hidden)
            def __call__(self, inputs, cache=None, mask=None):
                return self.embed_tokens(inputs)
        self.model = _Backbone()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def __call__(self, inputs, cache=None, mask=None, **kw):
        h = self.model(inputs, cache=cache, mask=mask)
        return self.lm_head(h)


def test_patch_is_idempotent():
    m = _FakeBareModel()
    assert apply_prefill_last_logit_patch(m) is True
    assert is_patched(m) is True
    # Second call: returns False (already patched).
    assert apply_prefill_last_logit_patch(m) is False


def test_single_token_decode_is_unchanged_shape():
    m = _FakeBareModel(vocab=8, hidden=4)
    apply_prefill_last_logit_patch(m)
    # Single-token decode: pass-through, shape (B, 1, vocab).
    out = m(mx.array([[3]]), cache="non-empty-sentinel")
    assert out.shape == (1, 1, 8)


def test_multi_token_prefill_projects_only_last_token():
    m = _FakeBareModel(vocab=8, hidden=4)
    apply_prefill_last_logit_patch(m)
    # Multi-token prefill with cache: should project ONLY last token.
    out = m(mx.array([[1, 2, 3, 4, 5]]), cache="non-empty-sentinel")
    assert out.shape == (1, 1, 8)   # Was (1, 5, 8) without patch.


def test_no_cache_pass_through_unchanged():
    m = _FakeBareModel(vocab=8, hidden=4)
    apply_prefill_last_logit_patch(m)
    # Without a cache (one-shot eval), patch passes through.
    out = m(mx.array([[1, 2, 3]]), cache=None)
    assert out.shape == (1, 3, 8)


def test_missing_attrs_returns_false():
    class _IncompleteModel(nn.Module):
        def __init__(self):
            super().__init__()
        def __call__(self, x, **kw):
            return x
    m = _IncompleteModel()
    assert apply_prefill_last_logit_patch(m) is False
    assert is_patched(m) is False


# --- model integration --------------------------------------------------------

@pytest.mark.model
def test_real_model_prefill_returns_single_token_logits(loaded_model):
    """On a real loaded mlx_lm model, multi-token prefill with cache returns
    only one timestep of logits."""
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = loaded_model
    # Apply patch (idempotent if test_subagents already loaded the model).
    apply_prefill_last_logit_patch(model)
    cache = make_prompt_cache(model)
    ids = mx.array([tok.encode("The capital of France is")])
    out = model(ids, cache=cache)
    # Patched: only last token's logits projected.
    assert out.shape[1] == 1, f"expected shape[1]==1, got {out.shape}"
