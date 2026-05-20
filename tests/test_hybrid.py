"""Tests for hybrid-attention support helpers (HX7)."""
from __future__ import annotations

import pytest

from gardener.mlxsuper.hybrid import (
    attention_layer_indices,
    eos_token_ids,
    format_chat,
    is_hybrid,
)


# --- attention_layer_indices / is_hybrid ------------------------------------


class _Layer:
    """Minimal layer mock for testing."""

    def __init__(self, has_attn: bool):
        if has_attn:
            self.self_attn = object()


class _Backbone:
    """Minimal model backbone for testing."""

    def __init__(self, has_attn_pattern: list[bool]):
        self.layers = [_Layer(b) for b in has_attn_pattern]


class _Model:
    """Minimal model mock for testing."""

    def __init__(self, has_attn_pattern: list[bool]):
        self.model = _Backbone(has_attn_pattern)


def test_dense_model_all_layers_are_attention():
    """Dense models have self_attn on every layer."""
    m = _Model([True] * 6)
    assert attention_layer_indices(m) == [0, 1, 2, 3, 4, 5]
    assert is_hybrid(m) is False


def test_hybrid_model_returns_only_attention_indices():
    """Hybrid models (e.g., Qwen3.6) have self_attn only on every-4th layer."""
    # Mimic Qwen3.6: every 4th layer is attention.
    # Indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 have attention.
    pattern = [(i + 1) % 4 == 0 for i in range(40)]
    m = _Model(pattern)
    expected = [i for i in range(40) if (i + 1) % 4 == 0]
    assert attention_layer_indices(m) == expected
    assert is_hybrid(m) is True


def test_missing_model_attrs_returns_empty():
    """Models without layers return empty list."""
    class _Bare:
        pass

    assert attention_layer_indices(_Bare()) is not None
    assert attention_layer_indices(_Bare()) == []
    assert is_hybrid(_Bare()) is False


def test_missing_layers_returns_empty():
    """Models without model.layers return empty list."""
    class _NoLayers:
        model = object()

    assert attention_layer_indices(_NoLayers()) == []
    assert is_hybrid(_NoLayers()) is False


def test_all_non_attention_returns_empty_indices():
    """All SSM layers (no attention) returns empty indices."""
    m = _Model([False] * 8)
    assert attention_layer_indices(m) == []
    assert is_hybrid(m) is True  # Has SSM layers interleaved


# --- eos_token_ids -----------------------------------------------------------


def test_eos_unifies_singular_and_plural():
    """When both singular and plural exist, union them."""

    class _Tok:
        eos_token_id = 5
        eos_token_ids = [5, 6, 7]

    assert eos_token_ids(_Tok()) == {5, 6, 7}


def test_eos_only_singular():
    """Extract from singular eos_token_id."""

    class _Tok:
        eos_token_id = 11

    assert eos_token_ids(_Tok()) == {11}


def test_eos_only_plural():
    """Extract from plural eos_token_ids."""

    class _Tok:
        eos_token_ids = {2, 3}

    assert eos_token_ids(_Tok()) == {2, 3}


def test_eos_plural_as_list():
    """Plural eos_token_ids can be a list."""

    class _Tok:
        eos_token_ids = [10, 20, 30]

    assert eos_token_ids(_Tok()) == {10, 20, 30}


def test_eos_neither_returns_empty():
    """No eos attributes returns empty set."""

    class _Tok:
        pass

    assert eos_token_ids(_Tok()) == set()


def test_eos_singular_is_none():
    """eos_token_id = None is safely ignored."""

    class _Tok:
        eos_token_id = None
        eos_token_ids = [7, 8]

    assert eos_token_ids(_Tok()) == {7, 8}


def test_eos_plural_is_none():
    """eos_token_ids = None is safely ignored."""

    class _Tok:
        eos_token_id = 42
        eos_token_ids = None

    assert eos_token_ids(_Tok()) == {42}


def test_eos_handles_deduplication():
    """Overlapping singular and plural are deduplicated."""

    class _Tok:
        eos_token_id = 5
        eos_token_ids = [5, 5, 6, 6]

    assert eos_token_ids(_Tok()) == {5, 6}


# --- format_chat -------------------------------------------------------------


def test_format_chat_forwards_enable_thinking_when_supported():
    """When apply_chat_template accepts enable_thinking, forward it."""
    captured = {}

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            captured.update(kw)
            return "rendered"

    out = format_chat(
        _Tok(), [{"role": "user", "content": "hi"}], enable_thinking=True
    )
    assert out == "rendered"
    assert captured.get("enable_thinking") is True


def test_format_chat_default_enable_thinking_false():
    """Default enable_thinking is False."""
    captured = {}

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            captured.update(kw)
            return "rendered"

    out = format_chat(_Tok(), [{"role": "user", "content": "hi"}])
    assert out == "rendered"
    assert captured.get("enable_thinking") is False


def test_format_chat_fallback_when_enable_thinking_unsupported():
    """When enable_thinking kwarg is unsupported, try without it."""
    captured_calls = []

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            captured_calls.append(dict(kw))
            if "enable_thinking" in kw:
                raise TypeError("enable_thinking not supported")
            return "rendered-no-thinking"

    out = format_chat(_Tok(), [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert out == "rendered-no-thinking"
    # First call had enable_thinking, second call did not.
    assert len(captured_calls) == 2
    assert "enable_thinking" in captured_calls[0]
    assert "enable_thinking" not in captured_calls[1]


def test_format_chat_missing_apply_chat_template():
    """Fallback to plain concat when apply_chat_template is missing."""

    class _Tok:
        pass

    out = format_chat(_Tok(), [{"role": "user", "content": "hello"}])
    assert "user" in out
    assert "hello" in out


def test_format_chat_preserves_message_structure():
    """Format chat preserves multi-turn structure."""

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            # Simple mock: join role:content pairs.
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 2+3?"},
    ]
    out = format_chat(_Tok(), messages)
    assert "user: What is 2+2?" in out
    assert "assistant: 4" in out
    assert "user: What is 2+3?" in out


# --- Model integration (uses the existing loaded_model fixture) ---------------


@pytest.mark.model
def test_real_model_eos_and_attn_indices(loaded_model):
    """Test on real Qwen2.5-0.5B model (dense attention, standard tokenizer)."""
    model, tok = loaded_model

    # Check attention indices.
    indices = attention_layer_indices(model)
    assert len(indices) > 0

    # Qwen2.5-0.5B is dense → indices should be [0..n-1].
    backbone = getattr(model, "model", model)
    n_layers = len(backbone.layers)
    assert indices == list(range(n_layers))
    assert is_hybrid(model) is False

    # Tokenizer has an eos id.
    eos_ids = eos_token_ids(tok)
    assert len(eos_ids) >= 1
    # All should be integers.
    assert all(isinstance(t, int) for t in eos_ids)


@pytest.mark.model
def test_real_model_format_chat(loaded_model):
    """Test format_chat on real tokenizer."""
    model, tok = loaded_model

    messages = [{"role": "user", "content": "Hello"}]
    out = format_chat(tok, messages)
    assert isinstance(out, str)
    assert len(out) > 0

    # Should not raise even if enable_thinking is unsupported.
    out2 = format_chat(tok, messages, enable_thinking=True)
    assert isinstance(out2, str)
    assert len(out2) > 0
