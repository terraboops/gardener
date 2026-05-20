"""Helpers for hybrid attention models (SSM + transformer interleaved).

Qwen3.6-35B-A3B uses every-4th-layer full-attention (10 of 40 layers carry
KV state); the rest are SSM/linear. Only KV-bearing layers need a cache.

Reference map (NOT imported):
- omlx/state_space/hybrid_layers.py:21-49
- omlx/bench/hypercar_bench.py _make_cache(...)
- research/qwen36_migration_ready.md
"""
from __future__ import annotations

from typing import Any


def attention_layer_indices(model: Any) -> list[int]:
    """Return sorted indices of layers whose module has `self_attn`.

    Dense models (Qwen3-Coder): all indices [0..n_layers-1].
    Hybrid models (Qwen3.6): only the every-4th indices like [3, 7, 11, ...].

    Examples
    --------
    Dense model (all layers have attention)::

        attention_layer_indices(qwen3_coder)
        # [0, 1, 2, ..., 47]

    Hybrid model (every-4th layer has attention)::

        attention_layer_indices(qwen3_6)
        # [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    """
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return []
    return [i for i, layer in enumerate(layers)
            if hasattr(layer, "self_attn")]


def is_hybrid(model: Any) -> bool:
    """True if the model has SSM/non-attention layers interleaved with attn.

    A model is hybrid if not all of its layers have self_attn modules.
    On dense models (Qwen3-Coder), all layers have attention → False.
    On hybrid models (Qwen3.6), only ~25% of layers have attention → True.
    """
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return False
    n_total = len(layers)
    n_attn = len(attention_layer_indices(model))
    return n_attn < n_total


def eos_token_ids(tokenizer: Any) -> set[int]:
    """Unify Qwen3-Coder's singular `eos_token_id` and Qwen3.6's plural
    `eos_token_ids` into one set.

    Qwen3-Coder uses:
        tokenizer.eos_token_id = 151643  # singular int

    Qwen3.6 uses:
        tokenizer.eos_token_ids = [151643, 151644, ...]  # plural list/set

    This function returns a unified set regardless of which format the
    tokenizer exposes.
    """
    ids: set[int] = set()
    # Qwen3.6 / hybrid style.
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural is not None:
        try:
            ids.update(int(t) for t in plural)
        except TypeError:
            pass
    # Standard singular.
    singular = getattr(tokenizer, "eos_token_id", None)
    if singular is not None:
        try:
            ids.add(int(singular))
        except (TypeError, ValueError):
            pass
    return ids


def format_chat(
    tokenizer: Any,
    messages: list[dict],
    *,
    enable_thinking: bool = False,
) -> str:
    """Apply the tokenizer's chat template.

    Forwards `enable_thinking` when the tokenizer's apply_chat_template
    accepts it (Qwen3.6 does, Qwen3-Coder doesn't); falls back gracefully
    when the kwarg is unsupported.

    Parameters
    ----------
    tokenizer : Any
        A tokenizer instance with an optional `apply_chat_template` method.
    messages : list[dict]
        List of message dicts with "role" and "content" keys.
    enable_thinking : bool, optional
        Whether to enable thinking/COT mode (Qwen3.6 only). Default False.

    Returns
    -------
    str
        Formatted chat string ready for tokenization.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        # Last-resort plain concat. Real chat models always have apply_chat_template.
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}"
                          for m in messages)
    try:
        return apply(messages, tokenize=False,
                     add_generation_prompt=True,
                     enable_thinking=enable_thinking)
    except TypeError:
        # Fallback: tokenizer doesn't support enable_thinking kwarg.
        return apply(messages, tokenize=False,
                     add_generation_prompt=True)
