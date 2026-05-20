"""Project only the last token through lm_head during prefill.

Standard mlx_lm models always compute logits for every input token. During
prefill (seq_len >> 1), only the last token's logits are actually used by
the sampling loop — the rest are discarded. For long-context prefill
(>8K tokens) this leaves a (seq_len, vocab_size) tensor in Metal memory
that can dominate the cache budget. At 128K context with vocab_size=151936
the logits tensor is ~40 GB.

This patch wraps the model's __call__ so that, when input has multiple
tokens AND cache is provided (i.e. we're in a prefill chunk), only the
last token is projected through lm_head. Single-token decode and prefill
without a cache pass through unchanged.

Reference map (NOT imported): omlx/patches/prefill_last_logit.py:1-71.
Re-derived on stock mlx_lm; no omlx import.
"""
from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger("gardener.mlxsuper.patches.prefill_last_logit")

_PATCHED_ATTR = "_gardener_prefill_last_logit_patched"


def apply_prefill_last_logit_patch(model: nn.Module) -> bool:
    """Monkey-patch model.__call__ to project only the last token through
    lm_head during multi-token prefill.

    Returns True if applied; False if already patched, missing required
    submodules, or the model architecture doesn't fit (no `lm_head` or
    no `model` backbone attribute).

    Idempotent: re-applying is a no-op (returns False).

    Implementation note: mlx.nn.Module's __call__ is a C-extension slot.
    Instance-level attribute assignment (types.MethodType) is shadowed by
    the C-level descriptor. The reliable approach is to reassign
    model.__class__ to a dynamically-created subclass that overrides
    __call__, which Python's type system dispatches correctly.
    """
    if getattr(model, _PATCHED_ATTR, False):
        return False

    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)

    # Some mlx_lm models tie embeddings (Qwen3.6-style: lm_head is
    # embed_tokens.as_linear()). Detect via args.tie_word_embeddings.
    tie_word_embeddings = False
    args = getattr(model, "args", None)
    if args is not None:
        tie_word_embeddings = getattr(args, "tie_word_embeddings", False)

    if backbone is None or (lm_head is None and not tie_word_embeddings):
        logger.warning(
            "prefill_last_logit patch: model missing required attributes "
            "(backbone=%s, lm_head=%s, tie=%s); skipping",
            backbone is not None, lm_head is not None, tie_word_embeddings,
        )
        return False

    original_class = type(model)
    original_call = original_class.__call__

    # Capture everything the inner call needs from the model up front so
    # the closure doesn't hold a strong reference cycle through `model`.
    _backbone = backbone
    _lm_head = lm_head
    _tie = tie_word_embeddings

    class _PrefillLastLogit(original_class):  # type: ignore[valid-type]
        """Dynamically-created subclass that overrides __call__ to project
        only the last token through lm_head during multi-token prefill."""

        def __call__(  # type: ignore[override]
            self,
            inputs: mx.array,
            cache: Any = None,
            mask: Any = None,
            **kwargs: Any,
        ) -> mx.array:
            # Single-token decode OR no cache: pass through unchanged.
            if inputs.ndim < 2 or inputs.shape[1] == 1 or cache is None:
                return original_call(self, inputs, cache=cache, mask=mask,
                                     **kwargs)

            # Multi-token prefill with cache: project only last token.
            # Pass **kwargs (e.g. input_embeddings) but NOT mask — backbone
            # signatures vary across mlx_lm model families; mask is a
            # top-level model param, not a backbone param in stock mlx_lm.
            hidden = _backbone(inputs, cache=cache, **kwargs)
            last_hidden = hidden[:, -1:, :]
            if _tie:
                logits = _backbone.embed_tokens.as_linear(last_hidden)
            else:
                logits = _lm_head(last_hidden)
            return logits

    # Rename for cleaner repr / logging.
    _PrefillLastLogit.__name__ = f"{original_class.__name__}_PrefillPatched"
    _PrefillLastLogit.__qualname__ = _PrefillLastLogit.__name__

    model.__class__ = _PrefillLastLogit
    setattr(model, _PATCHED_ATTR, True)
    logger.info("prefill_last_logit patch applied to %s",
                original_class.__name__)
    return True


def is_patched(model: nn.Module) -> bool:
    return bool(getattr(model, _PATCHED_ATTR, False))
