"""Shared helpers for the quality eval gates."""
from __future__ import annotations

import gc
from typing import Any

import mlx.core as mx


def per_item_cleanup() -> None:
    """gc.collect() + mx.clear_cache() — required between items on long eval
    runs to prevent the 100x slowdown that Task 259 documented for MMLU-Pro.
    Same pattern applied to all quality gates."""
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def format_chat_prompt(tokenizer: Any, text: str, *,
                       enable_thinking: bool = False) -> str:
    """Wrap raw text in the tokenizer's chat template if available.
    enable_thinking=False is critical for thinking models (Qwen3.6) because
    thinking blocks consume the budget before the answer emerges."""
    if not hasattr(tokenizer, "apply_chat_template"):
        return text
    msgs = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
