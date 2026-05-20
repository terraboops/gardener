"""MLX-backed agent runner. Each agent name maps to a persistent SessionPool
session so successive calls reuse the KV cache (warm context).

For v0 simplicity, the system prompt is sent fresh each call (no incremental
chat history beyond the cache). The next iteration will wire conversation
history through cache reuse + rewind."""
from __future__ import annotations

from typing import Any, Optional

from mlx_lm import generate, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from ..pipeline.ir import Node


class MLXAgentRunner:
    """Agent runner backed by mlx_lm.

    When ``draft_model`` is provided, uses the speculative-decoding path
    (``stream_generate`` with the drafter) instead of the standard
    ``generate`` call. Acceptance-rate (α) telemetry is accumulated across
    all calls and exposed via :meth:`spec_decode_stats`.

    Hypercar guidance (CLAUDE.md):
    - α ≥ 0.5 → ~3× decode speedup at N=8: spec-decoding is winning.
    - 0.3 ≤ α < 0.5 → partial win; try smaller drafter or larger N.
    - α < 0.3 → not winning on this workload; disable.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        draft_model: Optional[Any] = None,
        draft_tokenizer: Optional[Any] = None,
        num_draft_tokens: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._draft_model = draft_model
        self._num_draft_tokens = num_draft_tokens

        if draft_model is not None:
            if draft_tokenizer is None:
                raise ValueError(
                    "draft_tokenizer must be supplied when draft_model is set"
                )
            main_vocab = len(tokenizer)
            draft_vocab = len(draft_tokenizer)
            if main_vocab != draft_vocab:
                raise ValueError(
                    f"vocab size mismatch: main tokenizer has {main_vocab} tokens "
                    f"but draft tokenizer has {draft_vocab} tokens. "
                    "Drafter must share the same tokenizer as the main model."
                )

        self._spec_stats: dict[str, int] = {
            "accepted_from_draft": 0,
            "total_tokens": 0,
        }

    def __call__(self, node: Node, inputs: dict) -> str:
        system = node.params.get("system_prompt", "")
        max_tokens = int(node.params.get("max_tokens", 64))
        temperature = float(node.params.get("temperature", 0.0))
        prompt = _format(system, inputs, self.tokenizer)
        cache = make_prompt_cache(self.model)
        sampler = make_sampler(temp=temperature)

        if self._draft_model is not None:
            # Speculative decoding path — accumulate per-token α stats.
            text_parts: list[str] = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                draft_model=self._draft_model,
                num_draft_tokens=self._num_draft_tokens,
                sampler=sampler,
                prompt_cache=cache,
            ):
                text_parts.append(response.text)
                self._spec_stats["total_tokens"] += 1
                if response.from_draft:
                    self._spec_stats["accepted_from_draft"] += 1
            return "".join(text_parts).strip()
        else:
            text = generate(self.model, self.tokenizer, prompt=prompt,
                            max_tokens=max_tokens, sampler=sampler,
                            prompt_cache=cache, verbose=False)
            return text.strip()

    def spec_decode_stats(self) -> dict:
        """Return speculative-decoding telemetry accumulated across all calls.

        Keys:
          enabled (bool): True when a draft model is configured.
          accepted_from_draft (int): Tokens accepted from the drafter so far.
          total_tokens (int): Total tokens generated so far.
          alpha (float): accepted_from_draft / total_tokens, or 0.0 if no tokens yet.
        """
        total = self._spec_stats["total_tokens"]
        accepted = self._spec_stats["accepted_from_draft"]
        return {
            "enabled": self._draft_model is not None,
            "accepted_from_draft": accepted,
            "total_tokens": total,
            "alpha": accepted / total if total > 0 else 0.0,
        }


def _format(system: str, inputs: dict, tokenizer: Any = None) -> str:
    body = "\n".join(f"{k}: {v}" for k, v in inputs.items())
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": body})
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    # Fallback: simple template
    return (f"<|system|>\n{system}\n<|user|>\n{body}\n<|assistant|>\n"
            if system else body)
