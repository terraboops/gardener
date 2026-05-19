"""MLX-backed agent runner. Each agent name maps to a persistent SessionPool
session so successive calls reuse the KV cache (warm context).

For v0 simplicity, the system prompt is sent fresh each call (no incremental
chat history beyond the cache). The next iteration will wire conversation
history through cache reuse + rewind."""
from __future__ import annotations

from typing import Any

from mlx_lm import generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from ..pipeline.ir import Node


class MLXAgentRunner:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, node: Node, inputs: dict) -> str:
        system = node.params.get("system_prompt", "")
        max_tokens = int(node.params.get("max_tokens", 64))
        temperature = float(node.params.get("temperature", 0.0))
        prompt = _format(system, inputs, self.tokenizer)
        cache = make_prompt_cache(self.model)
        sampler = make_sampler(temp=temperature)
        text = generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=max_tokens, sampler=sampler,
                        prompt_cache=cache, verbose=False)
        return text.strip()


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
