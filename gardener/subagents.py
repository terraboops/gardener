"""Pre-cached subagent registry: prefill a profile's warmup text once,
save the KV cache to disk, then on each dispatch load+fork that warmed
cache so we skip re-prefilling the persistent context. Each call gets a
fresh independent fork — discarded after use."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mlx_lm import generate, stream_generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.sample_utils import make_sampler

from .mlxsuper.session import SessionPool
from .pipeline.ir import Node


@dataclass
class AgentProfile:
    name: str
    system_prompt: str
    warmup_text: str               # text prefilled into the cache
    params: dict = field(default_factory=dict)   # max_tokens, temperature, …
    draft_model: Optional[str] = None  # HF id of a smaller drafter (HX9)
    num_draft_tokens: int = 8          # speculative decoding lookahead (HX9)


class SubagentRegistry:
    """Registry of named AgentProfiles whose warm caches live under root."""

    def __init__(self, model: Any, tokenizer: Any, root: str | Path):
        self.model = model
        self.tokenizer = tokenizer
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, AgentProfile] = {}
        # Lazy cache: HF id -> (draft_model, draft_tokenizer).  Loaded once
        # on first runner() call that needs that drafter (HX9).
        self._drafters: dict[str, tuple[Any, Any]] = {}

    def _cache_path(self, name: str) -> Path:
        return self.root / f"{name}.safetensors"

    def register(self, profile: AgentProfile) -> Path:
        """Prefill warmup_text into a fresh prompt cache; save to disk.
        Returns the cache path. Idempotent: existing cache is overwritten
        only if the warmup_text changed (tracked via a .meta sidecar)."""
        self._profiles[profile.name] = profile
        meta_path = self.root / f"{profile.name}.meta"
        if (self._cache_path(profile.name).exists()
                and meta_path.exists()
                and meta_path.read_text() == profile.warmup_text):
            return self._cache_path(profile.name)

        cache = make_prompt_cache(self.model)
        # Run a 1-token generation to populate the cache with the warmup text.
        # Greedy, deterministic, minimal.
        _ = generate(self.model, self.tokenizer,
                      prompt=profile.warmup_text, max_tokens=1,
                      sampler=make_sampler(temp=0.0),
                      prompt_cache=cache, verbose=False)
        path = self._cache_path(profile.name)
        save_prompt_cache(str(path), cache, {})
        meta_path.write_text(profile.warmup_text)
        return path

    def has(self, name: str) -> bool:
        return self._cache_path(name).exists()

    def runner(self, profile_name: str) -> Callable[[Node, dict], str]:
        """Return an AgentRunner closure that, on each call:
           1. Loads the warm cache from disk into a SessionPool session.
           2. Forks the session (independent branch).
           3. Generates a completion with the fork's cache.
           4. Discards the fork (no save back).

        The system_prompt + inputs format follows MLXAgentRunner's convention
        (use the tokenizer's chat template if available)."""
        if profile_name not in self._profiles:
            raise KeyError(f"unknown profile: {profile_name!r}")
        if not self.has(profile_name):
            raise FileNotFoundError(f"no warm cache for {profile_name!r}; "
                                     "call register() first")

        profile = self._profiles[profile_name]
        cache_path = self._cache_path(profile_name)

        def _format_user(inputs: dict) -> str:
            body = "\n".join(f"{k}: {v}" for k, v in inputs.items())
            if hasattr(self.tokenizer, "apply_chat_template"):
                msgs = []
                if profile.system_prompt:
                    msgs.append({"role": "system",
                                  "content": profile.system_prompt})
                msgs.append({"role": "user", "content": body})
                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            return (f"<|system|>\n{profile.system_prompt}\n<|user|>\n{body}"
                    f"\n<|assistant|>\n" if profile.system_prompt else body)

        # Lazily load drafter once for this profile (HX9).
        draft_model_obj: Any = None
        if profile.draft_model is not None:
            if profile.draft_model not in self._drafters:
                from mlx_lm import load as _mlx_load
                dm, dt = _mlx_load(profile.draft_model)
                # Vocab check — mismatch raises immediately so misconfig
                # surfaces at runner() time, not silently mid-request.
                main_vocab = len(self.tokenizer)
                draft_vocab = len(dt)
                if main_vocab != draft_vocab:
                    raise ValueError(
                        f"vocab size mismatch for drafter {profile.draft_model!r}: "
                        f"main={main_vocab}, draft={draft_vocab}. "
                        "Drafter must share the same tokenizer as the main model."
                    )
                self._drafters[profile.draft_model] = (dm, dt)
            draft_model_obj, _ = self._drafters[profile.draft_model]

        def runner(node: Node, inputs: dict) -> str:
            # Load warm cache + fork in a SessionPool.
            pool = SessionPool(cache_factory=lambda: load_prompt_cache(str(cache_path)))
            sid = pool.create(uuid.uuid4().hex[:8])
            fork_id = pool.fork(sid)
            fork_cache = pool.get(fork_id)
            max_tokens = int(profile.params.get("max_tokens",
                                                  node.params.get("max_tokens", 64)))
            temperature = float(profile.params.get("temperature",
                                                     node.params.get("temperature", 0.0)))
            sampler = make_sampler(temp=temperature)
            prompt = _format_user(inputs)

            if draft_model_obj is not None:
                # Speculative decoding path (HX9).
                text_parts: list[str] = []
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    draft_model=draft_model_obj,
                    num_draft_tokens=profile.num_draft_tokens,
                    sampler=sampler,
                    prompt_cache=fork_cache,
                ):
                    text_parts.append(response.text)
                return "".join(text_parts).strip()
            else:
                text = generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=max_tokens, sampler=sampler,
                                 prompt_cache=fork_cache, verbose=False)
                return text.strip()

        return runner
