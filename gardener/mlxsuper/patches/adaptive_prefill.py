# SPDX-License-Identifier: Apache-2.0
"""Adaptive prefill chunk-size controller with memory-aware feedback.

Replaces the fixed prefill_step_size with a proportional controller that
adjusts chunk size based on Metal memory pressure and throughput.

At the start of prefill (KV cache small), uses large chunks for maximum
throughput. As the KV cache grows and Metal pressure builds, shrinks chunks
to avoid swap thrashing. If tok/s drops below a floor (O(n²) attention
cliff), shrinks aggressively.

Derived from Memory-aware Dynamic Batching (arXiv:2503.05248).
512K-validated upstream (NIAH PASS, 47 GB Metal peak, Qwen3.6-35B-A3B-4bit,
2026-05-03).

Reference map (NOT imported): omlx/patches/adaptive_prefill.py:1-217
in omlx-mamba3. Re-derived on stock mlx_lm; no omlx import.

Usage::

    from gardener.mlxsuper.patches.adaptive_prefill import apply_adaptive_prefill
    apply_adaptive_prefill()  # monkey-patches mlx_lm.generate.generate_step
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger("gardener.mlxsuper.patches.adaptive_prefill")

# ---------------------------------------------------------------------------
# Module-level defaults (exposed as public constant per HX2 spec)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "target_metal_pct": 0.65,
    "min_chunk": 512,
    "max_chunk": 16384,
    "throughput_floor": 200.0,
    "min_prompt_tokens": 1024,
}

# ---------------------------------------------------------------------------
# System RAM detection (cached at import time)
# ---------------------------------------------------------------------------
def _detect_system_ram_gb() -> float:
    """Return system RAM in GB via sysctl; falls back to 48 GB (M4 Pro)."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], timeout=2, text=True
        )
        return int(out.strip()) / 1e9
    except Exception:
        return 48.0


SYSTEM_MEMORY_GB: float = _detect_system_ram_gb()

# ---------------------------------------------------------------------------
# Patch-state sentinel
# ---------------------------------------------------------------------------
_PATCHED = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class AdaptivePrefillController:
    """Proportional controller for prefill chunk sizing.

    Two signals, one output:
    - Metal memory residency → shrink when above target
    - Prefill throughput     → shrink when below floor (O(n²) cliff)
    - Output: next chunk size clamped to [min_chunk, max_chunk]

    The controller starts at max_chunk and adjusts per iteration.

    Args:
        target_metal_pct: Target Metal usage as fraction of system RAM.
            Default: 0.65.
        min_chunk: Minimum chunk size (floor). Default: 512.
        max_chunk: Maximum chunk size (ceiling and starting point).
            Default: 16384.
        throughput_floor: tok/s below which the cliff is detected and the
            chunk is halved. Default: 200.0.
        min_prompt_tokens: Prompts shorter than this pass through unchanged.
            Stored as an attribute so callers (e.g. the patch wrapper) can
            read it. Default: 1024.
        system_memory_gb: Override system RAM (useful for tests). Defaults to
            the module-level auto-detected value.
    """

    def __init__(
        self,
        target_metal_pct: float = DEFAULTS["target_metal_pct"],
        min_chunk: int = DEFAULTS["min_chunk"],
        max_chunk: int = DEFAULTS["max_chunk"],
        throughput_floor: float = DEFAULTS["throughput_floor"],
        min_prompt_tokens: int = DEFAULTS["min_prompt_tokens"],
        system_memory_gb: float = SYSTEM_MEMORY_GB,
    ) -> None:
        self.target_metal_pct: float = target_metal_pct
        self.min_chunk: int = min_chunk
        self.max_chunk: int = max_chunk
        self.throughput_floor: float = throughput_floor
        self.min_prompt_tokens: int = min_prompt_tokens

        self._target_metal_gb: float = system_memory_gb * target_metal_pct
        self._chunk_size: int = max_chunk
        self._chunks_seen: int = 0
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_chunk_size(self, remaining: int) -> int:
        """Return the chunk size for the next prefill iteration.

        Always ≤ remaining and ≥ min_chunk (unless remaining itself is
        smaller than min_chunk).
        """
        return min(self._chunk_size, remaining)

    def feedback(self, metal_gb: float, tok_per_sec: float) -> None:
        """Update controller state after a chunk completes.

        Args:
            metal_gb: Current Metal active memory in GB.
            tok_per_sec: Prefill throughput for the chunk just completed.
        """
        metal_ratio = metal_gb / max(self._target_metal_gb, 0.1)
        prev = self._chunk_size

        if tok_per_sec < self.throughput_floor and tok_per_sec > 0:
            # Throughput cliff detected (O(n²) attention) — halve aggressively.
            self._chunk_size = max(self.min_chunk, self._chunk_size // 2)
        elif metal_ratio > 1.0:
            # Over memory target — shrink proportionally.
            factor = max(0.5, 1.0 / metal_ratio)
            self._chunk_size = max(self.min_chunk, int(self._chunk_size * factor))
        elif metal_ratio < 0.8:
            # Under target with headroom — grow gradually.
            self._chunk_size = min(self.max_chunk, int(self._chunk_size * 1.25))
        # else: sweet spot [0.8, 1.0] of target — hold steady.

        self._chunks_seen += 1
        self._history.append({
            "chunk_size": self._chunk_size,
            "prev_chunk": prev,
            "metal_gb": round(metal_gb, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "metal_ratio": round(metal_ratio, 3),
        })

    def summary(self) -> dict:
        """Return a summary dict of controller state for profiling/logging.

        Always includes the keys required by the HX2 spec:
        current_chunk, chunks_seen, min_chunk, max_chunk,
        target_metal_pct, throughput_floor.
        """
        base: dict[str, Any] = {
            "current_chunk": self._chunk_size,
            "chunks_seen": self._chunks_seen,
            "min_chunk": self.min_chunk,
            "max_chunk": self.max_chunk,
            "target_metal_pct": self.target_metal_pct,
            "throughput_floor": self.throughput_floor,
        }
        if self._history:
            sizes = [h["chunk_size"] for h in self._history]
            base.update({
                "chunk_min_seen": min(sizes),
                "chunk_max_seen": max(sizes),
                "chunk_final": sizes[-1],
                "metal_peak_gb": max(h["metal_gb"] for h in self._history),
                "throughput_min": min(h["tok_per_sec"] for h in self._history),
            })
        return base

    def reset(self) -> None:
        """Reset per-request state (chunk size returns to max_chunk)."""
        self._chunk_size = self.max_chunk
        self._chunks_seen = 0
        self._history = []


# ---------------------------------------------------------------------------
# Monkey-patch
# ---------------------------------------------------------------------------

def _build_patched_generate_step(
    original_generate_step,
    controller_kwargs: dict,
):
    """Return a replacement for mlx_lm.generate.generate_step that drives
    an AdaptivePrefillController for prompts >= min_prompt_tokens.
    """
    import importlib
    _cache_mod = importlib.import_module("mlx_lm.models.cache")

    def _patched_generate_step(
        prompt: mx.array,
        model,
        *,
        max_tokens: int = 256,
        sampler=None,
        logits_processors=None,
        max_kv_size: Optional[int] = None,
        prompt_cache=None,
        prefill_step_size: int = 2048,
        kv_bits=None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prompt_progress_callback=None,
        input_embeddings=None,
    ):
        prompt_len = prompt.shape[0] if hasattr(prompt, "shape") else len(prompt)
        min_prompt_tokens = controller_kwargs.get(
            "min_prompt_tokens", DEFAULTS["min_prompt_tokens"]
        )

        # Short prompts: pass through unchanged — no benefit from adaptive chunking.
        if prompt_len < min_prompt_tokens:
            yield from original_generate_step(
                prompt, model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                prompt_progress_callback=prompt_progress_callback,
                input_embeddings=input_embeddings,
            )
            return

        # Long prompt: run adaptive prefill for all but the last token, then
        # hand off to original generate_step with the pre-filled cache for
        # decode.  The original generate_step will find the cache already
        # populated and its own prefill loop will be a single-step no-op.
        if prompt_cache is None:
            prompt_cache = _cache_mod.make_prompt_cache(model, max_kv_size=max_kv_size)

        controller = AdaptivePrefillController(
            target_metal_pct=controller_kwargs.get(
                "target_metal_pct", DEFAULTS["target_metal_pct"]
            ),
            min_chunk=controller_kwargs.get("min_chunk", DEFAULTS["min_chunk"]),
            max_chunk=controller_kwargs.get("max_chunk", DEFAULTS["max_chunk"]),
            throughput_floor=controller_kwargs.get(
                "throughput_floor", DEFAULTS["throughput_floor"]
            ),
            min_prompt_tokens=min_prompt_tokens,
        )

        if not isinstance(prompt, mx.array):
            prompt = mx.array(prompt)

        # Prefill all tokens except the last (the last one is consumed by
        # generate_step's own _step call which also does sampling).
        tokens_to_prefill = prompt_len - 1
        processed = 0

        while processed < tokens_to_prefill:
            remaining = tokens_to_prefill - processed
            chunk_size = controller.next_chunk_size(remaining)

            chunk = prompt[processed: processed + chunk_size]
            t0 = time.perf_counter()

            model(chunk[None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])

            elapsed = time.perf_counter() - t0
            tok_per_sec = chunk_size / max(elapsed, 1e-6)
            metal_gb = mx.get_active_memory() / 1e9

            controller.feedback(metal_gb, tok_per_sec)
            processed += chunk_size
            mx.clear_cache()

        s = controller.summary()
        logger.info(
            "Adaptive prefill: %d tokens, %d chunks "
            "(chunk range %s-%s), Metal peak %.1f GB",
            prompt_len,
            s["chunks_seen"],
            s.get("chunk_min_seen", controller.min_chunk),
            s.get("chunk_max_seen", controller.max_chunk),
            s.get("metal_peak_gb", 0.0),
        )

        # Hand off to original generate_step for decode only.
        # The pre-filled cache means its prefill loop sees only 1 token
        # (prompt[-1:]) and proceeds directly to decode.
        yield from original_generate_step(
            prompt[-1:], model,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            max_kv_size=max_kv_size,
            prompt_cache=prompt_cache,
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            prompt_progress_callback=prompt_progress_callback,
            input_embeddings=None,  # consumed by prefill above if provided
        )

    return _patched_generate_step


def apply_adaptive_prefill(**controller_kwargs) -> bool:
    """Monkey-patch mlx_lm.generate.generate_step to use adaptive prefill.

    For prompts >= min_prompt_tokens, runs a custom prefill loop with
    memory-aware chunk sizing before handing off to the original
    generate_step for decode.  Short prompts pass through unchanged.

    Args:
        **controller_kwargs: Forwarded to AdaptivePrefillController.
            Supported keys: target_metal_pct, min_chunk, max_chunk,
            throughput_floor, min_prompt_tokens.

    Returns:
        True if the patch was applied; False if already patched (idempotent).
    """
    global _PATCHED
    if _PATCHED:
        return False

    import importlib
    _gen_mod = importlib.import_module("mlx_lm.generate")
    _original = _gen_mod.generate_step

    patched = _build_patched_generate_step(_original, controller_kwargs)
    _gen_mod.generate_step = patched

    # Also patch server's reference if already imported.
    try:
        import mlx_lm.server as _srv
        _srv.generate_step = patched
    except ImportError:
        pass

    _PATCHED = True

    target = controller_kwargs.get("target_metal_pct", DEFAULTS["target_metal_pct"])
    min_c = controller_kwargs.get("min_chunk", DEFAULTS["min_chunk"])
    max_c = controller_kwargs.get("max_chunk", DEFAULTS["max_chunk"])
    floor = controller_kwargs.get("throughput_floor", DEFAULTS["throughput_floor"])
    logger.info(
        "Adaptive prefill enabled: target=%.0f%% Metal, chunks %d-%d, "
        "floor=%.0f tok/s, system RAM=%.1f GB",
        target * 100, min_c, max_c, floor, SYSTEM_MEMORY_GB,
    )
    return True


def is_patched() -> bool:
    """Return True if apply_adaptive_prefill() has been called."""
    return _PATCHED
