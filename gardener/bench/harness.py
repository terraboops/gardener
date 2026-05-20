"""Gardener bench harness — slim port of hypercar_bench's
agent-platform-relevant phases.

Phases implemented:
  smoke          - model loads + generates >=N tokens, non-empty output
  coherence      - "2 + 2 = " → "4"; "Capital of France is " → "Paris"
  niah           - Needle in a Haystack at configurable context length
  decode_speed   - tok/s over N=30 generated tokens after warm prefill
  prefill_speed  - tok/s for single prefill of fixed-length prompt
  memory_profile - Metal peak + swap delta; swap_mode classification

Reference map (NOT imported): omlx/bench/hypercar_bench.py.
"""
from __future__ import annotations

import gc
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("gardener.bench")


# ---------------------------------------------------------------------------
# System-memory helpers
# ---------------------------------------------------------------------------

def _system_ram_gb() -> float:
    """Total system RAM in GB (macOS sysctl)."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
        )
        return int(out.strip()) / (2**30)
    except Exception:
        logger.warning("Could not detect system RAM; defaulting to 16 GB")
        return 16.0


def _swap_used_gb() -> float:
    """Current swap in use (GB), parsed from sysctl vm.swapusage."""
    try:
        out = subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True, timeout=5
        )
        # Format: vm.swapusage: total = 3072.00M  used = 512.00M  free = 2560.00M  ...
        for part in out.split():
            if part.endswith("M") and out.split().index(part) > 0:
                label = out.split()[out.split().index(part) - 1]
                if label == "=":
                    # walk back one more to find "used"
                    pass
        # More reliable: find "used = NNN.NN[MG]"
        import re
        m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
        if m:
            val = float(m.group(1))
            return val / 1024.0 if m.group(2) == "M" else val
        return 0.0
    except Exception:
        return 0.0


def _metal_peak_gb() -> float:
    """Metal peak memory in GB (mx.metal / mx.get_peak_memory)."""
    try:
        import mlx.core as mx
        # Try the non-deprecated top-level API first
        fn = getattr(mx, "get_peak_memory", None)
        if fn is not None:
            return fn() / (2**30)
        # Fall back to mx.metal
        metal = getattr(mx, "metal", None)
        if metal is not None:
            fn2 = getattr(metal, "get_peak_memory", None)
            if fn2 is not None:
                return fn2() / (2**30)
    except Exception:
        pass
    logger.warning("mx.get_peak_memory not available; reporting 0.0 GB")
    return 0.0


def _metal_sync():
    """Synchronize Metal GPU so wall-clock timings are accurate."""
    try:
        import mlx.core as mx
        fn = getattr(mx.metal, "synchronize", None) if hasattr(mx, "metal") else None
        if fn is not None:
            fn()
        else:
            # Fallback: eval a small constant to flush the graph
            mx.eval(mx.array(0.0))
    except Exception:
        pass


def _clear_mlx_cache():
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _classify_swap_mode(swap_delta_gb: float) -> str:
    """Classify swap delta into mode per hypercar's discriminator.

    CLAUDE.md:46: swap < 5 GB → fast; > 8 GB → slow; else neutral.
    """
    if swap_delta_gb < 5.0:
        return "fast"
    if swap_delta_gb > 8.0:
        return "slow"
    return "neutral"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    name: str
    status: str           # "passed" | "failed" | "skipped"
    metrics: dict[str, Any]
    elapsed_s: float
    error: str | None = None


@dataclass
class BenchReport:
    model: str
    config: dict[str, Any]
    phases: list[PhaseResult]
    swap_mode: str        # "fast" | "slow" | "neutral"
    swap_delta_gb: float
    metal_peak_gb: float
    started_at: str       # iso8601
    elapsed_s: float

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self._to_dict(), indent=2, default=str))

    def _to_dict(self) -> dict:
        return {
            "model": self.model,
            "config": self.config,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "metrics": p.metrics,
                    "elapsed_s": round(p.elapsed_s, 3),
                    "error": p.error,
                }
                for p in self.phases
            ],
            "swap_mode": self.swap_mode,
            "swap_delta_gb": round(self.swap_delta_gb, 3),
            "metal_peak_gb": round(self.metal_peak_gb, 3),
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 3),
        }

    def passed(self) -> bool:
        return all(p.status == "passed" for p in self.phases)

    @property
    def gates_met(self) -> list[str]:
        return [p.name for p in self.phases if p.status == "passed"]

    @property
    def gates_failed(self) -> list[str]:
        return [p.name for p in self.phases if p.status == "failed"]


@dataclass
class BenchConfig:
    model_id: str
    phases: list[str] = field(default_factory=lambda: [
        "smoke", "coherence", "niah",
        "decode_speed", "prefill_speed", "memory_profile",
    ])
    niah_context_tokens: int = 4096
    decode_tokens: int = 30
    prefill_tokens: int = 4096
    gate_smoke_min_tokens: int = 4
    gate_coherence_min: int = 2
    gate_niah_required: bool = True
    gate_decode_min_tok_s: float = 5.0
    gate_prefill_min_tok_s: float = 50.0
    # HumanEval-Lite (HX12.1) — opt-in via phases list
    humaneval_n_problems: int | None = None   # None = all 20
    gate_humaneval_min_pass_rate: float = 0.35
    # MMLU-Pro (HX12.2) — opt-in via phases list
    mmlu_pro_n_questions: int = 25            # quick default; bump to 100 for full gate
    mmlu_pro_categories: list[str] | None = None  # None = all 14 categories
    gate_mmlu_pro_min_accuracy: float = 0.35


# ---------------------------------------------------------------------------
# Chat prompt helper (matching hypercar's _format_short_answer_prompt)
# ---------------------------------------------------------------------------

def _format_chat_prompt(tokenizer, user_message: str) -> str:
    """Apply chat template with thinking disabled if possible."""
    messages = [{"role": "user", "content": user_message}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return user_message + "\n"
    except Exception:
        return user_message + "\n"


# ---------------------------------------------------------------------------
# Core generate helper
# ---------------------------------------------------------------------------

def _eos_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for attr in ("eos_token_ids", "eos_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is None:
            continue
        if isinstance(v, (set, list, tuple)):
            ids.update(int(x) for x in v)
        else:
            ids.add(int(v))
    return ids


def _generate(
    model, tokenizer, prompt: str, max_tokens: int = 64, cache=None
) -> tuple[str, float, float]:
    """Generate and return (text, prefill_tok_s, decode_tok_s).

    Returns measured prefill and decode throughput. Metal GPU is
    synchronised around each timed section so wall-clock is accurate.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    n_layers = len(model.layers)

    if cache is None:
        from mlx_lm.models.cache import KVCache
        cache = [KVCache() for _ in range(n_layers)]

    # ---- prefill ----
    CHUNK = 4096
    _metal_sync()
    t0 = time.perf_counter()
    logits = None
    for cs in range(0, len(tokens), CHUNK):
        ce = min(cs + CHUNK, len(tokens))
        x = mx.array([tokens[cs:ce]])
        logits = model(x, cache=cache)
        mx.eval(logits)
    _metal_sync()
    prefill_time = time.perf_counter() - t0
    prefill_toks = len(tokens) / prefill_time if prefill_time > 0 else 0.0

    # ---- decode ----
    generated: list[int] = []
    eos_ids = _eos_token_ids(tokenizer)

    _metal_sync()
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tok_id = int(token.item())
        generated.append(tok_id)
        if tok_id in eos_ids:
            break
        x = token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)
    _metal_sync()
    decode_time = time.perf_counter() - t0
    decode_toks = len(generated) / decode_time if decode_time > 0 else 0.0

    text = tokenizer.decode(generated)

    del cache
    gc.collect()
    _clear_mlx_cache()

    return text, prefill_toks, decode_toks


def _fresh_cache(model):
    """Return a plain fp16 KV cache (one entry per layer)."""
    from mlx_lm.models.cache import KVCache
    return [KVCache() for _ in range(len(model.layers))]


# ---------------------------------------------------------------------------
# Phase: smoke
# ---------------------------------------------------------------------------

def _phase_smoke(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    t0 = time.perf_counter()
    try:
        prompt = _format_chat_prompt(tokenizer, "Hello, world!")
        text, prefill_toks, decode_toks = _generate(
            model, tokenizer, prompt, max_tokens=16
        )
        tokens_generated = len(tokenizer.encode(text))
        ok = tokens_generated >= cfg.gate_smoke_min_tokens and len(text.strip()) > 0
        status = "passed" if ok else "failed"
        err = None if ok else (
            f"only {tokens_generated} tokens generated (min {cfg.gate_smoke_min_tokens})"
        )
        return PhaseResult(
            name="smoke", status=status, elapsed_s=time.perf_counter() - t0,
            metrics={
                "tokens_generated": tokens_generated,
                "prefill_tok_s": round(prefill_toks, 1),
                "decode_tok_s": round(decode_toks, 1),
                "output_preview": text[:80],
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="smoke", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: coherence
# ---------------------------------------------------------------------------

def _phase_coherence(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    t0 = time.perf_counter()
    checks: list[dict] = []
    try:
        # Check 1: arithmetic
        p1 = _format_chat_prompt(tokenizer, "What is 2 + 2? Answer with just the number.")
        text1, _, _ = _generate(model, tokenizer, p1, max_tokens=32)
        checks.append({"name": "math", "prompt": "2+2=?", "output": text1[:80], "passed": "4" in text1})

        # Check 2: geography
        p2 = _format_chat_prompt(tokenizer, "What is the capital of France? Answer with just the city name.")
        text2, _, _ = _generate(model, tokenizer, p2, max_tokens=16)
        checks.append({"name": "geography", "prompt": "capital of France", "output": text2[:80], "passed": "Paris" in text2})

        n_passed = sum(1 for c in checks if c["passed"])
        ok = n_passed >= cfg.gate_coherence_min
        status = "passed" if ok else "failed"
        err = None if ok else f"only {n_passed}/{len(checks)} coherence checks passed"
        return PhaseResult(
            name="coherence", status=status, elapsed_s=time.perf_counter() - t0,
            metrics={"checks": checks, "passed_count": n_passed, "total": len(checks)},
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="coherence", status="failed", metrics={"checks": checks},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: niah (Needle in a Haystack)
# ---------------------------------------------------------------------------

def _build_haystack(tokenizer, target_tokens: int, needle: str, needle_depth: float = 0.5) -> str:
    """Build a synthetic haystack of ~target_tokens tokens with needle at ~depth."""
    filler_unit = "The quick brown fox jumps over the lazy dog. "
    # Estimate tokens per filler unit
    unit_tokens = len(tokenizer.encode(filler_unit))
    needle_tokens = len(tokenizer.encode(needle))
    remaining = target_tokens - needle_tokens
    n_units = max(1, remaining // unit_tokens)

    before_count = int(n_units * needle_depth)
    after_count = n_units - before_count

    haystack = (filler_unit * before_count) + needle + " " + (filler_unit * after_count)
    return haystack


def _phase_niah(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    t0 = time.perf_counter()
    try:
        import mlx.core as mx

        timestamp = int(t0 * 1000) % 100000  # short but unique enough
        needle_phrase = f"gardener-{timestamp}"
        needle = f"The secret password is {needle_phrase}."

        haystack = _build_haystack(tokenizer, cfg.niah_context_tokens, needle)
        query = "What is the secret password?"

        # Build the full prompt: haystack context + retrieval question
        context_msg = f"{haystack}\n\n{query}"
        prompt = _format_chat_prompt(tokenizer, context_msg)

        prompt_tokens = tokenizer.encode(prompt)
        n_prompt = len(prompt_tokens)

        # Prefill
        cache = _fresh_cache(model)
        CHUNK = 4096
        _metal_sync()
        t_pre = time.perf_counter()
        logits = None
        for cs in range(0, len(prompt_tokens), CHUNK):
            ce = min(cs + CHUNK, len(prompt_tokens))
            x = mx.array([prompt_tokens[cs:ce]])
            logits = model(x, cache=cache)
            mx.eval(logits)
        _metal_sync()
        prefill_time = time.perf_counter() - t_pre
        prefill_toks = n_prompt / prefill_time if prefill_time > 0 else 0.0

        # Decode answer
        eos_ids = _eos_token_ids(tokenizer)
        generated: list[int] = []
        _metal_sync()
        t_dec = time.perf_counter()
        for _ in range(64):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_id = int(token.item())
            generated.append(tok_id)
            if tok_id in eos_ids:
                break
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)
        _metal_sync()
        decode_time = time.perf_counter() - t_dec
        decode_toks = len(generated) / decode_time if decode_time > 0 else 0.0

        answer = tokenizer.decode(generated)
        retrieved = needle_phrase in answer

        del cache
        gc.collect()
        _clear_mlx_cache()

        ok = retrieved if cfg.gate_niah_required else True
        status = "passed" if ok else "failed"
        err = None if ok else f"needle '{needle_phrase}' not found in output: {answer[:120]!r}"

        return PhaseResult(
            name="niah", status=status, elapsed_s=time.perf_counter() - t0,
            metrics={
                "context_tokens": n_prompt,
                "target_tokens": cfg.niah_context_tokens,
                "needle": needle_phrase,
                "retrieved": retrieved,
                "answer_preview": answer[:120],
                "prefill_tok_s": round(prefill_toks, 1),
                "decode_tok_s": round(decode_toks, 1),
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="niah", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: decode_speed
# ---------------------------------------------------------------------------

def _phase_decode_speed(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    """Measure decode tok/s over N tokens starting from a short warm cache."""
    t0 = time.perf_counter()
    try:
        import mlx.core as mx

        # Warm prefill with a short prompt so cache is populated
        warm_prompt = _format_chat_prompt(tokenizer, "Continue the following story:")
        warm_tokens = tokenizer.encode(warm_prompt)
        cache = _fresh_cache(model)

        x = mx.array([warm_tokens])
        logits = model(x, cache=cache)
        mx.eval(logits)

        # Now decode cfg.decode_tokens tokens and time it
        eos_ids = _eos_token_ids(tokenizer)
        generated: list[int] = []

        _metal_sync()
        td = time.perf_counter()
        for _ in range(cfg.decode_tokens):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_id = int(token.item())
            generated.append(tok_id)
            if tok_id in eos_ids:
                break
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)
        _metal_sync()
        decode_time = time.perf_counter() - td

        n_generated = len(generated)
        tok_s = n_generated / decode_time if decode_time > 0 else 0.0

        del cache
        gc.collect()
        _clear_mlx_cache()

        ok = tok_s >= cfg.gate_decode_min_tok_s
        status = "passed" if ok else "failed"
        err = None if ok else f"decode {tok_s:.1f} tok/s < {cfg.gate_decode_min_tok_s} tok/s gate"

        return PhaseResult(
            name="decode_speed", status=status, elapsed_s=time.perf_counter() - t0,
            metrics={
                "decode_tok_s": round(tok_s, 1),
                "tokens_generated": n_generated,
                "decode_time_s": round(decode_time, 3),
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="decode_speed", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: prefill_speed
# ---------------------------------------------------------------------------

def _phase_prefill_speed(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    """Measure prefill tok/s for a prompt of ~cfg.prefill_tokens tokens."""
    t0 = time.perf_counter()
    try:
        import mlx.core as mx

        filler = "The quick brown fox jumps over the lazy dog. "
        # Build a prompt that hits cfg.prefill_tokens tokens
        unit_toks = len(tokenizer.encode(filler))
        n_units = max(1, cfg.prefill_tokens // unit_toks)
        long_text = filler * n_units
        prompt = _format_chat_prompt(tokenizer, long_text)
        tokens = tokenizer.encode(prompt)
        n_tokens = len(tokens)

        cache = _fresh_cache(model)
        CHUNK = 4096

        _metal_sync()
        tp = time.perf_counter()
        logits = None
        for cs in range(0, n_tokens, CHUNK):
            ce = min(cs + CHUNK, n_tokens)
            x = mx.array([tokens[cs:ce]])
            logits = model(x, cache=cache)
            mx.eval(logits)
        _metal_sync()
        prefill_time = time.perf_counter() - tp
        tok_s = n_tokens / prefill_time if prefill_time > 0 else 0.0

        del logits, cache
        gc.collect()
        _clear_mlx_cache()

        ok = tok_s >= cfg.gate_prefill_min_tok_s
        status = "passed" if ok else "failed"
        err = None if ok else f"prefill {tok_s:.1f} tok/s < {cfg.gate_prefill_min_tok_s} tok/s gate"

        return PhaseResult(
            name="prefill_speed", status=status, elapsed_s=time.perf_counter() - t0,
            metrics={
                "prefill_tok_s": round(tok_s, 1),
                "prompt_tokens": n_tokens,
                "prefill_time_s": round(prefill_time, 3),
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="prefill_speed", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: memory_profile
# ---------------------------------------------------------------------------

def _phase_memory_profile(swap_at_start: float) -> PhaseResult:
    """Snapshot Metal peak and swap delta; classify swap_mode."""
    t0 = time.perf_counter()
    try:
        metal_peak = _metal_peak_gb()
        swap_now = _swap_used_gb()
        swap_delta = max(0.0, swap_now - swap_at_start)
        mode = _classify_swap_mode(swap_delta)
        return PhaseResult(
            name="memory_profile", status="passed", elapsed_s=time.perf_counter() - t0,
            metrics={
                "metal_peak_gb": round(metal_peak, 2),
                "swap_delta_gb": round(swap_delta, 3),
                "swap_at_start_gb": round(swap_at_start, 3),
                "swap_now_gb": round(swap_now, 3),
                "swap_mode": mode,
            },
        )
    except Exception as e:
        return PhaseResult(
            name="memory_profile", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: humaneval_lite (HX12.1)
# ---------------------------------------------------------------------------

def _phase_humaneval_lite(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    """Run the HumanEval-Lite quality gate (20 problems, exec+assert).

    Opt-in only — not in the default phases list because each problem takes
    seconds on a production model. Enable via phases=['humaneval_lite'] or
    --phases smoke,coherence,humaneval_lite on the CLI.
    """
    t0 = time.perf_counter()
    try:
        from gardener.bench.quality import run_humaneval_lite

        result = run_humaneval_lite(
            model, tokenizer,
            n_problems=cfg.humaneval_n_problems,
            gate_min=cfg.gate_humaneval_min_pass_rate,
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
        status = "passed" if result.passed_gate else "failed"
        err = (
            None if result.passed_gate
            else (
                f"pass_rate {result.pass_rate:.0%} < gate {result.gate_min:.0%} "
                f"({result.n_passed}/{result.n_problems})"
            )
        )
        return PhaseResult(
            name="humaneval_lite", status=status,
            elapsed_s=time.perf_counter() - t0,
            metrics={
                "pass_rate": round(result.pass_rate, 4),
                "n_passed": result.n_passed,
                "n_problems": result.n_problems,
                "gate_min": result.gate_min,
                "per_problem": result.per_problem,
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="humaneval_lite", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Phase: mmlu_pro (HX12.2)
# ---------------------------------------------------------------------------

def _phase_mmlu_pro(model, tokenizer, cfg: BenchConfig) -> PhaseResult:
    """Run the MMLU-Pro quality gate (CoT, 3-stage regex extraction).

    Default 25 questions (quick). Bump mmlu_pro_n_questions to 100 for the
    full hypercar gate (≥35% accuracy). enable_thinking=False is critical —
    thinking blocks consume the token budget before the answer emerges,
    collapsing Qwen3.6 from ~62% to ~22%.

    Opt-in only — not in the default phases list. Enable via
    phases=['mmlu_pro'] or --phases smoke,coherence,mmlu_pro on the CLI.
    """
    t0 = time.perf_counter()
    try:
        from gardener.bench.quality import run_mmlu_pro

        result = run_mmlu_pro(
            model, tokenizer,
            n_questions=cfg.mmlu_pro_n_questions,
            categories=cfg.mmlu_pro_categories,
            gate_min=cfg.gate_mmlu_pro_min_accuracy,
            enable_thinking=False,
            verbose=logger.isEnabledFor(logging.DEBUG),
        )
        status = "passed" if result.passed_gate else "failed"
        err = (
            None if result.passed_gate
            else (
                f"accuracy {result.accuracy:.0%} < gate {result.gate_min:.0%} "
                f"({result.n_correct}/{result.n_questions})"
            )
        )
        return PhaseResult(
            name="mmlu_pro", status=status,
            elapsed_s=time.perf_counter() - t0,
            metrics={
                "accuracy": round(result.accuracy, 4),
                "n_correct": result.n_correct,
                "n_questions": result.n_questions,
                "gate_min": result.gate_min,
                "by_category": result.by_category,
            },
            error=err,
        )
    except Exception as e:
        return PhaseResult(
            name="mmlu_pro", status="failed", metrics={},
            elapsed_s=time.perf_counter() - t0, error=str(e),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_PHASE_FNS = {
    "smoke": _phase_smoke,
    "coherence": _phase_coherence,
    "niah": _phase_niah,
    "decode_speed": _phase_decode_speed,
    "prefill_speed": _phase_prefill_speed,
    "humaneval_lite": _phase_humaneval_lite,
    "mmlu_pro": _phase_mmlu_pro,
}


def run_bench(config: BenchConfig) -> BenchReport:
    """Run the bench harness and return a BenchReport.

    Failure of any phase halts the pipeline and writes a partial report.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t_run = time.perf_counter()

    # Snapshot swap before any model work
    swap_at_start = _swap_used_gb()

    # Load model
    logger.info(f"Loading model {config.model_id!r} …")
    from mlx_lm import load
    model, tokenizer = load(config.model_id)
    logger.info("Model loaded.")

    phases_done: list[PhaseResult] = []

    for phase_name in config.phases:
        if phase_name == "memory_profile":
            # memory_profile runs at the end (or in-sequence if requested)
            result = _phase_memory_profile(swap_at_start)
        elif phase_name in _PHASE_FNS:
            fn = _PHASE_FNS[phase_name]
            result = fn(model, tokenizer, config)
        else:
            result = PhaseResult(
                name=phase_name, status="skipped",
                metrics={}, elapsed_s=0.0,
                error=f"unknown phase {phase_name!r}",
            )

        logger.info(
            f"  [{result.status.upper()}] {phase_name}  "
            f"({result.elapsed_s:.1f}s)"
            + (f"  error={result.error}" if result.error else "")
        )
        phases_done.append(result)

        # Halt on failure
        if result.status == "failed":
            logger.warning(f"Phase {phase_name!r} failed — halting pipeline.")
            break

    # Final memory snapshot
    metal_peak = _metal_peak_gb()
    swap_now = _swap_used_gb()
    swap_delta = max(0.0, swap_now - swap_at_start)
    swap_mode = _classify_swap_mode(swap_delta)

    # Override metal_peak / swap_mode from memory_profile phase if present
    for p in phases_done:
        if p.name == "memory_profile" and p.status == "passed":
            metal_peak = p.metrics.get("metal_peak_gb", metal_peak)
            swap_delta = p.metrics.get("swap_delta_gb", swap_delta)
            swap_mode = p.metrics.get("swap_mode", swap_mode)
            break

    return BenchReport(
        model=config.model_id,
        config={
            "phases": config.phases,
            "niah_context_tokens": config.niah_context_tokens,
            "decode_tokens": config.decode_tokens,
            "prefill_tokens": config.prefill_tokens,
        },
        phases=phases_done,
        swap_mode=swap_mode,
        swap_delta_gb=round(swap_delta, 3),
        metal_peak_gb=round(metal_peak, 3),
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - t_run, 3),
    )
