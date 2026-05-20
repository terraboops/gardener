"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable.

HX-tier extensions (ported from hypercar — never imported):
- KV caches: DuoKVCache (best-quality default), TurboQuantKVCache (TQ3, agentic).
- SnapKV eviction stack: snapkv_select + compact_cache + SnapKVOptions.
- Hybrid attention helpers: attention_layer_indices, is_hybrid, eos_token_ids,
  format_chat — Qwen3.6 enabler.
- Patches: apply_prefill_last_logit_patch (>8K), apply_adaptive_prefill (512K),
  apply_minference_prefill_patch (sparse; calibration placeholder),
  TTTHeadRouter (bit-equivalence mode).
"""
from .duo_kv_cache import DuoKVCache, StreamingKVCache, load_duo_policy
from .hybrid import (
    attention_layer_indices,
    eos_token_ids,
    format_chat,
    is_hybrid,
)
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .patches.adaptive_prefill import (
    AdaptivePrefillController,
    apply_adaptive_prefill,
)
from .patches.minference_prefill import (
    apply_minference_prefill_patch,
    load_pattern_table,
)
from .patches.prefill_last_logit import apply_prefill_last_logit_patch
from .patches.ttt_head_router import TTTHeadRouter
from .schedules import compute_layer_lrs
from .session import SessionPool
from .snapkv import SnapKVOptions, compact_cache, snapkv_select
from .ttt import Candidate, TrainStats, TTTEngine
from .turboquant_kv import TurboQuantKVCache

__all__ = [
    # MVP foundation
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
    # HX3 — agentic 3-bit KV
    "TurboQuantKVCache",
    # HX4 — best-quality KV mode
    "DuoKVCache", "StreamingKVCache", "load_duo_policy",
    # HX5 — long-context eviction
    "SnapKVOptions", "snapkv_select", "compact_cache",
    # HX7 — hybrid models (Qwen3.6)
    "attention_layer_indices", "is_hybrid", "eos_token_ids", "format_chat",
    # HX1/HX2/HX6/HX8 — patches
    "apply_prefill_last_logit_patch",
    "apply_adaptive_prefill", "AdaptivePrefillController",
    "load_pattern_table", "apply_minference_prefill_patch",
    "TTTHeadRouter",
]
