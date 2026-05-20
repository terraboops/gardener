"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable.

HX-tier extensions ported from hypercar:
- mlxsuper.duo_kv_cache: DuoKVCache (HX4) — fp16 streaming + 3-bit retrieval per-head.
- mlxsuper.turboquant_kv: TurboQuantKVCache (HX3) — WHT + Beta codebook, agentic save/load/fork.
- mlxsuper.snapkv: snapkv_select + compact_cache + SnapKVOptions (HX5) — attention-guided eviction.
- mlxsuper.hybrid: attention_layer_indices, is_hybrid, eos_token_ids, format_chat (HX7).
- mlxsuper.patches.prefill_last_logit (HX1) — last-token-only lm_head projection during prefill.
- mlxsuper.patches.adaptive_prefill (HX2) — memory-aware chunk controller (512K-validated).
- mlxsuper.patches.minference_prefill (HX6) — per-(layer, head) sparse prefill; calibration placeholder.
- mlxsuper.patches.ttt_head_router (HX8) — bit-equivalence mode router; Cycle 2 upstream-pending.
"""
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .schedules import compute_layer_lrs
from .session import SessionPool
from .ttt import Candidate, TrainStats, TTTEngine

# HX3 — TurboQuantKVCache
from .turboquant_kv import TurboQuantKVCache

# HX4 — DuoKVCache + policy loader
from .duo_kv_cache import DuoKVCache, StreamingKVCache, load_duo_policy

# HX5 — SnapKV eviction
from .snapkv import snapkv_select, compact_cache, SnapKVOptions

# HX7 — hybrid attention support
from .hybrid import (
    attention_layer_indices, is_hybrid, eos_token_ids, format_chat,
)

# HX1/HX2/HX6/HX8 — patches
from .patches.prefill_last_logit import apply_prefill_last_logit_patch
from .patches.adaptive_prefill import (
    apply_adaptive_prefill, AdaptivePrefillController,
)
from .patches.minference_prefill import (
    load_pattern_table, apply_minference_prefill_patch,
)
from .patches.ttt_head_router import TTTHeadRouter

__all__ = [
    # MVP foundation
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
    # HX3 — agentic KV
    "TurboQuantKVCache",
    # HX4 — best-quality KV mode
    "DuoKVCache", "StreamingKVCache", "load_duo_policy",
    # HX5 — long-context eviction
    "snapkv_select", "compact_cache", "SnapKVOptions",
    # HX7 — hybrid models (Qwen3.6)
    "attention_layer_indices", "is_hybrid", "eos_token_ids", "format_chat",
    # HX1/HX2/HX6/HX8 — patches
    "apply_prefill_last_logit_patch",
    "apply_adaptive_prefill", "AdaptivePrefillController",
    "load_pattern_table", "apply_minference_prefill_patch",
    "TTTHeadRouter",
]
