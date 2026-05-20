"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable."""
from .duo_kv_cache import DuoKVCache, StreamingKVCache, load_duo_policy
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .schedules import compute_layer_lrs
from .session import SessionPool
from .ttt import Candidate, TrainStats, TTTEngine

__all__ = [
    "DuoKVCache", "StreamingKVCache", "load_duo_policy",
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
]
