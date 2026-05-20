"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable."""
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .schedules import compute_layer_lrs
from .session import SessionPool
from .ttt import Candidate, TrainStats, TTTEngine
from .turboquant_kv import TurboQuantKVCache

__all__ = [
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
    "TurboQuantKVCache",
]
