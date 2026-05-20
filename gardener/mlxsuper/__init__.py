"""Superpowered-MLX core: stock mlx_lm made persistent/forkable/learnable."""
from .observability import counter, median_of_n, registry, timer
from .oplora import compute_svd_cache, project_lora_grads
from .patches.ttt_head_router import TTTHeadRouter
from .schedules import compute_layer_lrs
from .session import SessionPool
from .ttt import Candidate, TrainStats, TTTEngine

__all__ = [
    "SessionPool",
    "TTTEngine", "Candidate", "TrainStats",
    "TTTHeadRouter",
    "compute_layer_lrs",
    "project_lora_grads", "compute_svd_cache",
    "timer", "counter", "registry", "median_of_n",
]
