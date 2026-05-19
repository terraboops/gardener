"""Per-layer learning-rate schedules. Pure function; reference map:
omlx/ttt_schedules.py:23 (re-derived, no omlx import)."""
from __future__ import annotations

import math
from typing import Literal


def compute_layer_lrs(
    base_lr: float,
    n_layers: int = 48,
    schedule: Literal["uniform", "reservoir", "cosine"] = "reservoir",
    gamma: float = 1.5,
) -> list[float]:
    if schedule == "uniform":
        return [base_lr] * n_layers
    if schedule == "reservoir":
        return [
            base_lr * ((l / max(n_layers - 1, 1)) ** gamma)
            for l in range(n_layers)
        ]
    if schedule == "cosine":
        lrs = []
        for l in range(n_layers):
            t = l / max(n_layers - 1, 1)
            weight = 0.5 * (1 + math.cos(2 * math.pi * t))
            lrs.append(base_lr * (0.1 + 0.9 * weight))
        return lrs
    raise ValueError(f"Unknown schedule: {schedule}")
