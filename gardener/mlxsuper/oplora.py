"""OPLoRA gradient projection. Pure linear algebra; reference map:
omlx/oplora.py:87 (project_lora_grads), :129 (compute_svd_cache).
Re-derived on stock mlx; no omlx import."""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger("gardener.mlxsuper.oplora")


def _truncated_svd(W: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Return top-k left/right singular vectors (U_k:(m,k), V_k:(n,k)).

    Casts to float32 first so SVD works on quantized (uint32/int8) weights.
    """
    if W.dtype not in (mx.float32, mx.float64, mx.complex64):
        W = W.astype(mx.float32)
    U, S, Vt = mx.linalg.svd(W, stream=mx.cpu)
    k = min(k, U.shape[1], Vt.shape[0])
    return U[:, :k], Vt[:k, :].T


def project_lora_grads(
    W: mx.array,
    A: mx.array,
    B: mx.array,
    dA: mx.array,
    dB: mx.array,
    svd_data: dict[str, mx.array],
) -> tuple[mx.array, mx.array]:
    """Project LoRA grads onto the orthogonal complement of W's top-k SVD.

    dA_safe = dA - (dA V_k) V_k^T   (remove right-singular component)
    dB_safe = dB - U_k (U_k^T dB)   (remove left-singular component)
    """
    U_k = svd_data["U_k"]
    V_k = svd_data["V_k"]
    dA_safe = dA - (dA @ V_k) @ V_k.T
    dB_safe = dB - U_k @ (U_k.T @ dB)
    return dA_safe, dB_safe


def compute_svd_cache(
    weights: dict[str, mx.array],
    k: int = 8,
) -> dict[str, dict[str, mx.array]]:
    """Truncated SVD per named frozen weight matrix.

    `weights` maps a layer path -> its (m, n) base weight. Returns
    {path: {"U_k": (m,k), "V_k": (n,k)}}. (Caller supplies the weight
    dict; this keeps the function model-structure-agnostic and pure.)
    """
    cache: dict[str, dict[str, mx.array]] = {}
    for path, W in weights.items():
        if not isinstance(W, mx.array) or W.ndim != 2:
            logger.warning("SVD cache: skipping non-2D weight %s", path)
            continue
        U_k, V_k = _truncated_svd(W, k)
        cache[path] = {"U_k": U_k, "V_k": V_k}
    return cache
