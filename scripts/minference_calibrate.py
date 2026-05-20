#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MInference per-head sparse-attention pattern calibration (arXiv:2407.02490).

Runs a model over a calibration prompt, captures full attention maps,
and assigns each (layer, head) to one of:
  - a_shape:        sink tokens + local causal window
  - vertical_slash: vertical column indices + diagonal band
  - block_sparse:   block-diagonal pattern
  - dense:          no exploitable structure

Output: JSON pattern table for runtime sparse-attention dispatch by
``gardener.mlxsuper.patches.minference_prefill``.

Usage:
    .venv/bin/python scripts/minference_calibrate.py \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit

    .venv/bin/python scripts/minference_calibrate.py \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit \\
        --seq-len 4096 \\
        --out gardener/mlxsuper/patches/minference_patterns/qwen3_coder_30b_a3b_instruct_8bit.json

On success, prints:
    WROTE REAL TABLE → <path>

Reference: arXiv:2407.02490 "MInference 1.0: Accelerating Pre-filling for
Long-Context LLMs via Dynamic Sparse Attention".

Ported from omlx/patches/minference_prefill.py + scripts/minference_calibrate.py
(not imported — gardener has no omlx dependency).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger("minference.calibrate")

# Default output directory (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "gardener" / "mlxsuper" / "patches" / "minference_patterns"

# Sparsity budget: target fraction of attention matrix to SKIP
TARGET_SPARSITY = 0.85

# Max reconstruction MSE (relative to dense) for pattern to be valid
MAX_RECONSTRUCTION_MSE = 0.01


# ---------------------------------------------------------------------------
# Calibration corpus
# ---------------------------------------------------------------------------

CALIBRATION_PROMPT = '''\
You are a senior software engineer reviewing a large Python codebase.
Analyze the following modules and identify potential issues:

```python
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single entry in the LRU cache with TTL support."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        self.accessed_at = time.time()
        self.access_count += 1


class DistributedCache:
    """Thread-safe distributed cache with consistent hashing."""

    def __init__(self, capacity: int = 1000, num_shards: int = 16):
        self.capacity = capacity
        self.num_shards = num_shards
        self._shards: List[Dict[str, CacheEntry]] = [
            {} for _ in range(num_shards)
        ]
        self._locks = [asyncio.Lock() for _ in range(num_shards)]
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _shard_for_key(self, key: str) -> int:
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16) % self.num_shards

    async def get(self, key: str) -> Optional[Any]:
        shard_idx = self._shard_for_key(key)
        async with self._locks[shard_idx]:
            entry = self._shards[shard_idx].get(key)
            if entry is None or entry.is_expired:
                self._stats["misses"] += 1
                if entry and entry.is_expired:
                    del self._shards[shard_idx][key]
                return None
            entry.touch()
            self._stats["hits"] += 1
            return entry.value

    async def put(self, key: str, value: Any, ttl: float = 3600.0) -> None:
        shard_idx = self._shard_for_key(key)
        async with self._locks[shard_idx]:
            shard = self._shards[shard_idx]
            if len(shard) >= self.capacity // self.num_shards:
                self._evict_lru(shard)
            shard[key] = CacheEntry(key=key, value=value, ttl_seconds=ttl)

    def _evict_lru(self, shard: Dict[str, CacheEntry]) -> None:
        if not shard:
            return
        oldest_key = min(shard, key=lambda k: shard[k].accessed_at)
        del shard[oldest_key]
        self._stats["evictions"] += 1


def compute_similarity_matrix(
    vectors: List[List[float]],
    metric: str = "cosine",
) -> List[List[float]]:
    """Compute pairwise similarity matrix."""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if metric == "cosine":
                dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
                norm_i = math.sqrt(sum(a * a for a in vectors[i]))
                norm_j = math.sqrt(sum(b * b for b in vectors[j]))
                sim = dot / (norm_i * norm_j + 1e-8)
            elif metric == "euclidean":
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])))
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix
```

Provide a detailed code review covering:
1. Thread safety issues
2. Performance bottlenecks
3. Memory management concerns
4. API design improvements
'''


# ---------------------------------------------------------------------------
# Attention pattern classifiers
# ---------------------------------------------------------------------------

def _classify_a_shape(attn: np.ndarray, sparsity: float) -> dict:
    """Test if attention matches A-shape: sink tokens + local window."""
    L = attn.shape[0]
    num_sink = min(4, L // 4)
    budget = int(L * L * (1 - sparsity))

    band_w = max(1, budget // L - num_sink)
    mask = np.zeros((L, L), dtype=bool)
    mask[:, :num_sink] = True
    for i in range(L):
        start = max(0, i - band_w + 1)
        mask[i, start:i + 1] = True

    sparse_attn = attn * mask
    row_sums = sparse_attn.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    sparse_attn = sparse_attn / row_sums

    mse = float(np.mean((attn - sparse_attn) ** 2))
    actual_sparsity = 1.0 - mask.sum() / (L * L)

    return {
        "pattern": "a_shape",
        "mse": mse,
        "sparsity": float(actual_sparsity),
        "params": {"num_sink": num_sink, "band_width": band_w},
    }


def _classify_vertical_slash(attn: np.ndarray, sparsity: float) -> dict:
    """Test if attention matches vertical-slash: important columns + diagonal."""
    L = attn.shape[0]
    budget = int(L * L * (1 - sparsity))

    col_importance = attn.sum(axis=0)
    band_w = max(1, L // 8)
    diag_budget = band_w * L
    vert_budget = max(1, (budget - diag_budget) // L)
    vert_cols = np.argsort(col_importance)[-vert_budget:]

    mask = np.zeros((L, L), dtype=bool)
    mask[:, vert_cols] = True
    for i in range(L):
        start = max(0, i - band_w + 1)
        mask[i, start:i + 1] = True

    sparse_attn = attn * mask
    row_sums = sparse_attn.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    sparse_attn = sparse_attn / row_sums

    mse = float(np.mean((attn - sparse_attn) ** 2))
    actual_sparsity = 1.0 - mask.sum() / (L * L)

    return {
        "pattern": "vertical_slash",
        "mse": mse,
        "sparsity": float(actual_sparsity),
        "params": {
            "num_vertical_cols": int(vert_budget),
            "vertical_col_indices": sorted(vert_cols.tolist()),
            "band_width": band_w,
        },
    }


def _classify_block_sparse(attn: np.ndarray, sparsity: float) -> dict:
    """Test if attention matches block-sparse: block-diagonal pattern."""
    L = attn.shape[0]

    best: Optional[dict] = None
    for block_size in [32, 64, 128, 256]:
        if block_size > L:
            continue
        num_blocks = (L + block_size - 1) // block_size

        mask = np.zeros((L, L), dtype=bool)
        for b in range(num_blocks):
            row_start = b * block_size
            row_end = min((b + 1) * block_size, L)
            col_start = b * block_size
            col_end = min((b + 1) * block_size, L)
            mask[row_start:row_end, col_start:col_end] = True
            if b > 0:
                prev_start = (b - 1) * block_size
                mask[row_start:row_end, prev_start:col_start] = True

        sparse_attn = attn * mask
        row_sums = sparse_attn.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        sparse_attn = sparse_attn / row_sums

        mse = float(np.mean((attn - sparse_attn) ** 2))
        actual_sparsity = 1.0 - mask.sum() / (L * L)

        if best is None or mse < best["mse"]:
            best = {
                "pattern": "block_sparse",
                "mse": mse,
                "sparsity": float(actual_sparsity),
                "params": {"block_size": block_size},
            }

    assert best is not None
    return best


def classify_head(attn: np.ndarray, sparsity: float = TARGET_SPARSITY) -> dict:
    """Classify a single attention head's pattern.

    Tries all pattern types and picks the one with lowest MSE that meets
    the sparsity budget.

    Args:
        attn: (L, L) attention weights (already softmax'd, causal)
        sparsity: target fraction of matrix to skip

    Returns:
        dict with pattern type, MSE, sparsity, and pattern-specific params
    """
    candidates = [
        _classify_a_shape(attn, sparsity),
        _classify_vertical_slash(attn, sparsity),
        _classify_block_sparse(attn, sparsity),
    ]

    # Pick best: lowest MSE that meets sparsity target (10% tolerance)
    valid = [c for c in candidates if c["sparsity"] >= sparsity * 0.9]
    if not valid:
        valid = candidates

    best = min(valid, key=lambda c: c["mse"])

    # If best MSE is too high, classify as dense (no sparse pattern works)
    if best["mse"] > MAX_RECONSTRUCTION_MSE:
        return {
            "pattern": "dense",
            "mse": best["mse"],
            "sparsity": 0.0,
            "params": {},
        }

    return best


# ---------------------------------------------------------------------------
# Attention capture hook
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Hook into model attention layers to capture full attention maps."""

    def __init__(self, model: Any, num_layers: int, num_heads: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_maps: dict[tuple[int, int], np.ndarray] = {}
        self._model = model

    def capture(self, tokenizer: Any, prompt: str, max_seq_len: int = 2048) -> None:
        """Run model on prompt and capture attention maps."""
        import mlx_lm.models.base as mlx_base
        import sys as _sys

        original_sdpa = mlx_base.scaled_dot_product_attention
        layer_counter = [0]

        def capturing_sdpa(
            queries: mx.array,
            keys: mx.array,
            values: mx.array,
            cache: Any,
            scale: float,
            mask: Any,
            **kwargs: Any,
        ) -> mx.array:
            B, H_q, L_q, D = queries.shape
            _, H_kv, L_kv, _ = keys.shape

            GQA = H_q // H_kv
            if GQA > 1:
                keys_exp = mx.repeat(keys, GQA, axis=1)
                values_exp = mx.repeat(values, GQA, axis=1)
            else:
                keys_exp = keys
                values_exp = values

            scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale

            if isinstance(mask, str) and mask == "causal":
                offset = L_kv - L_q
                rows = mx.arange(L_q)[:, None] + offset
                cols = mx.arange(L_kv)[None, :]
                causal_mask = mx.where(cols <= rows, 0.0, -3.4e4).astype(scores.dtype)
                scores = scores + causal_mask
            elif mask is not None:
                scores = scores + mask

            weights = mx.softmax(scores, axis=-1)

            layer_idx = layer_counter[0] % self.num_layers
            weights_np = np.array(weights[0].astype(mx.float32))  # (H_q, L_q, L_kv)
            for h in range(min(H_q, self.num_heads)):
                self.attention_maps[(layer_idx, h)] = weights_np[h]

            layer_counter[0] += 1

            out = weights @ values_exp
            return out

        # Patch mlx_base and every already-imported model module
        mlx_base.scaled_dot_product_attention = capturing_sdpa
        _patched_modules: list[tuple[Any, Any]] = []
        for mod_name, mod in list(_sys.modules.items()):
            if mod is None:
                continue
            if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
                continue
            if hasattr(mod, "scaled_dot_product_attention"):
                func = getattr(mod, "scaled_dot_product_attention")
                if func is original_sdpa:
                    setattr(mod, "scaled_dot_product_attention", capturing_sdpa)
                    _patched_modules.append((mod, func))

        try:
            tokens = tokenizer.encode(prompt)[:max_seq_len]
            x = mx.array([tokens])
            logits = self._model(x)
            mx.eval(logits)
            logger.info(
                "Captured attention maps for %d (layer, head) pairs",
                len(self.attention_maps),
            )
        finally:
            mlx_base.scaled_dot_product_attention = original_sdpa
            for mod, func in _patched_modules:
                setattr(mod, "scaled_dot_product_attention", func)


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate(
    model_path: str,
    seq_len: int = 2048,
    output_path: Optional[Path] = None,
    sparsity: float = TARGET_SPARSITY,
) -> dict:
    """Run calibration and produce a REAL pattern table.

    Args:
        model_path: HuggingFace model ID or local path
        seq_len: sequence length for calibration (longer = more accurate)
        output_path: where to write the JSON pattern table (derived from
            model name if None)
        sparsity: target sparsity for pattern classification

    Returns:
        Pattern table dict (also written to disk)
    """
    from mlx_lm import load  # type: ignore[import]

    logger.info("Loading model: %s", model_path)
    model, tokenizer = load(model_path)

    # Walk attention layers — hybrid models (e.g. Qwen3.6) have SSM layers
    # that never call SDPA; only count layers that will call SDPA.
    attn_layer_indices: list[int] = []
    first_attn = None
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn_layer_indices.append(i)
        if first_attn is None:
            first_attn = attn

    if first_attn is None:
        raise RuntimeError(
            f"No attention layers found in {model_path}. Pure-SSM models "
            "have nothing for MInference to calibrate."
        )

    n_attn_layers = len(attn_layer_indices)
    n_total_layers = len(model.layers)

    def _attr(obj: Any, *names: str, default: Any = None) -> Any:
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return default

    num_heads = _attr(first_attn, "n_heads", "num_attention_heads", "num_heads")
    if num_heads is None:
        q_proj = getattr(first_attn, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "weight"):
            head_dim = _attr(first_attn, "head_dim")
            if head_dim is not None:
                num_heads = q_proj.weight.shape[0] // head_dim
        if num_heads is None:
            num_heads = 32
            logger.warning("Could not detect num_heads, defaulting to %d", num_heads)

    if n_attn_layers != n_total_layers:
        logger.info(
            "Hybrid model detected: %d attention layers / %d total layers",
            n_attn_layers,
            n_total_layers,
        )

    num_layers = n_attn_layers
    logger.info(
        "Model: %d attention layers, %d heads. Calibration seq_len=%d, sparsity=%.0f%%",
        num_layers, num_heads, seq_len, sparsity * 100,
    )

    logger.info("Capturing attention maps…")
    t0 = time.perf_counter()
    capture = AttentionCapture(model, num_layers, num_heads)
    capture.capture(tokenizer, CALIBRATION_PROMPT, max_seq_len=seq_len)
    capture_time = time.perf_counter() - t0
    logger.info("Attention capture complete in %.1fs", capture_time)

    logger.info("Classifying attention patterns…")
    pattern_table: dict[str, Any] = {
        "model": model_path,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "target_sparsity": sparsity,
        "calibration_time_s": round(capture_time, 1),
        # NOTE: no "note" field with SYNTHETIC — this is a real table.
        "heads": [],
    }

    pattern_counts = {"a_shape": 0, "vertical_slash": 0, "block_sparse": 0, "dense": 0}
    total_mse = 0.0
    total_sparsity = 0.0

    for layer in range(num_layers):
        for head in range(num_heads):
            key = (layer, head)
            if key in capture.attention_maps:
                attn = capture.attention_maps[key]
                result = classify_head(attn, sparsity)
            else:
                result = {"pattern": "dense", "mse": 0.0, "sparsity": 0.0, "params": {}}

            entry = {
                "layer": layer,
                "head": head,
                "pattern": result["pattern"],
                "mse": round(result["mse"], 6),
                "sparsity": round(result["sparsity"], 4),
                "params": result["params"],
            }
            pattern_table["heads"].append(entry)
            pattern_counts[result["pattern"]] += 1
            total_mse += result["mse"]
            total_sparsity += result["sparsity"]

    num_entries = num_layers * num_heads
    pattern_table["summary"] = {
        "pattern_counts": pattern_counts,
        "avg_mse": round(total_mse / max(1, num_entries), 6),
        "avg_sparsity": round(total_sparsity / max(1, num_entries), 4),
        "total_entries": num_entries,
    }

    logger.info("Pattern distribution: %s", pattern_counts)
    logger.info("Average MSE: %.6f", pattern_table["summary"]["avg_mse"])
    logger.info("Average sparsity: %.1f%%", pattern_table["summary"]["avg_sparsity"] * 100)

    # Validation
    avg_sparsity = pattern_table["summary"]["avg_sparsity"]
    avg_mse = pattern_table["summary"]["avg_mse"]
    sparsity_floor = 0.9 * sparsity
    assert avg_sparsity >= sparsity_floor, (
        f"Average sparsity {avg_sparsity:.1%} < {sparsity_floor:.1%} — "
        f"model may not suit MInference at this budget"
    )
    assert avg_mse < MAX_RECONSTRUCTION_MSE, (
        f"Average MSE {avg_mse:.6f} >= {MAX_RECONSTRUCTION_MSE} — patterns too lossy"
    )
    logger.info("Validation PASSED")

    # Determine output path
    if output_path is None:
        model_stem = model_path.rstrip("/").split("/")[-1].lower().replace("-", "_")
        output_path = DEFAULT_OUTPUT_DIR / f"{model_stem}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pattern_table, indent=2))

    print(f"WROTE REAL TABLE → {output_path}")
    return pattern_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MInference per-head sparse-attention calibration (arXiv:2407.02490).\n\n"
            "Captures real attention maps and classifies each (layer, head) into\n"
            "a_shape / vertical_slash / block_sparse / dense.\n\n"
            "The output replaces the SYNTHETIC PLACEHOLDER table shipped in\n"
            "gardener/mlxsuper/patches/minference_patterns/ with a real one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to a text file to use as calibration corpus (optional; "
             "defaults to the built-in code-review prompt)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Calibration sequence length (default: 2048). Longer = more accurate.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=TARGET_SPARSITY,
        help=f"Target sparsity (default: {TARGET_SPARSITY})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: derived from model name in the bundled patterns dir)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Optionally override calibration corpus from file
    global CALIBRATION_PROMPT
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            parser.error(f"--corpus path not found: {corpus_path}")
        CALIBRATION_PROMPT = corpus_path.read_text()
        logger.info("Using corpus from %s (%d chars)", corpus_path, len(CALIBRATION_PROMPT))

    out_path = Path(args.out) if args.out else None

    calibrate(
        model_path=args.model,
        seq_len=args.seq_len,
        output_path=out_path,
        sparsity=args.sparsity,
    )


if __name__ == "__main__":
    main()
