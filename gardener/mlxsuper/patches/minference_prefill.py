# SPDX-License-Identifier: Apache-2.0
"""MInference sparse prefill attention dispatch (arXiv:2407.02490).

Reads the per-head pattern table from calibration and, during prefill,
applies per-head sparse masks instead of full O(n²) attention:
  - a_shape:        sink tokens (first 4) + causal band
  - vertical_slash: top-importance columns + causal band
  - block_sparse:   block-diagonal with overlap
  - dense:          standard full attention (no optimization)

Applied via monkey-patching scaled_dot_product_attention.

Flag: --prefill-sparse minference on hypercar_server. Default off.

CRITICAL: Both shipped calibration tables (Qwen3-Coder, Qwen3.6) are
SYNTHETIC PLACEHOLDERS — programmatic defaults, NOT measured from actual
attention maps. Loading them emits a loud WARNING (see load_pattern_table).
Quality regressions on long-context gates (NIAH, RULER, MMLU-Pro) are
LIKELY when enabled. Predicted Goal 4 speedup (575 tok/s at 32K) is
UNSUBSTANTIATED until real calibration runs.

Ported from: omlx/patches/minference_prefill.py (not imported).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_PATTERN_TABLE: Optional[dict] = None
_LAYER_COUNTER = [0]  # Tracks which layer is being computed

# Default pattern table search path (same dir as this file, minference_patterns/ subdir)
DEFAULT_PATTERN_DIR = Path(__file__).parent / "minference_patterns"

# Synthetic-placeholder detection string — verbatim from the omlx source so
# both codebases agree on the sentinel.
_SYNTHETIC_SENTINEL = "SYNTHETIC"


def load_pattern_table(
    model_name: str,
    search_paths: list[Path] | None = None,
) -> dict:
    """Load a per-(layer, head) sparse-attention pattern calibration.

    Searches ``search_paths`` first, then falls back to the bundled
    ``minference_patterns/`` directory next to this file.

    Returns dict with ``_lookup: {(layer_idx, head_idx) -> pattern_entry}``
    plus the raw JSON contents.

    If the file's ``note`` field marks the table as a synthetic placeholder,
    emits a LOUD warning via ``logging.warning()`` so callers know quality
    on long-context gates is at risk.

    Raises:
        FileNotFoundError: if no matching ``.json`` is found.
        ValueError: if the JSON is missing required fields.
    """
    search_dirs = list(search_paths or []) + [DEFAULT_PATTERN_DIR]

    path: Optional[Path] = None
    for d in search_dirs:
        candidate = Path(d) / f"{model_name}.json"
        if candidate.exists():
            path = candidate
            break

    if path is None:
        searched = ", ".join(str(d) for d in search_dirs)
        raise FileNotFoundError(
            f"MInference pattern table not found: {model_name}.json\n"
            f"Searched: {searched}\n"
            f"Generate a real table with:\n"
            f"  scripts/minference_calibrate.py --model <your-model>"
        )

    table = json.loads(path.read_text())

    # Build lookup dict: (layer, head) -> pattern entry
    lookup: dict[tuple[int, int], dict] = {}
    for entry in table.get("heads", []):
        lookup[(entry["layer"], entry["head"])] = entry
    table["_lookup"] = lookup

    # Detect synthetic-placeholder tables and warn LOUDLY.
    # The pattern tables shipped with gardener may be programmatic defaults
    # (note field contains "SYNTHETIC") rather than real calibration output.
    # Dispatching with synthetic patterns:
    #   - Uses plausible per-head pattern types (vertical_slash / a_shape / …)
    #   - Uses plausible param values (num_vertical_cols, band_width, …)
    #   - But values DO NOT reflect the head's actual attention map.
    # → Likely quality regression on retrieval/long-context gates;
    #   speedup numbers in CLAUDE.md become unsubstantiated.
    note = table.get("note", "")
    is_synthetic = _SYNTHETIC_SENTINEL in note or "placeholder" in note.lower()
    if is_synthetic:
        logger.warning(
            "SYNTHETIC PLACEHOLDER calibration table loaded for %s.\n"
            "Quality regressions on long-context gates (NIAH, RULER, MMLU-Pro) are\n"
            "LIKELY when this is enabled. Predicted speedups are UNSUBSTANTIATED until\n"
            "calibration runs. Generate a real table with:\n"
            "  scripts/minference_calibrate.py --model %s",
            model_name,
            model_name,
        )

    num_layers = table.get("num_layers", 0)
    num_heads = table.get("num_heads", 0)
    summary = table.get("summary", {})
    pattern_counts = summary.get("pattern_counts", {})
    logger.info(
        "Loaded MInference patterns: %d layers × %d heads, distribution: %s%s",
        num_layers,
        num_heads,
        pattern_counts,
        " [SYNTHETIC]" if is_synthetic else "",
    )
    return table


# ---------------------------------------------------------------------------
# Sparse mask builders
# ---------------------------------------------------------------------------

def _build_a_shape_mask(L_q: int, params: dict, L_kv: int = 0) -> mx.array:
    """Build A-shape sparse mask: sink columns + causal band.

    Returns (L_q, L_kv) bool mask where True = attend.
    For chunked prefill, L_kv > L_q (accumulated context).
    """
    if L_kv == 0:
        L_kv = L_q
    num_sink = params.get("num_sink", 4)
    band_width = params.get("band_width", 64)

    # Row indices map to global positions: [L_kv - L_q, L_kv)
    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # (L_q, 1) global row positions
    cols = mx.arange(L_kv)[None, :]          # (1, L_kv)

    # Causal: cols <= rows (global positions)
    causal = cols <= rows

    # Sink columns: cols < num_sink
    sink = cols < num_sink

    # Band: rows - cols < band_width (local window)
    band = (rows - cols) < band_width

    mask = causal & (sink | band)
    return mask


def _build_vertical_slash_mask(
    L_q: int,
    params: dict,
    keys: Optional[mx.array] = None,
    L_kv: int = 0,
) -> mx.array:
    """Build vertical-slash sparse mask: important columns + causal band.

    Returns (L_q, L_kv) bool mask. For chunked prefill, L_kv > L_q.

    Column selection uses the runtime K-norm heuristic (content-adaptive)
    rather than the stored ``vertical_col_indices``. Calibration determines
    the PATTERN TYPE + PARAM COUNTS per head; runtime selects specific
    columns content-adaptively. (Stored indices are position-indexed to
    the calibration prompt's content — wrong at runtime with a different
    prompt.)
    """
    if L_kv == 0:
        L_kv = L_q
    band_width = params.get("band_width", 64)
    num_vert = params.get("num_vertical_cols", 16)

    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # global positions
    cols = mx.arange(L_kv)[None, :]

    causal = cols <= rows
    band = (rows - cols) < band_width

    # Runtime K-norm heuristic for vertical column selection
    if keys is not None and num_vert > 0:
        k_norms = mx.linalg.norm(keys[0, 0], axis=-1)  # (L_kv,)
        top_indices = mx.argsort(-k_norms)[:num_vert]
        vert_mask = mx.zeros((L_kv,), dtype=mx.bool_)
        vert_mask = vert_mask.at[top_indices].add(mx.ones((num_vert,), dtype=mx.bool_))
        vert = vert_mask[None, :]  # (1, L_kv)
    else:
        vert = cols < num_vert

    mask = causal & (band | vert)
    return mask


def _build_block_sparse_mask(L_q: int, params: dict, L_kv: int = 0) -> mx.array:
    """Build block-sparse mask: block-diagonal with overlap to previous block.

    Returns (L_q, L_kv) bool mask. For chunked prefill, L_kv > L_q.
    """
    if L_kv == 0:
        L_kv = L_q
    block_size = params.get("block_size", 64)

    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # global positions
    cols = mx.arange(L_kv)[None, :]

    causal = cols <= rows

    row_block = rows // block_size
    col_block = cols // block_size

    same_or_prev = (row_block == col_block) | (row_block == col_block + 1)

    mask = causal & same_or_prev
    return mask


# ---------------------------------------------------------------------------
# Sparse SDPA dispatch
# ---------------------------------------------------------------------------

def sparse_prefill_sdpa(
    queries: mx.array,   # (B, H_q, L, D)
    keys: mx.array,      # (B, H_kv, L, D)
    values: mx.array,    # (B, H_kv, L, D)
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """MInference sparse prefill: per-head pattern dispatch.

    For each query head, looks up its pattern from the calibration table
    and applies the corresponding sparse mask. Dense heads use standard SDPA.
    """
    global _LAYER_COUNTER

    if _PATTERN_TABLE is None:
        # No pattern table loaded — fall back to dense
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    B, H_q, L_q, D = queries.shape
    L_kv = keys.shape[2]
    H_kv = keys.shape[1]
    GQA = H_q // H_kv
    num_layers = _PATTERN_TABLE["num_layers"]

    layer_idx = _LAYER_COUNTER[0] % num_layers
    _LAYER_COUNTER[0] += 1

    lookup = _PATTERN_TABLE["_lookup"]

    # Check if ALL heads in this layer are dense — fast path
    all_dense = all(
        lookup.get((layer_idx, h), {}).get("pattern", "dense") == "dense"
        for h in range(H_q)
    )
    if all_dense or L_q <= 128:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    # Group heads by pattern type for batched dispatch
    # This avoids per-head kernel launches when many heads share a pattern.
    pattern_groups: dict[str, list[int]] = {}
    for h in range(H_q):
        entry = lookup.get((layer_idx, h))
        pattern = entry["pattern"] if entry is not None else "dense"
        pattern_groups.setdefault(pattern, []).append(h)

    # If majority is dense, just do full dense (overhead of slicing > savings)
    dense_count = len(pattern_groups.get("dense", []))
    if dense_count > H_q * 0.7:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    # Build outputs per head
    outputs: list[Optional[mx.array]] = [None] * H_q

    for pattern_type, head_indices in pattern_groups.items():
        if pattern_type == "dense":
            # Batch all dense heads together
            q_dense = queries[:, head_indices, :, :]
            kv_heads = [h // GQA for h in head_indices]
            k_dense = keys[:, kv_heads, :, :]
            v_dense = values[:, kv_heads, :, :]
            out = mx.fast.scaled_dot_product_attention(
                q_dense, k_dense, v_dense, scale=scale, mask=mask,
            )
            for i, h in enumerate(head_indices):
                outputs[h] = out[:, i:i + 1, :, :]
        else:
            # Sparse patterns: need per-head masks.
            # Get representative params from first head in group.
            first_entry = lookup.get((layer_idx, head_indices[0]), {})
            params = first_entry.get("params", {})

            # Build mask for this pattern — rectangular (L_q × L_kv) for chunked prefill
            if pattern_type == "a_shape":
                sparse_mask = _build_a_shape_mask(L_q, params, L_kv=L_kv)
            elif pattern_type == "vertical_slash":
                kv_h = head_indices[0] // GQA
                sparse_mask = _build_vertical_slash_mask(
                    L_q, params, keys=keys[:, kv_h:kv_h + 1, :, :], L_kv=L_kv,
                )
            elif pattern_type == "block_sparse":
                sparse_mask = _build_block_sparse_mask(L_q, params, L_kv=L_kv)
            else:
                sparse_mask = None

            if sparse_mask is not None:
                # -3.4e4 is fp16-safe (just-below-max-finite). Avoids -inf NaN
                # propagation through softmax at high sparsity.
                additive_mask = mx.where(sparse_mask, 0.0, -3.4e4).astype(queries.dtype)
                additive_mask = additive_mask[None, None, :, :]
                if mask is not None and not isinstance(mask, str):
                    combined_mask = additive_mask + mask
                else:
                    # "causal" string or None — sparse mask already includes causality
                    combined_mask = additive_mask
            else:
                combined_mask = mask

            # Batch these heads
            q_sparse = queries[:, head_indices, :, :]
            kv_heads = [h // GQA for h in head_indices]
            k_sparse = keys[:, kv_heads, :, :]
            v_sparse = values[:, kv_heads, :, :]

            out = mx.fast.scaled_dot_product_attention(
                q_sparse, k_sparse, v_sparse,
                scale=scale, mask=combined_mask,
            )
            for i, h in enumerate(head_indices):
                outputs[h] = out[:, i:i + 1, :, :]

    # Concatenate all heads
    return mx.concatenate(outputs, axis=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def _model_id_to_pattern_name(model_id: str) -> str:
    """Derive the pattern-table filename stem from a HuggingFace-style model ID.

    Mirrors the convention in ``scripts/minference_calibrate.py``:
    snake_case stem of the last path component.
    """
    last = model_id.rstrip("/").split("/")[-1]
    return last.lower().replace("-", "_")


def apply_minference_prefill_patch(
    model_name: str = "qwen3_coder_30b_a3b_instruct_8bit",
    model_id: str | None = None,
    pattern_table: dict | None = None,
) -> bool:
    """Monkey-patch SDPA for MInference sparse prefill.

    Args:
        model_name: pattern-table filename stem (without ``.json``). Used
            directly if provided and ``pattern_table`` is None.
        model_id: full HuggingFace-style model ID (e.g.
            ``"mlx-community/Qwen3.6-35B-A3B-4bit"``). If given, overrides
            ``model_name`` by deriving the stem via the calibration-script
            convention. Ignored when ``pattern_table`` is provided.
        pattern_table: pre-loaded pattern table dict (e.g. from
            ``load_pattern_table()``). If given, ``model_name``/``model_id``
            are ignored.

    Returns:
        True on first application; False if already patched (idempotent).
    """
    global _PATCHED, _PATTERN_TABLE, _LAYER_COUNTER

    if _PATCHED:
        return False

    if pattern_table is not None:
        _PATTERN_TABLE = pattern_table
    else:
        if model_id is not None:
            model_name = _model_id_to_pattern_name(model_id)

        try:
            _PATTERN_TABLE = load_pattern_table(model_name)
        except FileNotFoundError as e:
            logger.error(str(e))
            return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        logger.error(
            "mlx_lm not available — cannot apply MInference prefill patch"
        )
        return False

    existing_sdpa = mlx_base.scaled_dot_product_attention

    def minference_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask,
        sinks=None,
    ) -> mx.array:
        L = queries.shape[-2]

        # Only apply sparse dispatch during prefill (L > 1) with fp16 KV.
        # Skip quantized KV (tuples from QuantizedKVCache) — dequant params
        # are cache-object-specific and cannot be inferred from the tuple alone.
        if L > 1 and L > 128 and isinstance(keys, mx.array):
            return sparse_prefill_sdpa(queries, keys, values, scale, mask)

        # Short sequence or decode — pass through to existing SDPA
        return existing_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = minference_sdpa

    # Also patch any model modules that already imported the symbol by value
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is existing_sdpa or func is not minference_sdpa:
                setattr(mod, "scaled_dot_product_attention", minference_sdpa)

    _PATCHED = True
    _LAYER_COUNTER[0] = 0

    pattern_counts = (_PATTERN_TABLE or {}).get("summary", {}).get("pattern_counts", {})
    logger.info("MInference sparse prefill patch applied (%s)", pattern_counts)
    return True


def is_patched() -> bool:
    """Return True if the MInference SDPA patch has been applied."""
    return _PATCHED


def reset_layer_counter() -> None:
    """Reset layer counter at the start of each forward pass.

    Should be called before each new prefill to ensure correct layer mapping.
    """
    _LAYER_COUNTER[0] = 0
