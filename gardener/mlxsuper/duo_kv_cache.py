"""DuoKVCache — best-quality default KV mode.

For each KV head: classified as streaming (fp16 ring buffer of `window`
recent + `sink` early tokens) or retrieval (full history; quantized when
quantize_retrieval=True). Per-layer policy loaded from JSON calibration.

Memory profile at 256K context, Qwen3-Coder 48 layers, 4 KV heads/layer:
- Streaming heads (fp16 ring 256+4): ~256 KB / layer / head
- Retrieval heads (3-bit native):    ~6 MB  / layer / head at 256K
- Total: ~300 MB vs ~6.3 GB fully-fp16 (20× savings)

CRITICAL gotchas (from omlx Tasks 269/271/298-301):
1. NEVER set self.bits — mlx_lm SDPA checks hasattr(cache, 'bits') to route
   to quantized_matmul. Use self._quant_bits instead.
2. Module-level _TRIM_INDEX_CACHE is shared across all 48 layers (same
   T_total/sink/window each decode step). Hits 48× per token; gating
   16K decode ≥ 40 tok/s.
3. gather+mask, NOT slice+write — slice+write was 73% worse peak memory
   per Task 295(b) revert.

Reference map (NOT imported):
- omlx/duo_kv_cache.py:1-742
- omlx/patches/duoattention_policies/*.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

logger = logging.getLogger(__name__)

# Default search path for policy JSONs that ship with gardener.
_DEFAULT_POLICY_DIR = Path(__file__).parent / "duo_policies"

# ---------------------------------------------------------------------------
# Module-level trim-index cache (Tasks 269/271)
#
# All layers at any given decode step see the SAME T_total / sink / window.
# Caching here means the index arrays are built once per unique (T_total,
# sink, window) tuple and reused 48× across layers — critical for 16K
# decode ≥ 40 tok/s on a 48-layer model.
#
# Structure: {(T_total, sink, window): {arrays...}}
# The dict has at most one entry per unique (T_total, sink, window) key;
# in practice it holds exactly one entry at any decode step (all 48 layers
# share the same T_total/sink/window at any given moment).
# ---------------------------------------------------------------------------
_TRIM_INDEX_CACHE: dict = {}


def _trim_indices_for(T_total: int, sink: int, window: int) -> dict:
    """Build (and cache) trim-path arrays for a given T_total.

    Returns a dict with keys:
      stream_arr, retrieval_arr, stream_mask, retrieval_mask, stream_real_len.

    Cached in the module-level ``_TRIM_INDEX_CACHE`` keyed by
    ``(T_total, sink, window)`` — shared across all layers per decode step.
    """
    key = (T_total, sink, window)
    entry = _TRIM_INDEX_CACHE.get(key)
    if entry is not None:
        return entry

    stream_idx = list(range(sink)) + list(range(T_total - window, T_total))
    stream_real_len = len(stream_idx)  # = sink + window
    stream_padded = stream_idx + [0] * (T_total - stream_real_len)
    stream_arr = mx.array(stream_padded, dtype=mx.int32)
    retrieval_arr = mx.arange(T_total, dtype=mx.int32)

    if stream_real_len < T_total:
        stream_mask = mx.concatenate([
            mx.ones(stream_real_len, dtype=mx.bool_),
            mx.zeros(T_total - stream_real_len, dtype=mx.bool_),
        ])
    else:
        stream_mask = mx.ones(T_total, dtype=mx.bool_)
    retrieval_mask = mx.ones(T_total, dtype=mx.bool_)

    entry = {
        "stream_arr": stream_arr,
        "retrieval_arr": retrieval_arr,
        "stream_mask": stream_mask,
        "retrieval_mask": retrieval_mask,
        "stream_real_len": stream_real_len,
    }
    # Evict stale entries (old T_total values won't be seen again once context
    # advances — keep the dict lean).  At steady-state decode, there is exactly
    # one key in the cache.
    _TRIM_INDEX_CACHE.clear()
    _TRIM_INDEX_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Module-level duo-split decode-skip flag (Task 290)
#
# When the duo-split attention patch is active and T_new == 1 (decode),
# patched_sdpa re-fetches via get_streaming_kv / get_retrieval_kv and the
# unified update_and_fetch output is never used. The trim block would allocate
# ~67 MB of transient tensors per layer per decode step at 16K (~3.2 GB across
# 48 layers) for no benefit. This flag opts out of that work.
#
# At prefill (T_new > 1) the trim IS still needed because patched_sdpa falls
# through to original SDPA. Port the flag but keep it False — Gardener does
# not ship the duo-split patch in HX4.
# ---------------------------------------------------------------------------
_DUO_SPLIT_DECODE_SKIP: bool = False


def set_duo_split_decode_skip(enabled: bool) -> None:
    """Toggle the skip-trim-at-decode optimisation (for the duo-split patch).

    Safe to call only when the duo-split SDPA patch is also applied — the
    optimisation relies on patched_sdpa ignoring the trimmed output at decode.
    Not used in HX4 (flag reserved for a future patch).
    """
    global _DUO_SPLIT_DECODE_SKIP
    _DUO_SPLIT_DECODE_SKIP = bool(enabled)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_duo_policy(
    model_name: str = "qwen3_coder_30b_a3b_instruct_8bit",
    search_paths: list[Path] | None = None,
) -> dict:
    """Load per-head DuoAttention policy from a calibration JSON.

    Searches ``search_paths`` in order, then the built-in ``duo_policies/``
    directory that ships with gardener.  Raises ``FileNotFoundError`` if no
    matching file is found.

    The returned dict contains all original top-level keys plus an extra
    ``_lookup`` key: ``{(layer_idx, head_idx): "streaming" | "retrieval"}``.

    Two JSON schemas are supported:

    1. **heads-array** (the omlx calibration format)::

           {"heads": [{"layer": 0, "head": 0, "policy": "retrieval", ...}, ...]}

    2. **layers-dict** (the test/synthetic format)::

           {"layers": {"0": {"streaming_heads": [...], "retrieval_heads": [...]}, ...}}
    """
    paths_to_try: list[Path] = list(search_paths or []) + [_DEFAULT_POLICY_DIR]

    policy_path: Path | None = None
    for directory in paths_to_try:
        candidate = Path(directory) / f"{model_name}.json"
        if candidate.exists():
            policy_path = candidate
            break

    if policy_path is None:
        searched = ", ".join(str(Path(p) / f"{model_name}.json") for p in paths_to_try)
        raise FileNotFoundError(
            f"DuoAttention policy '{model_name}' not found.\n"
            f"Searched: {searched}\n"
            f"Run: python scripts/duoattention_calibrate.py to generate."
        )

    policy: dict = json.loads(policy_path.read_text(encoding="utf-8"))

    # Build the flat (layer, head) → policy_type lookup.
    lookup: dict[tuple[int, int], str] = {}

    if "heads" in policy:
        # omlx heads-array format
        for entry in policy["heads"]:
            lookup[(int(entry["layer"]), int(entry["head"]))] = entry["policy"]
    elif "layers" in policy:
        # Synthetic layers-dict format used by tests
        for layer_str, layer_data in policy["layers"].items():
            li = int(layer_str)
            for h in layer_data.get("streaming_heads", []):
                lookup[(li, int(h))] = "streaming"
            for h in layer_data.get("retrieval_heads", []):
                lookup[(li, int(h))] = "retrieval"
    else:
        raise ValueError(
            f"Unrecognised DuoAttention policy schema in {policy_path}. "
            "Expected 'heads' array or 'layers' dict."
        )

    policy["_lookup"] = lookup

    streaming_pct = policy.get("streaming_fraction", 0) * 100
    logger.debug(
        "DuoAttention policy '%s': %.0f%% streaming, window=%s, sink=%s",
        model_name, streaming_pct,
        policy.get("window", "?"), policy.get("sink", "?"),
    )
    return policy


# ---------------------------------------------------------------------------
# StreamingKVCache
# ---------------------------------------------------------------------------

class StreamingKVCache:
    """Ring-buffer KV cache for streaming attention heads.

    Keeps only the first ``sink`` tokens plus the most recent ``window``
    tokens.  Total capacity: ``sink + window`` tokens in fp16.

    Args:
        n_kv_heads: Number of KV heads handled by this instance.
        head_dim:   Key/value head dimension (used for initialisation
                    validation only; buffer is shaped from the first write).
        window:     Number of most-recent tokens to retain.
        sink:       Number of initial tokens to always retain.
    """

    def __init__(
        self,
        n_kv_heads: int = 1,
        head_dim: int = 128,
        window: int = 256,
        sink: int = 4,
    ):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window = window
        self.sink = sink
        self.capacity = sink + window
        self.offset: int = 0
        self._keys: Optional[mx.array] = None   # (B, n_kv_heads, capacity, D)
        self._values: Optional[mx.array] = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Add new K/V tokens and return the ring-buffer contents.

        ``keys`` / ``values`` shape: ``(B, n_kv_heads, T_new, D)``.
        """
        B, H, T_new, D = keys.shape

        if self._keys is None:
            self._keys = mx.zeros((B, H, self.capacity, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H, self.capacity, D), dtype=values.dtype)

        if self.offset < self.capacity:
            # Still filling the pre-allocated buffer — append directly.
            end = min(self.offset + T_new, self.capacity)
            actual = end - self.offset
            self._keys[:, :, self.offset:end] = keys[:, :, :actual]
            self._values[:, :, self.offset:end] = values[:, :, :actual]
            self.offset = end

            if T_new > actual:
                # Overflow into ring region.
                # When remaining > ring_len, earlier writes are overwritten by
                # later ones — clamp to the last ring_len tokens (Task 300).
                remaining = T_new - actual
                ring_start = self.sink
                ring_len = self.window
                effective = min(remaining, ring_len)
                start = actual + (remaining - effective)
                positions = ring_start + (
                    mx.arange(effective) + (remaining - effective)
                ) % ring_len
                self._keys[:, :, positions] = keys[:, :, start:start + effective]
                self._values[:, :, positions] = values[:, :, start:start + effective]
                self.offset += remaining
        else:
            # Full ring mode — vectorised scatter (Task 300 dedup pattern).
            ring_start = self.sink
            ring_len = self.window
            effective = min(T_new, ring_len)
            start = T_new - effective
            positions = ring_start + (
                mx.arange(effective) + (self.offset + start - self.capacity)
            ) % ring_len
            self._keys[:, :, positions] = keys[:, :, start:start + effective]
            self._values[:, :, positions] = values[:, :, start:start + effective]
            self.offset += T_new

        valid = min(self.offset, self.capacity)
        return self._keys[:, :, :valid], self._values[:, :, :valid]

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        if self._keys is None:
            return None, None
        valid = min(self.offset, self.capacity)
        return self._keys[:, :, :valid], self._values[:, :, :valid]


# ---------------------------------------------------------------------------
# DuoKVCache
# ---------------------------------------------------------------------------

class DuoKVCache:
    """Two-storage-class KV cache per the DuoAttention policy.

    For each KV head the per-layer policy assigns:
      - **retrieval**: full history kept in an fp16 pre-allocated slab
        (optionally 3-bit quantized via mlx_lm ``QuantizedKVCache``).
      - **streaming**: fp16 ring buffer of ``sink`` initial + ``window``
        recent tokens.

    The attention layer interacts with a single cache object via
    ``update_and_fetch``.  DuoKVCache splits heads by type, dispatches to the
    appropriate sub-cache, assembles a unified ``(K_out, V_out)`` and returns.

    Args:
        policy:              Dict returned by :func:`load_duo_policy`.
        layer_idx:           Index of the transformer layer this cache serves.
        n_kv_heads:          Total number of KV heads.
        window:              Streaming head ring-buffer size (recent tokens).
        sink:                Streaming head sink size (initial tokens kept).
        quantize_retrieval:  Route retrieval heads through ``QuantizedKVCache``.
        group_size:          Quantisation group size (ignored when not quantizing).
        quant_bits:          Quantisation bits (ignored when not quantizing).
    """

    def __init__(
        self,
        policy: dict,
        layer_idx: int,
        n_kv_heads: int = 4,
        window: int = 256,
        sink: int = 4,
        quantize_retrieval: bool = False,
        group_size: int = 64,
        quant_bits: int = 3,
    ):
        self.layer_idx = layer_idx
        self.n_kv_heads = n_kv_heads
        self.window = window
        self.sink = sink
        self.capacity = sink + window

        # NOTE: NEVER set self.bits — mlx_lm SDPA checks hasattr(cache, 'bits')
        # and routes to quantized_matmul which is incompatible with fp16 streaming
        # heads.  Store as self._quant_bits instead (omlx/duo_kv_cache.py:242).
        self._quant_bits = quant_bits
        self.group_size = group_size
        self._quantize_retrieval = quantize_retrieval

        # ------------------------------------------------------------------
        # Classify each KV head using the flat (layer, head) lookup.
        # ------------------------------------------------------------------
        lookup: dict[tuple[int, int], str] = policy.get("_lookup", {})
        n_q_heads: int = policy.get("n_heads", n_kv_heads)
        gqa: int = max(1, n_q_heads // n_kv_heads)

        self.head_types: list[str] = []
        for kv_h in range(n_kv_heads):
            # A KV head is streaming only when ALL its Q heads are streaming.
            q_heads = range(kv_h * gqa, (kv_h + 1) * gqa)
            all_streaming = all(
                lookup.get((layer_idx, qh), "retrieval") == "streaming"
                for qh in q_heads
            )
            self.head_types.append("streaming" if all_streaming else "retrieval")

        self._is_streaming: list[bool] = [t == "streaming" for t in self.head_types]
        self._n_streaming: int = sum(self._is_streaming)
        self._streaming_head_indices: list[int] = [
            h for h in range(n_kv_heads) if self._is_streaming[h]
        ]
        self._retrieval_head_indices: list[int] = [
            h for h in range(n_kv_heads) if not self._is_streaming[h]
        ]

        # Precomputed assembly plan — avoids list.index() per output head.
        _ret_pos = {h: i for i, h in enumerate(self._retrieval_head_indices)}
        _str_pos = {h: i for i, h in enumerate(self._streaming_head_indices)}
        self._assembly_plan_src: list[str] = []
        self._assembly_plan_row: list[int] = []
        for h in range(n_kv_heads):
            if self._is_streaming[h]:
                self._assembly_plan_src.append("str")
                self._assembly_plan_row.append(_str_pos[h])
            else:
                self._assembly_plan_src.append("ret")
                self._assembly_plan_row.append(_ret_pos[h])

        # ------------------------------------------------------------------
        # Sub-cache allocation.
        # ------------------------------------------------------------------
        self._step = 256  # headroom for pre-allocated fp16 slab grows

        if quantize_retrieval:
            # One shared QuantizedKVCache for all retrieval heads.
            self._retrieval_cache: Optional[QuantizedKVCache] = (
                QuantizedKVCache(bits=quant_bits, group_size=group_size)
                if self._retrieval_head_indices else None
            )
            # Per-head StreamingKVCache (ring needs independent offset).
            self._streaming_head_caches: dict[int, StreamingKVCache] = {
                h: StreamingKVCache(n_kv_heads=1, window=window, sink=sink)
                for h in self._streaming_head_indices
            }
            # fp16 slab not used in this mode.
            self._keys: Optional[mx.array] = None
            self._values: Optional[mx.array] = None
            self._kv_len: int = 0
        else:
            # fp16 pre-allocated slab (all heads together, streaming trim applied
            # by gather+mask at decode time — Task 295(b) / 301).
            self._retrieval_cache = None
            self._streaming_head_caches = {}
            self._keys = None
            self._values = None
            self._kv_len = 0

        logger.debug(
            "Layer %d: %d/%d streaming KV heads (%s)",
            layer_idx, self._n_streaming, n_kv_heads,
            "quantized-retrieval" if quantize_retrieval else "fp16-all",
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Store new K/V, apply streaming-head trim, return assembled K/V.

        When ``quantize_retrieval=True``: dispatches per-head type to
        ``QuantizedKVCache`` (retrieval) or ``StreamingKVCache`` (streaming),
        then merges dequantized results.

        When ``quantize_retrieval=False`` (default): uses a pre-allocated fp16
        slab.  Slice assignment is O(1); gather+mask trim is applied to the
        slab for streaming heads at long context (Task 295(b) revert /
        Task 301).
        """
        B, H_kv, T_new, D = keys.shape

        if self._quantize_retrieval and self._retrieval_cache is not None:
            return self._update_quantized(keys, values, B, H_kv, T_new, D)

        # -- fp16 slab path --------------------------------------------------
        if self._keys is None:
            alloc = T_new + self._step
            self._keys = mx.zeros((B, H_kv, alloc, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H_kv, alloc, D), dtype=values.dtype)
            self._keys[:, :, :T_new] = keys
            self._values[:, :, :T_new] = values
            self._kv_len = T_new
        elif self._kv_len + T_new <= self._keys.shape[2]:
            self._keys[:, :, self._kv_len:self._kv_len + T_new] = keys
            self._values[:, :, self._kv_len:self._kv_len + T_new] = values
            self._kv_len += T_new
        else:
            # Slab exhausted — grow with headroom (rare: every _step tokens).
            new_alloc = self._kv_len + T_new + self._step
            new_k = mx.zeros((B, H_kv, new_alloc, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H_kv, new_alloc, D), dtype=values.dtype)
            new_k[:, :, :self._kv_len] = self._keys[:, :, :self._kv_len]
            new_v[:, :, :self._kv_len] = self._values[:, :, :self._kv_len]
            new_k[:, :, self._kv_len:self._kv_len + T_new] = keys
            new_v[:, :, self._kv_len:self._kv_len + T_new] = values
            self._keys = new_k
            self._values = new_v
            self._kv_len += T_new

        T_total = self._kv_len

        # -- Streaming-head trim (gather+mask) --------------------------------
        # Skip when: context still within capacity, no streaming heads, OR the
        # duo-split patch is active at decode (T_new == 1) — patched_sdpa will
        # re-fetch via get_streaming_kv / get_retrieval_kv directly.
        if (T_total > self.capacity and self._n_streaming > 0
                and not (_DUO_SPLIT_DECODE_SKIP and T_new == 1)):
            # Task 295(b) revert (validated by Task 301 microbench):
            # gather+mask via take_along_axis is 73% better peak memory than
            # slice+write because MLX fancy-indexed assignment is functional
            # (allocates a new tensor) — 8 writes × output-shape blows the budget.
            tc = _trim_indices_for(T_total, self.sink, self.window)
            stream_arr = tc["stream_arr"]
            retrieval_arr = tc["retrieval_arr"]
            stream_real_len = tc["stream_real_len"]

            gather_rows = [
                stream_arr if self._is_streaming[h] else retrieval_arr
                for h in range(H_kv)
            ]
            g_2d = mx.stack(gather_rows, axis=0)        # (H_kv, T_total)
            g = g_2d[None, :, :, None]                   # (1, H_kv, T_total, 1)
            g = mx.broadcast_to(g, (B, H_kv, T_total, D))

            out_k = mx.take_along_axis(self._keys[:, :, :T_total, :], g, axis=2)
            out_v = mx.take_along_axis(self._values[:, :, :T_total, :], g, axis=2)

            if stream_real_len < T_total:
                stream_mask = tc["stream_mask"]
                retrieval_mask = tc["retrieval_mask"]
                mask_rows = [
                    stream_mask if self._is_streaming[h] else retrieval_mask
                    for h in range(H_kv)
                ]
                mask = mx.stack(mask_rows, axis=0)  # (H_kv, T_total)
                # Scalar 0.0 broadcasts without allocating a fresh tensor
                # (Task 291 — saves ~32 MB/layer/step at T_total=16K).
                out_k = mx.where(mask[None, :, :, None], out_k, 0.0)
                out_v = mx.where(mask[None, :, :, None], out_v, 0.0)

            return out_k, out_v

        return self._keys[:, :, :T_total], self._values[:, :, :T_total]

    def _update_quantized(
        self, keys: mx.array, values: mx.array,
        B: int, H_kv: int, T_new: int, D: int,
    ) -> tuple[mx.array, mx.array]:
        """Dispatch retrieval heads to QuantizedKVCache, streaming to ring buffers."""
        ret_idx = self._retrieval_head_indices
        str_idx = self._streaming_head_indices

        ret_k: Optional[mx.array] = None
        ret_v: Optional[mx.array] = None
        if ret_idx and self._retrieval_cache is not None:
            k_ret = keys[:, ret_idx, :, :]
            v_ret = values[:, ret_idx, :, :]
            ret_k, ret_v = self._retrieval_cache.update_and_fetch(k_ret, v_ret)
            # QuantizedKVCache may return quantized tuples — dequantize.
            if isinstance(ret_k, tuple):
                rc = self._retrieval_cache
                ret_k = mx.dequantize(*ret_k, group_size=rc.group_size, bits=rc.bits)
                ret_v = mx.dequantize(*ret_v, group_size=rc.group_size, bits=rc.bits)

        str_k: Optional[mx.array] = None
        str_v: Optional[mx.array] = None
        if str_idx and self._streaming_head_caches:
            str_k_heads: list[mx.array] = []
            str_v_heads: list[mx.array] = []
            for h in str_idx:
                k_h = keys[:, h:h + 1, :, :]
                v_h = values[:, h:h + 1, :, :]
                sk, sv = self._streaming_head_caches[h].update_and_fetch(k_h, v_h)
                str_k_heads.append(sk)
                str_v_heads.append(sv)
            if str_k_heads:
                str_k = mx.concatenate(str_k_heads, axis=1)
                str_v = mx.concatenate(str_v_heads, axis=1)

        if ret_k is not None and str_k is not None:
            max_len = max(ret_k.shape[2], str_k.shape[2])
            out_k = mx.zeros((B, H_kv, max_len, D), dtype=keys.dtype)
            out_v = mx.zeros((B, H_kv, max_len, D), dtype=values.dtype)
            out_k[:, ret_idx, :ret_k.shape[2], :] = ret_k
            out_v[:, ret_idx, :ret_v.shape[2], :] = ret_v
            out_k[:, str_idx, :str_k.shape[2], :] = str_k
            out_v[:, str_idx, :str_v.shape[2], :] = str_v
            return out_k, out_v
        elif ret_k is not None:
            return ret_k, ret_v  # type: ignore[return-value]
        elif str_k is not None:
            return str_k, str_v  # type: ignore[return-value]
        else:
            return keys, values

    # ------------------------------------------------------------------
    # Split-attention accessors (for future duo-split SDPA patch)
    # ------------------------------------------------------------------

    def head_indices(self) -> tuple[list[int], list[int]]:
        """Return ``(streaming_head_indices, retrieval_head_indices)``."""
        return list(self._streaming_head_indices), list(self._retrieval_head_indices)

    def get_streaming_kv(self) -> Optional[tuple[mx.array, mx.array]]:
        """Return (K, V) for streaming heads only — sink + window slice."""
        if not self._streaming_head_indices or self._keys is None or self._kv_len == 0:
            return None
        T_total = self._kv_len
        keys = self._keys[:, self._streaming_head_indices, :T_total, :]
        values = self._values[:, self._streaming_head_indices, :T_total, :]
        if T_total <= self.capacity:
            return keys, values
        sink_k = keys[:, :, :self.sink, :]
        sink_v = values[:, :, :self.sink, :]
        win_k = keys[:, :, T_total - self.window:, :]
        win_v = values[:, :, T_total - self.window:, :]
        return mx.concatenate([sink_k, win_k], axis=2), mx.concatenate([sink_v, win_v], axis=2)

    def get_retrieval_kv(self) -> Optional[tuple[mx.array, mx.array]]:
        """Return (K, V) for retrieval heads only — full T_total tokens."""
        if not self._retrieval_head_indices or self._keys is None or self._kv_len == 0:
            return None
        T_total = self._kv_len
        return (
            self._keys[:, self._retrieval_head_indices, :T_total, :],
            self._values[:, self._retrieval_head_indices, :T_total, :],
        )

    # ------------------------------------------------------------------
    # Lifecycle helpers (trim, empty)
    # ------------------------------------------------------------------

    def is_trimmable(self) -> bool:
        return True

    def empty(self) -> bool:
        return self._kv_len == 0

    def trim(self, n: int) -> int:
        """Drop the first ``n`` tokens from the slab.  Returns tokens dropped."""
        if n <= 0 or self._keys is None or self._kv_len == 0:
            return 0
        n = min(n, self._kv_len)
        new_len = self._kv_len - n
        new_k = self._keys[:, :, n:self._kv_len, :]
        new_v = self._values[:, :, n:self._kv_len, :]
        new_alloc = new_len + self._step
        slabk = mx.zeros((new_k.shape[0], new_k.shape[1], new_alloc, new_k.shape[3]),
                         dtype=new_k.dtype)
        slabv = mx.zeros_like(slabk)
        slabk[:, :, :new_len] = new_k
        slabv[:, :, :new_len] = new_v
        self._keys = slabk
        self._values = slabv
        self._kv_len = new_len
        return n

    # ------------------------------------------------------------------
    # offset property — single source of truth (Task 299)
    # ------------------------------------------------------------------

    @property
    def offset(self) -> int:
        """Token count.  Property alias for ``_kv_len`` (Task 299)."""
        if self._quantize_retrieval and self._retrieval_cache is not None:
            return max(
                self._retrieval_cache.offset,
                max((c.offset for c in self._streaming_head_caches.values()), default=0),
            )
        return self._kv_len

    @offset.setter
    def offset(self, v: int) -> None:
        self._kv_len = int(v)

    # ------------------------------------------------------------------
    # state / meta_state — stock mlx_lm save/load interop
    # ------------------------------------------------------------------

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return ``(keys, values)`` sliced to the live token count."""
        if self._keys is None:
            return None, None
        return (
            self._keys[:, :, :self._kv_len],
            self._values[:, :, :self._kv_len],
        )

    @state.setter
    def state(self, v: tuple[mx.array | None, mx.array | None]) -> None:
        """Set state — needed for SnapKV ``compact_cache`` compatibility."""
        keys, values = v
        if keys is not None:
            self._kv_len = keys.shape[2]
            alloc = self._kv_len + self._step
            B, H, _, D = keys.shape
            self._keys = mx.zeros((B, H, alloc, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H, alloc, D), dtype=values.dtype)
            self._keys[:, :, :self._kv_len] = keys
            self._values[:, :, :self._kv_len] = values
        else:
            self._keys = None
            self._values = None
            self._kv_len = 0

    @property
    def meta_state(self) -> dict[str, str]:
        """Serialisable metadata — mirrors ``QuantizedKVCache.meta_state``."""
        return {
            "layer_idx": str(self.layer_idx),
            "n_kv_heads": str(self.n_kv_heads),
            "window": str(self.window),
            "sink": str(self.sink),
            "quantize_retrieval": str(int(self._quantize_retrieval)),
            "quant_bits": str(self._quant_bits),
            "group_size": str(self.group_size),
        }

    @classmethod
    def from_state(
        cls,
        state: tuple[mx.array | None, mx.array | None],
        meta_state: dict[str, str],
        policy: dict | None = None,
    ) -> "DuoKVCache":
        """Reconstruct a ``DuoKVCache`` from ``state`` + ``meta_state``.

        ``policy`` may be ``None`` — in that case a bare policy with no head
        classifications is used (all heads default to retrieval).  Pass the
        original policy for full behaviour restoration.
        """
        layer_idx = int(meta_state["layer_idx"])
        n_kv_heads = int(meta_state["n_kv_heads"])
        window = int(meta_state["window"])
        sink = int(meta_state["sink"])
        quantize_retrieval = bool(int(meta_state.get("quantize_retrieval", "0")))
        quant_bits = int(meta_state.get("quant_bits", "3"))
        group_size = int(meta_state.get("group_size", "64"))

        bare_policy: dict = policy or {"_lookup": {}, "n_heads": n_kv_heads}
        cache = cls(
            policy=bare_policy,
            layer_idx=layer_idx,
            n_kv_heads=n_kv_heads,
            window=window,
            sink=sink,
            quantize_retrieval=quantize_retrieval,
            group_size=group_size,
            quant_bits=quant_bits,
        )
        cache.state = state
        return cache
