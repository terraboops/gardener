"""TurboQuantKVCache (TQ3) — 3-bit quantized KV cache with WHT rotation
and Beta codebook. Ported (not imported) from omlx/turboquant_kv.py:1248-1765.

Memory: ~5.3× smaller than fp16 KV at the same context length.
Use case: agentic workloads needing save/load/fork/rewind at long context.
For pure decode speed, stock mlx_lm.QuantizedKVCache is now faster (mlx_lm
0.31.2 added native quantized_matmul); TQ3's superpower is the agentic API.

Reference map (NOT imported):
- WHT + Beta codebook: omlx/turboquant_kv.py:34-102
- Cache class core:    omlx/turboquant_kv.py:1248-1765
- fork():              omlx/turboquant_kv.py:1624
- rewind_to():         omlx/turboquant_kv.py:1648
- save_to_disk():      omlx/turboquant_kv.py:1670
- load_from_disk():    omlx/turboquant_kv.py:1722

# TODO(HX3.1): fused SDPA kernel — currently obsoleted by mlx_lm.quantized_matmul
# (2.5× faster at 4K per hypercar Task 332-334). Port when a compelling
# Gardener-specific reason emerges.
"""

from __future__ import annotations

import copy
import logging
import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import _BaseCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codebook generation (TurboQuant paper: arXiv:2504.19874)
#
# The correct distribution for coordinates of a randomly rotated unit vector
# in R^d has density: (1-x^2)^((d-3)/2) on [-1, 1].
# This is Beta((d-1)/2, (d-1)/2) after rescaling [0,1] → [-1,1].
# The codebook is DATA-INDEPENDENT — depends only on dim and bits.
# Reference: omlx/turboquant_kv.py:34-56
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _codebook(dim: int, bits: int, n_samples: int = 200_000) -> mx.array:
    """Optimal scalar codebook for TQ3's coordinate distribution.

    Per the paper (arXiv:2504.19874), after WHT rotation each coordinate of a
    unit vector follows density f(x) = C * (1-x^2)^((d-3)/2) on [-1,1].
    This is Beta((d-1)/2, (d-1)/2) mapped from [0,1] to [-1,1].

    lru_cache keyed by (dim, bits, n_samples): identical args return the same
    mx.array object, so the codebook allocation is truly a one-time cost.
    """
    n_levels = 1 << bits
    # Correct parameter: (d-1)/2, NOT d/2
    alpha = (dim - 1) / 2.0
    rng = np.random.default_rng(seed=0)
    samples = 2.0 * rng.beta(alpha, alpha, size=n_samples) - 1.0

    # Lloyd-Max optimal quantizer: iterate assignments → centroids
    centroids = np.linspace(samples.min(), samples.max(), n_levels)
    for _ in range(200):
        dists = np.abs(samples[:, None] - centroids[None, :])
        assignments = np.argmin(dists, axis=1)
        for j in range(n_levels):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = samples[mask].mean()

    cb = mx.array(sorted(centroids), dtype=mx.float32)
    mx.eval(cb)
    return cb


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# Reference: omlx/turboquant_kv.py:68-107
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _random_signs(dim: int, seed: int = 0) -> mx.array:
    """Random ±1 sign vector for randomized WHT. Cached per (dim, seed)."""
    key = mx.random.key(seed)
    uniform = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(uniform > 0.5, mx.ones(dim), -mx.ones(dim)).astype(mx.float32)
    mx.eval(signs)
    return signs


def _wht(x: mx.array) -> mx.array:
    """Walsh-Hadamard Transform via butterfly passes.

    Input:  (..., D) where D must be a power of 2.
    Output: (..., D) transformed, normalized by 1/sqrt(D).

    WHT decorrelates ALL dimensions (vs Givens which only mixes pairs) making
    the Beta distribution codebook valid.
    """
    shape = x.shape
    D = shape[-1]
    flat = x.reshape(-1, D).astype(mx.float32)

    h = 1
    while h < D:
        flat_r = flat.reshape(-1, D // (2 * h), 2, h)
        a = flat_r[:, :, 0, :]
        b = flat_r[:, :, 1, :]
        flat_r = mx.stack([a + b, a - b], axis=2)
        flat = flat_r.reshape(-1, D)
        h *= 2

    flat = flat / math.sqrt(D)
    return flat.reshape(shape)


def _apply_wht_rotation(x: mx.array, seed: int) -> mx.array:
    """Randomized WHT: sign flip → WHT. Decorrelates all dimensions.

    Accepts seed (int) rather than a pre-built signs array so the call-site
    signature is clean; _random_signs is cached so the lookup is O(1).
    """
    signs = _random_signs(x.shape[-1], seed)
    return _wht(x * signs)


def _apply_wht_rotation_inverse(x: mx.array, seed: int) -> mx.array:
    """Inverse randomized WHT: WHT → sign flip (WHT is its own inverse up to scale)."""
    signs = _random_signs(x.shape[-1], seed)
    return _wht(x) * signs


# ---------------------------------------------------------------------------
# Scalar codec helpers
# ---------------------------------------------------------------------------


def _quantize_to_codebook(x: mx.array, codebook: mx.array) -> mx.array:
    """Nearest-codeword scalar quantization.

    Args:
        x:        (..., D) float array (rotated, unit-scale coordinates)
        codebook: (n_levels,) float32 array of centroids (sorted ascending)

    Returns:
        (..., D) uint32 index array
    """
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = mx.zeros(x.shape, dtype=mx.uint32)
    for b_val in range(len(boundaries)):
        indices = indices + (x > boundaries[b_val]).astype(mx.uint32)
    return indices


def _dequantize_from_codebook(indices: mx.array, codebook: mx.array) -> mx.array:
    """Reconstruct float coordinates from codebook indices.

    Args:
        indices:  (..., D) uint32 index array
        codebook: (n_levels,) float32 centroids

    Returns:
        (..., D) float32 array of reconstructed coordinates
    """
    return codebook[indices]


# ---------------------------------------------------------------------------
# Packed-width helper (pure Python, no Metal)
# ---------------------------------------------------------------------------


def _packed_width(dim: int, bits: int) -> int:
    """Number of uint32 words needed to pack `dim` `bits`-wide indices."""
    return (dim * bits + 31) // 32


# ---------------------------------------------------------------------------
# High-level quantize / dequantize with norm decomposition
# ---------------------------------------------------------------------------


def _quantize_vectors(
    vectors: mx.array,
    codebook: mx.array,
    seed: int,
    bits: int,
) -> tuple[mx.array, mx.array]:
    """Quantize (..., D) vectors → (norms (...,), packed_indices (..., pw)).

    Pipeline: decompose norm → unit_vec → WHT rotate → scalar quantize.
    Storage format: per-vector L2 norm (fp32) + packed uint32 indices.
    """
    shape = vectors.shape
    D = shape[-1]
    flat = vectors.reshape(-1, D).astype(mx.float32)

    # Step 1: norm decomposition
    norms = mx.linalg.norm(flat, axis=-1, keepdims=False)
    safe_norms = mx.maximum(norms, 1e-10)
    unit = flat / safe_norms[..., None]

    # Step 2: WHT rotation
    rotated = _apply_wht_rotation(unit, seed)

    # Step 3: scalar quantize
    indices = _quantize_to_codebook(rotated, codebook)

    # Step 4: bit pack into uint32 words
    pw = _packed_width(D, bits)
    n_rows = flat.shape[0]
    packed = mx.zeros((n_rows, pw), dtype=mx.uint32)
    for i in range(D):
        bit_offset = i * bits
        word_idx = bit_offset // 32
        shift = bit_offset % 32
        idx_i = indices[:, i]
        packed[:, word_idx] = packed[:, word_idx] | (idx_i << shift)
        spill = shift + bits - 32
        if spill > 0 and word_idx + 1 < pw:
            packed[:, word_idx + 1] = packed[:, word_idx + 1] | (idx_i >> (bits - spill))

    return norms.reshape(shape[:-1]), packed.reshape(*shape[:-1], pw)


def _dequantize_vectors(
    norms: mx.array,
    packed: mx.array,
    codebook: mx.array,
    seed: int,
    bits: int,
    dim: int,
) -> mx.array:
    """Dequantize (norms, packed_indices) → (..., D) fp16 vectors.

    Pipeline (inverse): unpack → codebook lookup → inverse WHT → scale by norm.
    """
    batch_shape = norms.shape
    n_rows = int(np.prod(batch_shape))
    pw = packed.shape[-1]

    flat_packed = packed.reshape(n_rows, pw).astype(mx.uint32)

    # Unpack bit indices
    indices = mx.zeros((n_rows, dim), dtype=mx.uint32)
    mask = (1 << bits) - 1
    for i in range(dim):
        bit_offset = i * bits
        word_idx = bit_offset // 32
        shift = bit_offset % 32
        val = (flat_packed[:, word_idx] >> shift) & mask
        spill = shift + bits - 32
        if spill > 0 and word_idx + 1 < pw:
            val = val | ((flat_packed[:, word_idx + 1] << (bits - spill)) & mask)
        indices[:, i] = val

    # Codebook lookup → coordinates in rotated space
    coords = _dequantize_from_codebook(indices, codebook).astype(mx.float32)

    # Inverse WHT (sign flip then WHT)
    restored = _apply_wht_rotation_inverse(coords, seed)

    # Scale by norm
    flat_norms = norms.reshape(n_rows)
    result = (restored * flat_norms[..., None]).astype(mx.float16)
    return result.reshape(*batch_shape, dim)


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# Reference: omlx/turboquant_kv.py:1248-1765
# ---------------------------------------------------------------------------


class TurboQuantKVCache(_BaseCache):
    """KV cache with 3-bit TurboQuant codebook quantization.

    ~5.3× smaller than fp16 KV at the same context length. Suitable for
    agentic long-context workloads where save/load/fork/rewind matter more
    than last-5% decode throughput.

    Agentic API:
        fork()                 — independent deep copy, O(cache_size)
        rewind_to(offset)      — O(1) tail drop, no re-prefill
        save_to_disk(path)     — .npz freeze; returns metadata dict
        load_from_disk(path)   — thaw; returns token count

    Stock mlx_lm interop (save_prompt_cache / load_prompt_cache):
        state / state.setter   — array tree of quantized K/V
        meta_state             — (offset, bits, seed, min_quant_tokens, dequant_chunk_size)
        from_state(state, ms)  — reconstruct from the above (inherited from _BaseCache)
        trim(n)                — drop n tail tokens; returns actual count dropped
        is_trimmable()         — returns True
        empty()                — returns True if no tokens cached
    """

    def __init__(
        self,
        bits: int = 3,
        group_size: int = 64,  # accepted for API parity; TQ3 uses dim-level norms not groups
        seed: int = 0,
        min_quant_tokens: int = 512,
        dequant_chunk_size: int = 2048,
    ):
        self.bits = bits
        self.group_size = 0  # mlx_lm checks group_size on caches that have .bits; must be 0
        self.seed = seed
        self._min_quant_tokens = min_quant_tokens
        self._dequant_chunk_size = dequant_chunk_size

        self.offset = 0

        # Quantized storage
        self._k_norms: Optional[mx.array] = None   # (B, H, alloc)
        self._k_packed: Optional[mx.array] = None  # (B, H, alloc, pw)
        self._v_norms: Optional[mx.array] = None
        self._v_packed: Optional[mx.array] = None

        # fp16 warmup buffer (below min_quant_tokens)
        self._fp16_keys: Optional[mx.array] = None
        self._fp16_values: Optional[mx.array] = None
        self._quantized: bool = False

        # Geometric growth: initial allocation step
        self._step = 256

        # Codec state (dim is set on first call)
        self._dim: Optional[int] = None
        self._codebook_cache: Optional[mx.array] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_codebook(self, dim: int) -> mx.array:
        if self._codebook_cache is None or self._dim != dim:
            self._dim = dim
            self._codebook_cache = _codebook(dim, self.bits)
        return self._codebook_cache

    def _ensure_compressed_storage(self, B: int, H: int, new_end: int, pw: int) -> None:
        """Allocate or grow compressed storage with geometric doubling.

        At 64K: ~7 allocs vs ~250 with linear step=256.
        Reference: omlx/turboquant_kv.py:1285-1311
        """
        if self._k_norms is None:
            alloc = max(
                self._step,
                ((new_end + self._step - 1) // self._step) * self._step,
            )
            self._k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
            self._k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
            self._v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
            self._v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
        elif new_end > self._k_norms.shape[2]:
            cur = self._k_norms.shape[2]
            alloc = max(cur * 2, new_end)  # geometric doubling
            new_k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
            new_k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
            new_v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
            new_v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
            new_k_norms[:, :, : self.offset] = self._k_norms[:, :, : self.offset]
            new_k_packed[:, :, : self.offset] = self._k_packed[:, :, : self.offset]
            new_v_norms[:, :, : self.offset] = self._v_norms[:, :, : self.offset]
            new_v_packed[:, :, : self.offset] = self._v_packed[:, :, : self.offset]
            self._k_norms = new_k_norms
            self._k_packed = new_k_packed
            self._v_norms = new_v_norms
            self._v_packed = new_v_packed

    def _quantize_fp16_buffer(self) -> None:
        """Convert accumulated fp16 KV to quantized. Called once threshold is crossed.
        Reference: omlx/turboquant_kv.py:1313-1337
        """
        if self._fp16_keys is None or self._quantized:
            return
        B, H, T, D = self._fp16_keys.shape
        logger.info(
            "TurboQuantKVCache: quantizing fp16 buffer %d tokens "
            "(%d×%d heads, dim=%d) to %d-bit",
            T, B, H, D, self.bits,
        )
        cb = self._ensure_codebook(D)
        k_norms, k_packed = _quantize_vectors(self._fp16_keys, cb, self.seed, self.bits)
        v_norms, v_packed = _quantize_vectors(self._fp16_values, cb, self.seed, self.bits)
        pw = _packed_width(D, self.bits)
        alloc = ((T + self._step - 1) // self._step) * self._step
        self._k_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
        self._k_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
        self._v_norms = mx.zeros((B, H, alloc), dtype=mx.float32)
        self._v_packed = mx.zeros((B, H, alloc, pw), dtype=mx.uint32)
        self._k_norms[:, :, :T] = k_norms
        self._k_packed[:, :, :T] = k_packed
        self._v_norms[:, :, :T] = v_norms
        self._v_packed[:, :, :T] = v_packed
        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None

    # ------------------------------------------------------------------
    # Core cache protocol
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Store new K,V and return the full history (K, V) for attention.

        Prefill (T_new > 1):
            - Below min_quant_tokens: buffer in fp16.
            - At/above threshold: quantize, store compressed, dequantize for attention.
            - Short history (≤ dequant_chunk_size): dequantize full history.
            - Long history: dequantize in two passes (history + new chunk).

        Decode (T_new == 1):
            - In fp16 phase: append to fp16 buffer.
            - In quantized phase: quantize single token, append, dequantize full.

        Reference: omlx/turboquant_kv.py:1354-1460
        """
        B, H, T_new, D = keys.shape
        cb = self._ensure_codebook(D)

        if T_new > 1:
            # --- Prefill path ---
            if self.offset + T_new < self._min_quant_tokens and not self._quantized:
                # Below threshold: accumulate fp16
                if self._fp16_keys is None:
                    self._fp16_keys = keys
                    self._fp16_values = values
                else:
                    self._fp16_keys = mx.concatenate(
                        [self._fp16_keys, keys], axis=2
                    )
                    self._fp16_values = mx.concatenate(
                        [self._fp16_values, values], axis=2
                    )
                self.offset += T_new
                return self._fp16_keys, self._fp16_values

            # Cross quantization threshold — flush fp16 buffer first
            if not self._quantized and self._fp16_keys is not None:
                self._quantize_fp16_buffer()

            pw = _packed_width(D, self.bits)
            k_norms, k_packed = _quantize_vectors(keys, cb, self.seed, self.bits)
            v_norms, v_packed = _quantize_vectors(values, cb, self.seed, self.bits)

            new_end = self.offset + T_new
            self._ensure_compressed_storage(B, H, new_end, pw)

            self._k_norms[:, :, self.offset:new_end] = k_norms
            self._k_packed[:, :, self.offset:new_end] = k_packed
            self._v_norms[:, :, self.offset:new_end] = v_norms
            self._v_packed[:, :, self.offset:new_end] = v_packed
            self.offset = new_end
            self._quantized = True

            # Dequantize for attention
            all_k = _dequantize_vectors(
                self._k_norms[:, :, :self.offset],
                self._k_packed[:, :, :self.offset],
                cb, self.seed, self.bits, D,
            )
            all_v = _dequantize_vectors(
                self._v_norms[:, :, :self.offset],
                self._v_packed[:, :, :self.offset],
                cb, self.seed, self.bits, D,
            )
            return all_k, all_v

        else:
            # --- Decode path (T_new == 1) ---
            if not self._quantized and self._fp16_keys is not None:
                # Still in fp16 phase
                self._fp16_keys = mx.concatenate([self._fp16_keys, keys], axis=2)
                self._fp16_values = mx.concatenate(
                    [self._fp16_values, values], axis=2
                )
                self.offset += 1
                return self._fp16_keys, self._fp16_values

            # Quantized phase: quantize single token and append
            if not self._quantized and self._fp16_keys is None:
                # First token ever (no warmup buffer) — initialize
                self._fp16_keys = keys
                self._fp16_values = values
                self.offset += 1
                return self._fp16_keys, self._fp16_values

            k_norms, k_packed = _quantize_vectors(keys, cb, self.seed, self.bits)
            v_norms, v_packed = _quantize_vectors(values, cb, self.seed, self.bits)

            new_end = self.offset + 1
            pw = _packed_width(D, self.bits)
            self._ensure_compressed_storage(B, H, new_end, pw)

            self._k_norms[:, :, self.offset:new_end] = k_norms
            self._k_packed[:, :, self.offset:new_end] = k_packed
            self._v_norms[:, :, self.offset:new_end] = v_norms
            self._v_packed[:, :, self.offset:new_end] = v_packed
            self.offset = new_end

            # Dequantize for attention
            all_k = _dequantize_vectors(
                self._k_norms[:, :, :self.offset],
                self._k_packed[:, :, :self.offset],
                cb, self.seed, self.bits, D,
            )
            all_v = _dequantize_vectors(
                self._v_norms[:, :, :self.offset],
                self._v_packed[:, :, :self.offset],
                cb, self.seed, self.bits, D,
            )
            return all_k, all_v

    # ------------------------------------------------------------------
    # state / meta_state — mlx_lm interop
    # Reference: omlx/turboquant_kv.py:1469-1497
    # ------------------------------------------------------------------

    @property
    def state(self):
        """Return the serializable array tree.

        When in fp16 warmup phase, returns fp16 arrays directly.
        When quantized, returns ((k_norms, k_packed), (v_norms, v_packed)).
        Empty cache returns (None, None).
        """
        if self._fp16_keys is not None and not self._quantized:
            return (
                self._fp16_keys[:, :, : self.offset],
                self._fp16_values[:, :, : self.offset],
            )
        if self._k_norms is None:
            return None, None
        T = self.offset
        return (
            (self._k_norms[:, :, :T], self._k_packed[:, :, :T]),
            (self._v_norms[:, :, :T], self._v_packed[:, :, :T]),
        )

    @state.setter
    def state(self, v):
        if v is None or (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] is None):
            self._k_norms = self._k_packed = self._v_norms = self._v_packed = None
            self._fp16_keys = self._fp16_values = None
            self.offset = 0
            self._quantized = False
            return
        # Determine if the state is quantized ((k_norms, k_packed), ...) or fp16 (k, v)
        s0 = v[0]
        if isinstance(s0, (list, tuple)):
            # Quantized state: ((k_norms, k_packed), (v_norms, v_packed))
            (self._k_norms, self._k_packed), (self._v_norms, self._v_packed) = v
            self.offset = self._k_norms.shape[2]
            self._quantized = True
            self._fp16_keys = self._fp16_values = None
        else:
            # fp16 state: (k, v)
            self._fp16_keys, self._fp16_values = v
            self.offset = self._fp16_keys.shape[2]
            self._quantized = False
            self._k_norms = self._k_packed = self._v_norms = self._v_packed = None

    @property
    def meta_state(self):
        """Return a tuple of stringified ints — same contract as mlx_lm."""
        return (
            str(self.offset),
            str(self.bits),
            str(self.seed),
            str(self._min_quant_tokens),
            str(self._dequant_chunk_size),
        )

    @meta_state.setter
    def meta_state(self, v):
        if isinstance(v, (list, tuple)) and len(v) >= 5:
            self.offset = int(v[0])
            self.bits = int(v[1])
            self.seed = int(v[2])
            self._min_quant_tokens = int(v[3])
            self._dequant_chunk_size = int(v[4])
        elif isinstance(v, (list, tuple)) and len(v) >= 3:
            self.offset = int(v[0])
            self.bits = int(v[1])
            self.seed = int(v[2])
        elif isinstance(v, (list, tuple)) and len(v) >= 1:
            self.offset = int(v[0])
        elif v is not None and v != "":
            self.offset = int(v)
        # Rebuild codebook cache reference lazily (on next update_and_fetch)
        self._codebook_cache = None
        self._dim = None

    # ------------------------------------------------------------------
    # Additional _BaseCache protocol methods
    # ------------------------------------------------------------------

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Drop the last n tokens. Returns actual count dropped.
        Reference: omlx/turboquant_kv.py:1619-1622
        """
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        if self._k_norms is None and self._fp16_keys is None:
            return 0
        T = self.offset
        total = 0
        if self._k_norms is not None:
            total += (
                self._k_norms[:, :, :T].nbytes
                + self._k_packed[:, :, :T].nbytes
                + self._v_norms[:, :, :T].nbytes
                + self._v_packed[:, :, :T].nbytes
            )
        if self._fp16_keys is not None:
            total += self._fp16_keys[:, :, :T].nbytes + self._fp16_values[:, :, :T].nbytes
        return total

    # ------------------------------------------------------------------
    # Agentic API
    # ------------------------------------------------------------------

    def fork(self) -> "TurboQuantKVCache":
        """Create an independent deep copy for context branching.

        Unlike copy.copy() (which shares array references), fork() copies
        all internal arrays so both branches can generate independently.
        Codec (_codebook_cache) is immutable — shared, not copied.

        Cost: O(cache_size) — copies all compressed arrays.
        At 8K context with TQ3: ~0.3 MB per layer.

        Reference: omlx/turboquant_kv.py:1624-1646
        """
        new = copy.copy(self)
        if self._k_norms is not None:
            new._k_norms = mx.array(self._k_norms)
            new._k_packed = mx.array(self._k_packed)
            new._v_norms = mx.array(self._v_norms)
            new._v_packed = mx.array(self._v_packed)
        if self._fp16_keys is not None:
            new._fp16_keys = mx.array(self._fp16_keys)
            new._fp16_values = mx.array(self._fp16_values)
        # _codebook_cache is immutable — shallow copy is fine
        return new

    def rewind_to(self, target_offset: int) -> int:
        """O(1) context rewind — drop tokens after target_offset.

        Compressed storage is unchanged; just moves the offset pointer.
        Future writes will overwrite the discarded range. No re-prefill.

        Args:
            target_offset: keep only the first N tokens.

        Returns:
            Number of tokens kept (may be less if cache is shorter).

        Reference: omlx/turboquant_kv.py:1648-1668
        """
        target_offset = max(0, min(target_offset, self.offset))
        prev = self.offset
        self.offset = target_offset
        logger.info(
            "TurboQuantKVCache: rewound %d → %d (dropped %d tokens)",
            prev, target_offset, prev - target_offset,
        )
        return target_offset

    def save_to_disk(self, filepath: str) -> dict:
        """Freeze compressed KV cache to .npz.

        If the cache is in fp16 warmup phase, it is quantized first.
        Returns a metadata dict: {tokens, path, size_gb, bits}.

        Reference: omlx/turboquant_kv.py:1670-1720
        """
        # Flush fp16 warmup buffer before saving
        if not self._quantized and self._fp16_keys is not None:
            self._quantize_fp16_buffer()

        if self._k_norms is None or self.offset == 0:
            raise ValueError("Cannot save an empty TurboQuantKVCache")

        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        T = self.offset
        data = {
            "k_norms": np.array(self._k_norms[:, :, :T]),
            "k_packed": np.array(self._k_packed[:, :, :T]),
            "v_norms": np.array(self._v_norms[:, :, :T]),
            "v_packed": np.array(self._v_packed[:, :, :T]),
            "offset": np.int64(T),
            "bits": np.int64(self.bits),
            "seed": np.int64(self.seed),
            "min_quant_tokens": np.int64(self._min_quant_tokens),
            "dequant_chunk_size": np.int64(self._dequant_chunk_size),
            "quantized": np.bool_(True),
        }

        np.savez_compressed(filepath, **data)

        size_gb = sum(
            v.nbytes for v in data.values() if hasattr(v, "nbytes")
        ) / 1e9
        logger.info(
            "TurboQuantKVCache: saved %d tokens to %s (%.3f GB)", T, filepath, size_gb
        )
        return {"tokens": T, "path": filepath, "size_gb": size_gb, "bits": self.bits}

    def load_from_disk(self, filepath: str) -> int:
        """Thaw a frozen KV cache from .npz. Returns token count.

        Reference: omlx/turboquant_kv.py:1722-1764
        """
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        data = np.load(filepath)
        self.bits = int(data["bits"])
        self.seed = int(data["seed"])
        self._min_quant_tokens = int(data["min_quant_tokens"])
        self._dequant_chunk_size = int(data["dequant_chunk_size"])

        self._k_norms = mx.array(data["k_norms"])
        self._k_packed = mx.array(data["k_packed"])
        self._v_norms = mx.array(data["v_norms"])
        self._v_packed = mx.array(data["v_packed"])
        self.offset = int(data["offset"])
        self._quantized = True
        self._fp16_keys = None
        self._fp16_values = None

        # Eagerly prime the codebook so the first decode is fast
        pw = self._k_packed.shape[-1]
        D = pw * 32 // self.bits
        self._ensure_codebook(D)

        logger.info(
            "TurboQuantKVCache: loaded %d tokens from %s", self.offset, filepath
        )
        return self.offset
