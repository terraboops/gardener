"""TTT-Linear head router — bit-equivalence mode (HX8).

Per-(layer, head) dispatcher backed by a DuoAttention policy + optional
TTT-Linear blocks. When no blocks are loaded (Cycle 1, the only mode this
port ships), all heads route through the original mlx_lm SDPA — the router
is a no-op pass-through with telemetry.

Cycle 2 (active routing: streaming-tagged heads with loaded blocks pass
through a TTT-Linear recurrence; retrieval heads keep softmax SDPA) is
upstream-pending in hypercar — the omlx reference's `_route_with_ttt` also
raises NotImplementedError. Gardener mirrors that.

Reference map (NOT imported):
- omlx/patches/ttt_head_router.py:41-156
- gardener/mlxsuper/duo_policies/*.json (reused for head classification)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("gardener.mlxsuper.patches.ttt_head_router")


@dataclass
class TTTHeadRouter:
    policy: dict                     # the DuoAttention policy dict (load_duo_policy result)
    ttt_blocks: dict = field(default_factory=dict)   # {(layer_idx, head_idx) -> opaque block}
    _telemetry: dict = field(default_factory=lambda: {"routed_dense": 0, "routed_ttt": 0})

    @classmethod
    def from_policy(
        cls,
        policy_path: str | Path,
        ttt_blocks: dict | None = None,
        ttt_dir: str | Path | None = None,
    ) -> "TTTHeadRouter":
        """Build a router from a DuoAttention policy JSON.

        If `ttt_blocks` (a dict mapping (layer, head) → block) is provided,
        active routing is possible (but currently raises NotImplementedError
        per Cycle-2 upstream-pending status). If `ttt_dir` is provided, the
        port scans for `block_L<layer>_H<head>.safetensors` files and loads
        them — schema not finalized upstream; this is a stub.
        """
        import json
        policy = json.loads(Path(policy_path).read_text())
        blocks: dict = dict(ttt_blocks or {})
        if ttt_dir is not None:
            blocks.update(_scan_block_dir(Path(ttt_dir)))
        return cls(policy=policy, ttt_blocks=blocks)

    def reset_state(self) -> None:
        """Clear per-sequence telemetry. Call between requests."""
        self._telemetry = {"routed_dense": 0, "routed_ttt": 0}

    def streaming_heads_with_ttt(self, layer_idx: int) -> list[int]:
        """Heads in `layer_idx` that are tagged streaming AND have a loaded
        TTT block. Returns [] in bit-equivalence mode (no blocks loaded)."""
        lookup = self.policy.get("_lookup", {})
        return [h for (li, h), klass in lookup.items()
                if li == layer_idx
                and klass == "streaming"
                and (layer_idx, h) in self.ttt_blocks]

    def route_sdpa(
        self,
        queries,                # (B, H, T, D)
        keys,                   # (B, H_kv, T_kv, D)
        values,                 # (B, H_kv, T_kv, D)
        cache: Any = None,
        scale: float = 1.0,
        mask: Any = None,
        *,
        layer_idx: int,
        original_sdpa: Callable,
    ):
        """Per-call SDPA dispatch.

        Bit-equivalence: always delegates to `original_sdpa`. Telemetry
        records that this layer's call routed dense. When per-(layer, head)
        TTT blocks are loaded, the streaming heads would split off via
        `_route_with_ttt` and the remaining heads via `original_sdpa` —
        Cycle 2, upstream-pending. The current code path falls through to
        original_sdpa unconditionally.
        """
        if self.streaming_heads_with_ttt(layer_idx):
            return self._route_with_ttt(
                queries, keys, values, cache=cache, scale=scale, mask=mask,
                layer_idx=layer_idx, original_sdpa=original_sdpa)
        self._telemetry["routed_dense"] += 1
        return original_sdpa(queries, keys, values,
                             cache=cache, scale=scale, mask=mask)

    def _route_with_ttt(self, *args, **kwargs):
        raise NotImplementedError(
            "TTT-Linear active routing — upstream Cycle 2 pending. "
            "The bit-equivalence path (no TTT blocks loaded) works today; "
            "actively routing streaming-tagged heads through a TTT-Linear "
            "recurrence is not yet implemented in either hypercar or Gardener."
        )

    def telemetry(self) -> dict:
        return dict(self._telemetry)


def _scan_block_dir(d: Path) -> dict:
    """Look for block_L<int>_H<int>.safetensors files. Stub: returns the file
    paths as opaque values — actual block loading awaits Cycle-2 schema."""
    out: dict = {}
    if not d.exists():
        return out
    pattern = re.compile(r"^block_L(\d+)_H(\d+)\.safetensors$")
    for path in sorted(d.glob("block_L*_H*.safetensors")):
        m = pattern.match(path.name)
        if not m:
            continue
        layer = int(m.group(1))
        head = int(m.group(2))
        out[(layer, head)] = path   # stub: opaque
    return out
