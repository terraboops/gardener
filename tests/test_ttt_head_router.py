"""Tests for the TTT head router (HX8) — bit-equivalence mode."""
from __future__ import annotations

import json
import pytest
import mlx.core as mx

from gardener.mlxsuper.patches.ttt_head_router import TTTHeadRouter


def _write_policy(tmp_path, layers_dict):
    policy = {
        "model": "test", "schema_version": 1,
        "layers": layers_dict,
        # Pre-build _lookup the way load_duo_policy does (so the router can
        # consume the file directly even without invoking load_duo_policy).
        "_lookup": {
            (int(li), h): "streaming" if h in v["streaming_heads"] else "retrieval"
            for li, v in layers_dict.items()
            for h in v["streaming_heads"] + v["retrieval_heads"]
        },
    }
    # JSON can't represent tuple keys; write the layers dict only and re-derive
    # _lookup on load if needed. For tests we just keep it in-memory.
    serializable = {**policy, "_lookup": None}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(serializable))
    return path, policy   # return the in-memory policy for direct use too


def test_from_policy_loads_and_no_blocks_loaded(tmp_path):
    path, _ = _write_policy(tmp_path, {
        "0": {"streaming_heads": [0, 1], "retrieval_heads": [2, 3]},
    })
    router = TTTHeadRouter.from_policy(path)
    assert isinstance(router.policy, dict)
    assert router.ttt_blocks == {}


def test_streaming_heads_with_ttt_empty_when_no_blocks(tmp_path):
    # Build router with policy in-memory (avoid the tuple-key JSON issue).
    layers = {"0": {"streaming_heads": [0, 1], "retrieval_heads": [2, 3]}}
    policy = {
        "model": "t", "schema_version": 1, "layers": layers,
        "_lookup": {
            (0, 0): "streaming", (0, 1): "streaming",
            (0, 2): "retrieval", (0, 3): "retrieval",
        },
    }
    router = TTTHeadRouter(policy=policy)
    assert router.streaming_heads_with_ttt(0) == []


def test_streaming_heads_with_ttt_filters_to_loaded_blocks():
    policy = {
        "_lookup": {
            (0, 0): "streaming", (0, 1): "streaming",
            (0, 2): "retrieval", (0, 3): "retrieval",
            (1, 0): "streaming",
        },
    }
    blocks = {(0, 0): "block-0-0-stub", (1, 0): "block-1-0-stub"}
    router = TTTHeadRouter(policy=policy, ttt_blocks=blocks)
    assert router.streaming_heads_with_ttt(0) == [0]
    assert router.streaming_heads_with_ttt(1) == [0]


def test_route_sdpa_bit_equivalence_calls_original():
    policy = {"_lookup": {(0, 0): "streaming", (0, 1): "retrieval"}}
    router = TTTHeadRouter(policy=policy)   # no blocks -> bit-equivalence
    called = {"n": 0}
    sentinel = mx.array([[1.0]])
    def original_sdpa(q, k, v, cache=None, scale=1.0, mask=None):
        called["n"] += 1
        return sentinel
    q = mx.zeros((1, 2, 1, 4)); k = mx.zeros((1, 2, 1, 4)); v = mx.zeros((1, 2, 1, 4))
    out = router.route_sdpa(q, k, v, layer_idx=0, original_sdpa=original_sdpa)
    assert called["n"] == 1
    assert mx.array_equal(out, sentinel)
    assert router.telemetry()["routed_dense"] == 1
    assert router.telemetry()["routed_ttt"] == 0


def test_route_with_ttt_raises_clean_not_implemented():
    """When blocks ARE loaded for a streaming head, _route_with_ttt fires —
    and must raise NotImplementedError with the upstream-pending message."""
    policy = {"_lookup": {(0, 0): "streaming"}}
    blocks = {(0, 0): "block-stub"}
    router = TTTHeadRouter(policy=policy, ttt_blocks=blocks)
    q = mx.zeros((1, 1, 1, 4)); k = mx.zeros((1, 1, 1, 4)); v = mx.zeros((1, 1, 1, 4))
    def original_sdpa(*a, **kw): return None
    with pytest.raises(NotImplementedError, match="Cycle 2"):
        router.route_sdpa(q, k, v, layer_idx=0, original_sdpa=original_sdpa)


def test_reset_state_clears_telemetry():
    policy = {"_lookup": {(0, 0): "streaming"}}
    router = TTTHeadRouter(policy=policy)
    def fake(*a, **kw): return None
    q = mx.zeros((1, 1, 1, 4)); k = mx.zeros((1, 1, 1, 4)); v = mx.zeros((1, 1, 1, 4))
    router.route_sdpa(q, k, v, layer_idx=0, original_sdpa=fake)
    assert router.telemetry()["routed_dense"] == 1
    router.reset_state()
    assert router.telemetry()["routed_dense"] == 0


def test_scan_block_dir_picks_up_safetensors(tmp_path):
    (tmp_path / "block_L0_H0.safetensors").write_bytes(b"")
    (tmp_path / "block_L3_H7.safetensors").write_bytes(b"")
    (tmp_path / "ignored.txt").write_text("nope")
    from gardener.mlxsuper.patches.ttt_head_router import _scan_block_dir
    blocks = _scan_block_dir(tmp_path)
    assert (0, 0) in blocks
    assert (3, 7) in blocks
    assert len(blocks) == 2


def test_scan_missing_dir_returns_empty():
    from gardener.mlxsuper.patches.ttt_head_router import _scan_block_dir
    from pathlib import Path
    assert _scan_block_dir(Path("/does/not/exist")) == {}
