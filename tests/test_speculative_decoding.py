"""Tests for speculative decoding wiring (HX9).

The mlx_lm 0.31.2 stream_generate API ships speculative decoding natively;
this test pins the wiring (vocab mismatch → ValueError; α telemetry tracked;
non-spec path unchanged)."""
from __future__ import annotations

import pytest

from gardener.harness.mlx_agent import MLXAgentRunner
from gardener.pipeline.ir import Node


# --- pure-logic: vocab check / telemetry surface -----------------------------

class _FakeTokenizer:
    """Minimal tokenizer with a vocab attribute. Matches mlx_lm's
    TokenizerWrapper duck-type just enough for the vocab check."""
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self.eos_token_id = 0
        # mlx_lm TokenizerWrapper exposes vocab_size via len(self) typically.
    def __len__(self):
        return self._vocab_size
    def encode(self, s):
        return [1] * 3
    def decode(self, ids):
        return ""


class _FakeModel:
    """Stand-in for an mlx.nn.Module that supports forward + has a tied head
    field. We don't actually run inference in the pure test."""
    pass


def test_no_drafter_keeps_today_behavior():
    """Without draft_model, MLXAgentRunner stays in its single-generate path
    and spec_decode_stats reports zero spec activity."""
    runner = MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000))
    stats = runner.spec_decode_stats()
    assert stats["enabled"] is False
    assert stats["accepted_from_draft"] == 0
    assert stats["total_tokens"] == 0


def test_vocab_mismatch_raises_at_construction():
    """Construction with mismatched-vocab drafter raises ValueError so the
    misconfig surfaces immediately."""
    with pytest.raises(ValueError, match="vocab"):
        MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000),
                       draft_model=_FakeModel(),
                       draft_tokenizer=_FakeTokenizer(7000))


def test_missing_draft_tokenizer_raises():
    """draft_model without draft_tokenizer raises ValueError at construction."""
    with pytest.raises(ValueError, match="draft_tokenizer"):
        MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000),
                       draft_model=_FakeModel())


def test_spec_decode_stats_alpha_calculation():
    """alpha = accepted_from_draft / total_tokens (clamped to 0 when no
    tokens generated)."""
    runner = MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000),
                             draft_model=_FakeModel(),
                             draft_tokenizer=_FakeTokenizer(8000))
    # Manually push stats (simulating what __call__ would record).
    runner._spec_stats["accepted_from_draft"] = 30
    runner._spec_stats["total_tokens"] = 100
    s = runner.spec_decode_stats()
    assert s["enabled"] is True
    assert s["accepted_from_draft"] == 30
    assert s["total_tokens"] == 100
    assert abs(s["alpha"] - 0.30) < 1e-9


def test_spec_decode_alpha_is_zero_when_no_tokens():
    runner = MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000),
                             draft_model=_FakeModel(),
                             draft_tokenizer=_FakeTokenizer(8000))
    s = runner.spec_decode_stats()
    assert s["alpha"] == 0.0


def test_spec_decode_stats_enabled_flag():
    """enabled is True iff a draft_model was provided."""
    runner_no_draft = MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000))
    runner_with_draft = MLXAgentRunner(_FakeModel(), _FakeTokenizer(8000),
                                        draft_model=_FakeModel(),
                                        draft_tokenizer=_FakeTokenizer(8000))
    assert runner_no_draft.spec_decode_stats()["enabled"] is False
    assert runner_with_draft.spec_decode_stats()["enabled"] is True


# --- AgentProfile field --------------------------------------------------------

def test_agent_profile_accepts_draft_model_field():
    from gardener.subagents import AgentProfile
    p = AgentProfile(name="x", system_prompt="y", warmup_text="z",
                     params={"max_tokens": 4},
                     draft_model="some/hf-id", num_draft_tokens=4)
    assert p.draft_model == "some/hf-id"
    assert p.num_draft_tokens == 4


def test_agent_profile_default_no_draft():
    """Default AgentProfile has no drafter configured."""
    from gardener.subagents import AgentProfile
    p = AgentProfile(name="x", system_prompt="y", warmup_text="z")
    assert p.draft_model is None
    assert p.num_draft_tokens == 8


# --- model integration (requires @pytest.mark.model) -------------------------

@pytest.mark.model
def test_runner_without_drafter_still_responds(loaded_model):
    model, tok = loaded_model
    runner = MLXAgentRunner(model, tok)
    node = Node(id="a", kind="agent", agent="answerer",
                params={"system_prompt": "Answer one word.", "max_tokens": 4})
    out = runner(node, {"question": "Capital of Japan?"})
    assert isinstance(out, str) and len(out) > 0
    s = runner.spec_decode_stats()
    assert s["enabled"] is False
    assert s["total_tokens"] == 0
