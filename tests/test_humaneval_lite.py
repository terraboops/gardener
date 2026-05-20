"""Tests for the HumanEval-Lite quality gate (HX12.1)."""
from __future__ import annotations

import pytest

from gardener.bench.quality import HUMANEVAL_LITE, HumanEvalResult, run_humaneval_lite
from gardener.bench.quality.common import per_item_cleanup, format_chat_prompt


# --- pure-logic --------------------------------------------------------------

def test_humaneval_lite_has_20_problems():
    assert len(HUMANEVAL_LITE) == 20


def test_each_problem_has_required_fields():
    for prob in HUMANEVAL_LITE:
        assert "task_id" in prob and prob["task_id"].startswith("HE/")
        assert "prompt" in prob and prob["prompt"].startswith("def ")
        assert "test" in prob and "assert" in prob["test"]
        assert "entry_point" in prob


def test_task_ids_are_unique():
    ids = [p["task_id"] for p in HUMANEVAL_LITE]
    assert len(set(ids)) == len(ids)


def test_per_item_cleanup_does_not_raise():
    per_item_cleanup()  # smoke


def test_format_chat_prompt_fallback_when_no_template():
    class _BareTok:
        pass

    assert format_chat_prompt(_BareTok(), "hello world") == "hello world"


def test_humaneval_result_dataclass_defaults():
    r = HumanEvalResult(n_problems=20, n_passed=10, pass_rate=0.5)
    assert r.gate_min == 0.35
    assert r.passed_gate is False  # defaults until set by runner


# --- model integration -------------------------------------------------------

@pytest.mark.model
def test_runs_on_real_model_subset(loaded_model):
    """Run a small subset (3 problems) on the test model. Pass rate may be low
    on the 0.5B test model — we just confirm the runner executes end-to-end
    and returns a well-formed HumanEvalResult."""
    model, tok = loaded_model
    r = run_humaneval_lite(model, tok, n_problems=3, max_tokens=128,
                           gate_min=0.0)
    assert r.n_problems == 3
    assert 0 <= r.n_passed <= 3
    assert isinstance(r.pass_rate, float)
    assert len(r.per_problem) == 3
    for p in r.per_problem:
        assert "task_id" in p and "passed" in p
    assert r.passed_gate is True  # gate_min=0.0 → trivially passed
