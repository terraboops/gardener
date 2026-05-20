"""Tests for the MMLU-Pro quality gate (HX12.2)."""
from __future__ import annotations

import pytest

from gardener.bench.quality import (
    MMLUProResult, extract_answer, format_prompt, run_mmlu_pro,
)


# --- prompt template + answer extraction (pure-logic) ------------------------

def test_format_prompt_includes_question_and_options():
    q = {"question": "What is 2+2?", "options": ["3", "4", "5", "6"],
         "answer": "B", "category": "math"}
    p = format_prompt(q)
    assert "What is 2+2?" in p
    assert "A. 3" in p and "B. 4" in p and "D. 6" in p
    assert "math" in p
    assert 'output the answer in the format of "The answer is (X)"' in p

def test_extract_stage_1_the_answer_is_with_parens():
    assert extract_answer("So the answer is (C).") == "C"

def test_extract_stage_1_the_answer_is_no_parens():
    assert extract_answer("Therefore The answer is D.") == "D"

def test_extract_stage_2_answer_colon():
    assert extract_answer("blah blah\nAnswer: A\n") == "A"

def test_extract_stage_3_last_letter_fallback():
    assert extract_answer("I think it's between B and E... I'll go with E.") == "E"

def test_extract_returns_none_when_no_letter():
    assert extract_answer("There's no letter answer here at all.") is None


# --- runner with injected questions (no HF fetch) ---------------------------

def test_run_with_injected_questions_no_model(monkeypatch):
    """Inject pre-formed questions and stub generate to return a canned
    response. Exercises the runner shape without HF or model."""
    import gardener.bench.quality.mmlu_pro as mod

    fake_qs = [
        {"question": "Q1", "options": ["X", "Y"], "answer": "B", "category": "math"},
        {"question": "Q2", "options": ["X", "Y"], "answer": "A", "category": "math"},
    ]
    def fake_generate(model, tok, prompt, max_tokens, sampler, verbose):
        return "...think...\nThe answer is (B)."
    monkeypatch.setattr(mod, "generate", fake_generate)
    # Stub tokenizer.apply_chat_template: just return the user content.
    class _Tok:
        def apply_chat_template(self, msgs, *, tokenize, add_generation_prompt, **kw):
            return msgs[-1]["content"]
    class _Model: pass
    r = run_mmlu_pro(_Model(), _Tok(), questions=fake_qs, gate_min=0.0)
    assert r.n_questions == 2
    # 1 of 2 correct (Q1 expects B, model says B; Q2 expects A, model says B)
    assert r.n_correct == 1
    assert r.accuracy == 0.5
    assert "math" in r.by_category
    assert r.by_category["math"]["accuracy"] == 0.5
    assert r.passed_gate is True   # gate_min=0.0


# --- model integration (very small N because each Q is slow) -----------------

@pytest.mark.model
def test_runs_on_real_model_tiny_subset(loaded_model):
    """Run 2 MMLU-Pro questions on the test model. The 0.5B model is unlikely
    to score well; the test just confirms end-to-end execution + result
    shape."""
    model, tok = loaded_model
    # Inject tiny subset to skip HuggingFace download in CI.
    tiny = [
        {"question": "What is 2 + 2?",
         "options": ["3", "4", "5", "6", "7", "8", "9", "10"],
         "answer": "B", "category": "math"},
        {"question": "Capital of France?",
         "options": ["London", "Paris", "Berlin", "Rome", "Madrid",
                     "Vienna", "Lisbon", "Athens"],
         "answer": "B", "category": "geography"},
    ]
    r = run_mmlu_pro(model, tok, questions=tiny, max_tokens=64, gate_min=0.0)
    assert r.n_questions == 2
    assert 0 <= r.n_correct <= 2
    assert r.passed_gate is True  # gate_min=0.0
    assert isinstance(r.accuracy, float)
