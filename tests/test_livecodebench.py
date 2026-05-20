"""Tests for LiveCodeBench quality gate (HX12.4)."""
from __future__ import annotations

import pytest

from gardener.bench.quality.livecodebench import (
    LCBProblem, LCBProblemResult, LCBResult,
    LCB_RETRIES, LCB_RETRY_TEMP, LCB_TIMEOUT_S,
    load_livecodebench, format_prompt, extract_last_code_block,
    execute_code, run_livecodebench,
)


# --- dataset --------------------------------------------------------------

def test_load_returns_at_least_n_problems():
    probs = load_livecodebench(n=5)
    assert len(probs) == 5
    for p in probs:
        assert isinstance(p, LCBProblem)
        assert p.question_id
        assert p.description
        assert len(p.inputs) >= 1 and len(p.outputs) == len(p.inputs)

def test_load_is_deterministic_per_seed():
    a = load_livecodebench(n=3, seed=42)
    b = load_livecodebench(n=3, seed=42)
    assert [p.question_id for p in a] == [p.question_id for p in b]


# --- prompt template ------------------------------------------------------

def test_format_prompt_includes_required_phrases():
    p = LCBProblem(question_id="x", difficulty="easy",
                    description="print hello", inputs=[""], outputs=["hello"])
    s = format_prompt(p)
    assert "Solve the following programming problem in Python" in s
    assert "stdin" in s and "stdout" in s
    assert "print hello" in s
    assert "```python" in s
    assert s.rstrip().endswith("Solution:")


# --- extractor (LAST match) ----------------------------------------------

def test_extracts_last_python_block_when_multiple():
    response = (
        "Here's a first attempt:\n"
        "```python\nprint('wrong')\n```\n\n"
        "Actually wait, here's the correct one:\n"
        "```python\nprint('right')\n```"
    )
    assert extract_last_code_block(response).strip() == "print('right')"

def test_extracts_generic_block_when_no_python_tag():
    response = "```\nprint('x')\n```"
    assert extract_last_code_block(response).strip() == "print('x')"

def test_extracts_unclosed_block_truncation_case():
    response = "Solution:\n```python\nprint('partial')"
    out = extract_last_code_block(response)
    assert "print('partial')" in out

def test_returns_empty_when_no_code_block():
    assert extract_last_code_block("no code here") == ""


# --- subprocess execution ------------------------------------------------

def test_execute_simple_program_returns_output():
    stdout, ok, err = execute_code("print('hello')", "")
    assert ok is True
    assert stdout.strip() == "hello"
    assert err is None

def test_execute_with_stdin_input():
    stdout, ok, err = execute_code(
        "import sys\nprint(sys.stdin.read().strip().upper())",
        "abc",
    )
    assert ok is True and stdout.strip() == "ABC"

def test_execute_runtime_error_reports_failure():
    stdout, ok, err = execute_code("raise ValueError('boom')", "")
    assert ok is False
    assert err is not None and "ValueError" in err

def test_execute_timeout_caps_runaway_loop():
    stdout, ok, err = execute_code(
        "while True:\n    pass",
        "",
        timeout_s=0.5,
    )
    assert ok is False
    assert err is not None and ("timeout" in err.lower() or "killed" in err.lower())


# --- run_livecodebench with injected problems (no model) -----------------

def test_runner_passes_when_greedy_solves(monkeypatch):
    """Stub generate so the very first response solves the problem; runner
    should record 1 attempt + pass."""
    import gardener.bench.quality.livecodebench as mod
    monkeypatch.setattr(mod, "generate",
        lambda *a, **k: "```python\nprint('hi')\n```")
    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass
    probs = [LCBProblem(question_id="p", difficulty="easy",
                         description="print hi", inputs=[""], outputs=["hi"])]
    r = run_livecodebench(_M(), _Tok(), problems=probs, gate_min=0.0)
    assert r.n_problems == 1
    assert r.n_passed == 1
    assert r.per_problem[0].attempts == 1
    assert r.per_problem[0].passed is True
    assert r.passed_gate is True

def test_runner_retries_when_greedy_fails(monkeypatch):
    """First (greedy) generation produces wrong output; first retry produces
    correct one. Runner records attempts=2."""
    import gardener.bench.quality.livecodebench as mod
    responses = iter([
        "```python\nprint('wrong')\n```",         # greedy
        "```python\nprint('right')\n```",         # retry 1 — wins
    ])
    monkeypatch.setattr(mod, "generate", lambda *a, **k: next(responses))
    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass
    probs = [LCBProblem(question_id="p", difficulty="easy",
                         description="print right", inputs=[""], outputs=["right"])]
    r = run_livecodebench(_M(), _Tok(), problems=probs, gate_min=0.0,
                          retries=2)
    assert r.n_passed == 1
    assert r.per_problem[0].attempts == 2

def test_runner_records_failure_after_retries_exhausted(monkeypatch):
    import gardener.bench.quality.livecodebench as mod
    monkeypatch.setattr(mod, "generate",
        lambda *a, **k: "```python\nprint('wrong')\n```")
    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass
    probs = [LCBProblem(question_id="p", difficulty="easy",
                         description="print right", inputs=[""], outputs=["right"])]
    r = run_livecodebench(_M(), _Tok(), problems=probs, gate_min=0.0,
                          retries=2)
    assert r.n_passed == 0
    assert r.per_problem[0].attempts == 3   # 1 greedy + 2 retries
    assert r.per_problem[0].passed is False
    assert r.passed_gate is True   # gate_min=0.0


# --- model integration (single problem) ---------------------------------

@pytest.mark.model
def test_runs_one_real_problem_on_test_model(loaded_model):
    """One actual LCB problem on the test model. The 0.5B model is unlikely
    to solve a competition problem; we just confirm the runner executes
    end-to-end + returns a well-formed result."""
    model, tok = loaded_model
    probs = load_livecodebench(n=1)
    r = run_livecodebench(model, tok, problems=probs, max_tokens=256,
                           gate_min=0.0, retries=0)
    assert r.n_problems == 1
    assert isinstance(r.pass_rate, float)
    assert len(r.per_problem) == 1
    assert r.passed_gate is True   # gate_min=0.0
