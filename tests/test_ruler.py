"""Tests for the RULER quality gate (HX12.3)."""
from __future__ import annotations

import pytest

from gardener.bench.quality.ruler import (
    RulerTaskResult, RulerSuiteResult, RULER_QUICK_SUITE, RULER_FULL_SUITE,
    generate_multi_key_niah, generate_variable_tracking, generate_frequent_word,
    run_ruler_task, run_ruler_suite,
)


# --- generator shape ---------------------------------------------------------

def test_multi_key_niah_generator_returns_required_fields():
    # Stub tokenizer: encode returns char-level list, decode joins them.
    class _Tok:
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids): return bytes(ids).decode("utf-8", errors="replace")

    t = generate_multi_key_niah(_Tok(), target_tokens=2048, num_keys=3, seed=101)
    assert t["task_type"] == "multi_key_niah"
    assert isinstance(t["context"], str) and len(t["context"]) > 0
    assert isinstance(t["question"], str)
    assert isinstance(t["expected"], list) and len(t["expected"]) == 3
    assert all(isinstance(v, str) for v in t["expected"])
    # Each expected value should appear in the context (otherwise impossible).
    for v in t["expected"]:
        assert v in t["context"]


def test_variable_tracking_generator_chain_resolves():
    class _Tok:
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids): return bytes(ids).decode("utf-8", errors="replace")

    t = generate_variable_tracking(_Tok(), target_tokens=2048, chain_length=4, seed=304)
    assert t["task_type"] == "variable_tracking"
    assert len(t["expected"]) >= 1
    # The canonical expected value MUST appear in the context.
    assert any(v in t["context"] for v in t["expected"])


def test_frequent_word_generator_marker_is_majority():
    class _Tok:
        def encode(self, text): return list(text.encode("utf-8"))
        def decode(self, ids): return bytes(ids).decode("utf-8", errors="replace")

    t = generate_frequent_word(_Tok(), target_tokens=2048, num_target_words=5, seed=201)
    assert t["task_type"] == "frequent_word"
    canonical = t["expected"][0]
    if "marker_words" in t.get("params", {}):
        markers = t["params"]["marker_words"]
        canon_count = t["context"].count(canonical)
        for w in markers:
            if w.lower() != canonical.lower():
                assert canon_count >= t["context"].count(w)


# --- scoring logic (substring AND vs OR) ------------------------------------

def test_run_ruler_task_multi_key_scoring_and_logic(monkeypatch):
    """multi_key_niah uses AND — fraction of expected values found."""
    import gardener.bench.quality.ruler as mod
    task = {"context": "ignored", "question": "q",
            "expected": ["A1", "B2", "C3"],
            "task_type": "multi_key_niah",
            "params": {"target_tokens": 1024, "num_keys": 3, "seed": 0}}
    # Stub generate: 2 of 3 expected substrings present.
    monkeypatch.setattr(mod, "generate",
        lambda *a, **k: "found A1 and B2 only")
    class _Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]
    class _M: pass
    r = run_ruler_task(_M(), _Tok(), task)
    assert r.task_type == "multi_key_niah"
    assert abs(r.accuracy - 2/3) < 1e-9
    assert r.passed is True   # >= 0.5


def test_run_ruler_task_variable_tracking_or_logic(monkeypatch):
    """variable_tracking uses OR — 1.0 if any variant matches, else 0."""
    import gardener.bench.quality.ruler as mod
    task = {"context": "x", "question": "q",
            "expected": ["XKCD-9921", "xkcd-9921", "Xkcd-9921"],
            "task_type": "variable_tracking",
            "params": {"target_tokens": 1024, "chain_length": 4, "seed": 0}}
    monkeypatch.setattr(mod, "generate",
        lambda *a, **k: "answer: xkcd-9921 here")
    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass
    r = run_ruler_task(_M(), _Tok(), task)
    assert r.accuracy == 1.0
    assert r.passed is True


def test_run_ruler_task_frequent_word_or_logic_zero_on_miss(monkeypatch):
    import gardener.bench.quality.ruler as mod
    task = {"context": "x", "question": "q",
            "expected": ["ZEPHYR", "zephyr"],
            "task_type": "frequent_word",
            "params": {"target_tokens": 1024, "num_target_words": 5, "seed": 0}}
    monkeypatch.setattr(mod, "generate", lambda *a, **k: "QUARTZ")
    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass
    r = run_ruler_task(_M(), _Tok(), task)
    assert r.accuracy == 0.0
    assert r.passed is False


# --- suite gate aggregator --------------------------------------------------

def test_quick_suite_has_six_tasks_with_expected_types():
    suite = RULER_QUICK_SUITE
    assert len(suite) == 6
    task_types = [t["generator"] for t in suite]
    assert task_types.count("multi_key_niah") == 2
    assert task_types.count("variable_tracking") == 2
    assert task_types.count("frequent_word") == 2


def test_full_suite_has_fifteen_tasks():
    assert len(RULER_FULL_SUITE) == 15


def test_suite_gate_passes_when_both_thresholds_met(monkeypatch):
    """Aggregator passes when mk@16K >= gate_mk_min AND vt@4K >= gate_vt_min."""
    import gardener.bench.quality.ruler as mod
    # Tiny suite of two task descriptors, just enough to hit the two gates.
    suite = [
        {"context": "x", "question": "q",
         "expected": ["k1", "k2", "k3"],
         "task_type": "multi_key_niah",
         "params": {"target_tokens": 16384, "num_keys": 3, "seed": 0}},
        {"context": "x", "question": "q",
         "expected": ["v"],
         "task_type": "variable_tracking",
         "params": {"target_tokens": 4096, "chain_length": 4, "seed": 0}},
    ]

    # Pre-build task dicts (run_ruler_suite calls generators, so we need to
    # stub the generator functions as well as the generate function).
    # Easier approach: pre-bake tasks and use a wrapper that intercepts
    # run_ruler_task directly.
    responses = iter(["k1 k2 k3", "v"])
    monkeypatch.setattr(mod, "generate", lambda *a, **k: next(responses))

    # Stub generators to return pre-built task dicts (avoiding tokenizer dep)
    monkeypatch.setattr(mod, "_GENERATORS", {
        "multi_key_niah": lambda **kw: suite[0],
        "variable_tracking": lambda **kw: suite[1],
    })

    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass

    # Minimal suite spec matching the pre-built tasks
    spec_suite = [
        {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 3, "seed": 0},
        {"generator": "variable_tracking", "target_tokens": 4096, "chain_length": 4, "seed": 0},
    ]
    r = run_ruler_suite(_M(), _Tok(), suite=spec_suite)
    assert r.mk_16k_accuracy == 1.0
    assert r.vt_4k_accuracy == 1.0
    assert r.passed_gate is True


def test_suite_gate_fails_when_mk_below(monkeypatch):
    import gardener.bench.quality.ruler as mod
    suite = [
        {"context": "x", "question": "q",
         "expected": ["k1", "k2", "k3", "k4", "k5"],   # 5 keys
         "task_type": "multi_key_niah",
         "params": {"target_tokens": 16384, "num_keys": 5, "seed": 0}},
        {"context": "x", "question": "q",
         "expected": ["v"],
         "task_type": "variable_tracking",
         "params": {"target_tokens": 4096, "chain_length": 4, "seed": 0}},
    ]
    # mk gets 3/5 = 0.6 < 0.80; vt gets 1.0
    responses = iter(["k1 k2 k3", "v"])
    monkeypatch.setattr(mod, "generate", lambda *a, **k: next(responses))
    monkeypatch.setattr(mod, "_GENERATORS", {
        "multi_key_niah": lambda **kw: suite[0],
        "variable_tracking": lambda **kw: suite[1],
    })

    class _Tok:
        def apply_chat_template(self, m, **kw): return m[-1]["content"]
    class _M: pass

    spec_suite = [
        {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 5, "seed": 0},
        {"generator": "variable_tracking", "target_tokens": 4096, "chain_length": 4, "seed": 0},
    ]
    r = run_ruler_suite(_M(), _Tok(), suite=spec_suite)
    assert r.mk_16k_accuracy == pytest.approx(0.6)
    assert r.passed_gate is False


# --- model integration (small subset) ----------------------------------------

@pytest.mark.model
def test_runs_one_real_task_on_test_model(loaded_model):
    """Run a single small multi_key_niah task on the test model. Pass rate
    may be low on 0.5B — we just confirm the runner executes end-to-end."""
    model, tok = loaded_model
    task = generate_multi_key_niah(tok, target_tokens=1024, num_keys=2, seed=999)
    r = run_ruler_task(model, tok, task, max_tokens=64)
    assert r.task_type == "multi_key_niah"
    assert 0.0 <= r.accuracy <= 1.0
    assert isinstance(r.passed, bool)
