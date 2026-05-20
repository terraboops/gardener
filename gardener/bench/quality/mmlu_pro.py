"""MMLU-Pro — 25 to 100 multi-choice questions across 14 categories.
Step-by-step CoT prompt; 3-stage regex extraction; thinking=False required
for thinking models or the answer never emerges within the budget.

Reference map (NOT imported):
- omlx/eval/mmlu_pro/tasks.py:38-138
- omlx/bench/hypercar_bench.py:1291-1415

External dep: `datasets` (HuggingFace) — first call downloads
`TIGER-Lab/MMLU-Pro` (~50 MB).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .common import format_chat_prompt, per_item_cleanup

# Defaults from hypercar (omlx/eval/mmlu_pro/tasks.py + hypercar_bench.py)
DEFAULT_N_QUESTIONS = 100
DEFAULT_MAX_TOKENS = 1536          # bumped from 512 (omlx: silent 64→24% regression)
DEFAULT_GATE_MIN_ACCURACY = 0.35   # hypercar's gate
DEFAULT_CATEGORIES: Optional[list[str]] = None   # None = all 14 categories


def load_mmlu_pro(
    categories: Optional[list[str]] = None,
    n: int = DEFAULT_N_QUESTIONS,
    seed: int = 42,
) -> list[dict]:
    """Load N questions from TIGER-Lab/MMLU-Pro on HuggingFace.

    Returns: list of {question, options (list[str]), answer (str A-J),
                       category (str)}.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The `datasets` package is required for MMLU-Pro. "
            "Install it via `pip install datasets` or add to your env."
        ) from e

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = list(ds)
    if categories:
        rows = [r for r in rows if r.get("category") in categories]

    # Deterministic sample.
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    # Normalize shape: options as list[str].
    norm: list[dict] = []
    for r in rows:
        opts = r.get("options") or []
        norm.append({
            "question": r["question"],
            "options": list(opts),
            "answer": r["answer"],         # "A" … "J"
            "category": r.get("category", "unknown"),
        })
    return norm


def format_prompt(q: dict) -> str:
    """The exact prompt template from omlx/eval/mmlu_pro/tasks.py:97-114."""
    options_str = "\n".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(q["options"])
    )
    return (
        f"The following is a multiple choice question about "
        f"{q['category']}. "
        f"Think step by step and then output the answer in the format of "
        f'"The answer is (X)" at the end.\n\n'
        f"Question: {q['question']}\n"
        f"Options:\n{options_str}\n\n"
        f"Let me think step by step."
    )


def extract_answer(response: str) -> Optional[str]:
    """3-stage extraction (omlx/eval/mmlu_pro/tasks.py:117-138).

    Stage 1: 'The answer is (X)' or 'the answer is X'.
    Stage 2: 'Answer: X' or 'answer: X'.
    Stage 3: last standalone letter A-J in the response.
    """
    m = re.search(r"[Tt]he answer is \(?([A-J])\)?", response)
    if m:
        return m.group(1)
    m = re.search(r"[Aa]nswer:\s*\(?([A-J])\)?", response)
    if m:
        return m.group(1)
    m = re.findall(r"\b([A-J])\b", response)
    if m:
        return m[-1]
    return None


@dataclass
class MMLUProResult:
    n_questions: int
    n_correct: int
    accuracy: float
    gate_min: float = DEFAULT_GATE_MIN_ACCURACY
    passed_gate: bool = False
    elapsed_s: float = 0.0
    per_question: list[dict] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)


def run_mmlu_pro(
    model: Any,
    tokenizer: Any,
    *,
    n_questions: int = DEFAULT_N_QUESTIONS,
    categories: Optional[list[str]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gate_min: float = DEFAULT_GATE_MIN_ACCURACY,
    seed: int = 42,
    enable_thinking: bool = False,
    verbose: bool = False,
    questions: Optional[list[dict]] = None,  # injection point for tests
) -> MMLUProResult:
    """Run the MMLU-Pro gate.

    For each question: format CoT prompt → generate up to max_tokens (greedy)
    → 3-stage extract → compare to gold answer → cleanup. Returns
    per-question + per-category accuracy.
    """
    t0 = time.perf_counter()
    qs = questions if questions is not None else \
         load_mmlu_pro(categories=categories, n=n_questions, seed=seed)

    sampler = make_sampler(temp=0.0)
    per_question: list[dict] = []
    n_correct = 0
    cat_counts: dict[str, dict] = {}

    for q in qs:
        prompt = format_chat_prompt(
            tokenizer, format_prompt(q), enable_thinking=enable_thinking,
        )
        try:
            text = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, sampler=sampler, verbose=False,
            )
            extracted = extract_answer(text)
            correct = (extracted == q["answer"])
        except BaseException as e:
            extracted = None
            correct = False
            text = f"<error: {e}>"
        if correct:
            n_correct += 1
        per_question.append({
            "category": q["category"], "expected": q["answer"],
            "extracted": extracted, "correct": correct,
        })
        cat = q["category"]
        c = cat_counts.setdefault(cat, {"n": 0, "correct": 0})
        c["n"] += 1
        if correct:
            c["correct"] += 1
        if verbose:
            tag = "OK" if correct else "X "
            print(f"  [{tag}] {cat:>20s}  exp={q['answer']} got={extracted}")
        per_item_cleanup()

    by_category = {
        cat: {**v, "accuracy": v["correct"] / v["n"] if v["n"] else 0.0}
        for cat, v in cat_counts.items()
    }
    accuracy = n_correct / len(qs) if qs else 0.0
    return MMLUProResult(
        n_questions=len(qs), n_correct=n_correct, accuracy=accuracy,
        gate_min=gate_min, passed_gate=(accuracy >= gate_min),
        elapsed_s=time.perf_counter() - t0,
        per_question=per_question, by_category=by_category,
    )
