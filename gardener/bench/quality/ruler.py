"""RULER synthetic long-context quality gate (HX12.3).

Three synthetic task types ported from hypercar's Phase 3b:
  - multi_key_niah: needle-in-a-haystack with N keys (AND scoring)
  - variable_tracking: chain of variable assignments (OR scoring)
  - frequent_word: count marker frequency and identify the winner (OR scoring)

Two gates:
  1. multi_key_niah @ 16K accuracy >= 0.80
  2. variable_tracking @ 4K accuracy >= 0.70

Reference map (NOT imported):
  omlx/eval/ruler/tasks.py:145-409
  omlx/bench/hypercar_bench.py:959-1284

Based on RULER (arXiv:2404.06654) with hypercar adaptations.
"""
from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .common import format_chat_prompt, per_item_cleanup


# ---------------------------------------------------------------------------
# Filler text — same code-block pool as hypercar's omlx/eval/ruler/tasks.py
# ---------------------------------------------------------------------------

_CODE_BLOCKS = [
    '''def binary_search(arr: list, target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    '''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._store: dict = {}
        self._order: list = []

    def get(self, key: str):
        if key in self._store:
            self._order.remove(key)
            self._order.append(key)
            return self._store[key]
        return None

    def put(self, key: str, value):
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self.capacity:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = value
        self._order.append(key)
''',
    '''import hashlib
from pathlib import Path

def checksum(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
''',
    '''# Graph utilities
def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    visited = set()
    queue = [start]
    order = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            order.append(node)
            queue.extend(graph.get(node, []))
    return order
''',
    '''from typing import TypeVar, Generic
T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)
''',
    '''def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result
''',
]


def _build_filler(tokenizer: Any, target_tokens: int, seed: int = 42) -> str:
    """Build filler text from shuffled code blocks to approximately fill
    target_tokens. Uses the tokenizer to measure length."""
    rng = random.Random(seed)
    filler = "\n\n".join(_CODE_BLOCKS)
    filler_tokens = len(tokenizer.encode(filler))

    reps = max(1, (target_tokens // filler_tokens) + 1)
    blocks = _CODE_BLOCKS * reps
    rng.shuffle(blocks)
    text = "\n\n".join(blocks)

    tokens = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(tokens)


def _random_code(length: int = 8, seed: int | None = None) -> str:
    """Generate a random alphanumeric code like 'XKCD-4829'."""
    rng = random.Random(seed)
    prefix = "".join(rng.choices(string.ascii_uppercase, k=4))
    suffix = "".join(rng.choices(string.digits, k=4))
    return f"{prefix}-{suffix}"


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def generate_multi_key_niah(
    tokenizer: Any,
    target_tokens: int,
    num_keys: int = 3,
    seed: int = 42,
) -> dict:
    """Generate a multi-key needle-in-a-haystack task.

    Hides `num_keys` key-value pairs at random depths in filler text.
    The model must retrieve all values.

    Returns dict with context, question, expected answers (AND scoring).
    """
    rng = random.Random(seed)

    # Generate unique key-value pairs
    keys_and_values = []
    for i in range(num_keys):
        code = _random_code(seed=seed + i)
        keys_and_values.append((f"key_{i+1}", code))

    # Build filler, leaving room for needles + question overhead
    overhead_tokens = num_keys * 30 + 100
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)

    # Split filler into chunks and insert needles at random positions
    lines = filler.split("\n")
    chunk_size = max(1, len(lines) // (num_keys + 1))

    for i, (key, value) in enumerate(keys_and_values):
        needle = f"# IMPORTANT: The value for {key} is {value}"
        insert_pos = min(
            (i + 1) * chunk_size + rng.randint(0, max(1, chunk_size // 2)),
            len(lines) - 1,
        )
        lines.insert(insert_pos, needle)

    context = "\n".join(lines)

    # Trim to target tokens
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    key_list = ", ".join(k for k, _ in keys_and_values)
    question = (
        f"Based on the text above, what are the values for the following keys: "
        f"{key_list}? List each value on its own line in the format 'key: value'."
    )

    expected = [v for _, v in keys_and_values]

    return {
        "context": context,
        "question": question,
        "expected": expected,
        "task_type": "multi_key_niah",
        "params": {
            "target_tokens": target_tokens,
            "num_keys": num_keys,
            "seed": seed,
            "keys": keys_and_values,
        },
    }


def generate_variable_tracking(
    tokenizer: Any,
    target_tokens: int,
    chain_length: int = 4,
    seed: int = 42,
) -> dict:
    """Generate a variable tracking task.

    Creates a chain of variable assignments spread across the context:
      x1 = <value>
      x2 = x1
      x3 = x2
      ...
    The model must determine the final value of the last variable.

    Returns dict with expected (single-value list; OR scoring on case variants).
    """
    rng = random.Random(seed)

    value = _random_code(seed=seed)
    var_names = [f"var_{chr(ord('a') + i)}" for i in range(chain_length)]

    assignments = []
    assignments.append(f"# Assignment: {var_names[0]} = '{value}'")
    for i in range(1, chain_length):
        assignments.append(f"# Assignment: {var_names[i]} = {var_names[i-1]}")

    overhead_tokens = chain_length * 20 + 100
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)
    lines = filler.split("\n")

    chunk_size = max(1, len(lines) // (chain_length + 1))
    for i, assignment in enumerate(assignments):
        insert_pos = min(
            (i + 1) * chunk_size + rng.randint(0, max(1, chunk_size // 3)),
            len(lines) - 1,
        )
        lines.insert(insert_pos, assignment)

    context = "\n".join(lines)
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    question = (
        f"Based on the variable assignments in the text above, "
        f"trace the chain of assignments to find what {var_names[-1]} "
        f"ultimately resolves to. The first variable in the chain was "
        f"assigned a literal string value like 'XXXX-1234'. "
        f"What is that string value? Respond with ONLY the string value "
        f"(e.g. ABCD-5678), nothing else."
    )

    return {
        "context": context,
        "question": question,
        "expected": [value],
        "task_type": "variable_tracking",
        "params": {
            "target_tokens": target_tokens,
            "chain_length": chain_length,
            "seed": seed,
            "var_names": var_names,
            "value": value,
        },
    }


def generate_frequent_word(
    tokenizer: Any,
    target_tokens: int,
    num_target_words: int = 5,
    seed: int = 42,
) -> dict:
    """Generate a frequent-word aggregation task.

    Inserts specific marker words at known frequencies throughout the context.
    One word appears significantly more often than the others. The model must
    identify which word is most frequent.

    Returns dict with expected = [WINNER, winner, Winner] (case variants);
    OR scoring.
    """
    rng = random.Random(seed)

    marker_pool = [
        "ZEPHYR", "QUARTZ", "VORTEX", "NEBULA", "PRISM",
        "COBALT", "HELIX", "ZENITH", "AXIOM", "CIPHER",
    ]
    rng.shuffle(marker_pool)
    markers = marker_pool[:num_target_words]

    # Assign frequencies — one word is clearly the winner
    frequencies = []
    winner_freq = 10 + rng.randint(0, 5)
    frequencies.append(winner_freq)
    for i in range(1, num_target_words):
        frequencies.append(max(2, winner_freq // 2 - rng.randint(0, 2)))

    winner_word = markers[0]

    overhead_tokens = sum(frequencies) * 15 + 100
    filler = _build_filler(tokenizer, target_tokens - overhead_tokens, seed=seed)
    lines = filler.split("\n")

    insertions = []
    for marker, freq in zip(markers, frequencies):
        for _ in range(freq):
            insertions.append(f"# marker: {marker}")

    rng.shuffle(insertions)

    if lines:
        step = max(1, len(lines) // (len(insertions) + 1))
        for i, insertion in enumerate(insertions):
            pos = min((i + 1) * step, len(lines))
            lines.insert(pos + i, insertion)

    context = "\n".join(lines)
    tokens = tokenizer.encode(context)[:target_tokens]
    context = tokenizer.decode(tokens)

    marker_list = ", ".join(markers)
    question = (
        f"In the text above, the following marker words appear as comments: "
        f"{marker_list}. Which marker word appears MOST frequently? "
        f"Respond with ONLY the word, nothing else."
    )

    return {
        "context": context,
        "question": question,
        "expected": [winner_word, winner_word.lower(), winner_word.title()],
        "task_type": "frequent_word",
        "params": {
            "target_tokens": target_tokens,
            "num_target_words": num_target_words,
            "seed": seed,
            "marker_words": [m for m, _ in zip(markers, frequencies)],
            "markers": list(zip(markers, frequencies)),
            "winner": winner_word,
        },
    }


# ---------------------------------------------------------------------------
# Task suites — identical seeds/params to hypercar omlx/eval/ruler/tasks.py
# ---------------------------------------------------------------------------

# Quick suite: 3 task types × 2 context lengths = 6 tasks
RULER_QUICK_SUITE: list[dict] = [
    {"generator": "multi_key_niah",    "target_tokens": 4096,  "num_keys": 2,           "seed": 100},
    {"generator": "multi_key_niah",    "target_tokens": 16384, "num_keys": 3,           "seed": 101},
    {"generator": "variable_tracking", "target_tokens": 4096,  "chain_length": 4,       "seed": 304},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 4,       "seed": 305},
    {"generator": "frequent_word",     "target_tokens": 4096,  "num_target_words": 4,   "seed": 200},
    {"generator": "frequent_word",     "target_tokens": 16384, "num_target_words": 5,   "seed": 201},
]

# Full suite: 3 task types × multiple configs × 3 context lengths = 15 tasks
RULER_FULL_SUITE: list[dict] = [
    # Multi-key NIAH at 3 context lengths, varying key counts
    {"generator": "multi_key_niah", "target_tokens": 4096,  "num_keys": 2, "seed": 100},
    {"generator": "multi_key_niah", "target_tokens": 4096,  "num_keys": 4, "seed": 102},
    {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 3, "seed": 101},
    {"generator": "multi_key_niah", "target_tokens": 16384, "num_keys": 5, "seed": 103},
    {"generator": "multi_key_niah", "target_tokens": 65536, "num_keys": 3, "seed": 104},
    # Variable tracking at 3 context lengths, chain lengths 3-8
    {"generator": "variable_tracking", "target_tokens": 4096,  "chain_length": 3, "seed": 300},
    {"generator": "variable_tracking", "target_tokens": 4096,  "chain_length": 4, "seed": 304},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 4, "seed": 301},
    {"generator": "variable_tracking", "target_tokens": 16384, "chain_length": 8, "seed": 306},
    {"generator": "variable_tracking", "target_tokens": 65536, "chain_length": 4, "seed": 303},
    {"generator": "variable_tracking", "target_tokens": 65536, "chain_length": 8, "seed": 307},
    # Frequent-word aggregation at 3 context lengths
    {"generator": "frequent_word", "target_tokens": 4096,  "num_target_words": 4, "seed": 200},
    {"generator": "frequent_word", "target_tokens": 16384, "num_target_words": 5, "seed": 201},
    {"generator": "frequent_word", "target_tokens": 16384, "num_target_words": 7, "seed": 202},
    {"generator": "frequent_word", "target_tokens": 65536, "num_target_words": 5, "seed": 203},
]

# Map spec 'generator' key -> generator function
_GENERATORS = {
    "multi_key_niah": generate_multi_key_niah,
    "variable_tracking": generate_variable_tracking,
    "frequent_word": generate_frequent_word,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RulerTaskResult:
    task_type: str            # "multi_key_niah" | "variable_tracking" | "frequent_word"
    target_tokens: int        # 4096 | 16384 | 65536
    seed: int
    accuracy: float           # 0.0–1.0
    passed: bool              # accuracy >= 0.5
    elapsed_s: float
    expected: list[str]
    response: str


@dataclass
class RulerSuiteResult:
    tasks: list[RulerTaskResult]
    mk_16k_accuracy: float | None        # multi_key_niah @ 16K — gates >= 0.80
    vt_4k_accuracy: float | None         # variable_tracking @ 4K — gates >= 0.70
    gate_mk_min: float = 0.80
    gate_vt_min: float = 0.70
    passed_gate: bool = False
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_ruler_task(
    model: Any,
    tokenizer: Any,
    task: dict,
    *,
    max_tokens: int = 256,
    enable_thinking: bool = False,
) -> RulerTaskResult:
    """Build chat prompt from task['context'] + task['question'], generate
    (greedy), then score:
      - task_type == 'multi_key_niah': AND logic — fraction of expected
        values that appear as substrings in the response.
      - task_type in {'variable_tracking', 'frequent_word'}: OR logic —
        accuracy is 1.0 if any expected variant substring-matches, else 0.
    """
    t0 = time.perf_counter()

    task_type = task["task_type"]
    expected = task["expected"]
    params = task.get("params", {})
    target_tokens = params.get("target_tokens", 0)
    seed = params.get("seed", 0)

    user_message = f"{task['context']}\n\n{task['question']}"
    prompt = format_chat_prompt(tokenizer, user_message,
                                enable_thinking=enable_thinking)

    sampler = make_sampler(temp=0.0)
    response = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()

    # Scoring
    if task_type in ("frequent_word", "variable_tracking"):
        # OR logic: any case variant matching = 1.0
        any_found = any(exp in response for exp in expected)
        accuracy = 1.0 if any_found else 0.0
    else:
        # AND logic (multi_key_niah): fraction of expected values found
        found = sum(1 for exp in expected if exp in response)
        accuracy = found / len(expected) if expected else 0.0

    passed = accuracy >= 0.5

    per_item_cleanup()

    return RulerTaskResult(
        task_type=task_type,
        target_tokens=target_tokens,
        seed=seed,
        accuracy=accuracy,
        passed=passed,
        elapsed_s=time.perf_counter() - t0,
        expected=expected,
        response=response[:200],
    )


# ---------------------------------------------------------------------------
# Suite runner + gate aggregator
# ---------------------------------------------------------------------------

def run_ruler_suite(
    model: Any,
    tokenizer: Any,
    *,
    suite: list[dict] | None = None,
    gate_mk_min: float = 0.80,
    gate_vt_min: float = 0.70,
    max_tokens: int = 256,
    verbose: bool = False,
) -> RulerSuiteResult:
    """Run each task in the suite, aggregate per-task results, and check
    the two gate thresholds: multi_key_niah@16K >= gate_mk_min AND
    variable_tracking@4K >= gate_vt_min.

    The suite dicts use a 'generator' key and generator-specific kwargs.
    The tokenizer is injected automatically into each generator call.
    """
    if suite is None:
        suite = RULER_QUICK_SUITE

    t0 = time.perf_counter()
    task_results: list[RulerTaskResult] = []

    for i, spec in enumerate(suite):
        gen_name = spec["generator"]
        gen_fn = _GENERATORS[gen_name]

        # Build generator kwargs: all spec keys except 'generator'
        gen_kwargs = {k: v for k, v in spec.items() if k != "generator"}
        gen_kwargs["tokenizer"] = tokenizer

        task = gen_fn(**gen_kwargs)

        if verbose:
            ctx_k = spec["target_tokens"] // 1024
            print(f"  [{i+1}/{len(suite)}] {gen_name}@{ctx_k}K seed={spec.get('seed')}")

        result = run_ruler_task(model, tokenizer, task,
                                max_tokens=max_tokens, enable_thinking=False)
        task_results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status}: acc={result.accuracy:.0%} | {result.response[:60]!r}")

    # Gate 1: multi_key_niah @ 16K
    mk_16k = [r for r in task_results
               if r.task_type == "multi_key_niah" and r.target_tokens == 16384]
    mk_16k_accuracy: float | None
    if mk_16k:
        mk_16k_accuracy = sum(r.accuracy for r in mk_16k) / len(mk_16k)
    else:
        mk_16k_accuracy = None

    # Gate 2: variable_tracking @ 4K
    vt_4k = [r for r in task_results
              if r.task_type == "variable_tracking" and r.target_tokens == 4096]
    vt_4k_accuracy: float | None
    if vt_4k:
        vt_4k_accuracy = sum(r.accuracy for r in vt_4k) / len(vt_4k)
    else:
        vt_4k_accuracy = None

    # Both gates must pass (None means no tasks of that type ran — skip gate)
    mk_gate = (mk_16k_accuracy is None) or (mk_16k_accuracy >= gate_mk_min)
    vt_gate = (vt_4k_accuracy is None) or (vt_4k_accuracy >= gate_vt_min)
    passed_gate = mk_gate and vt_gate

    return RulerSuiteResult(
        tasks=task_results,
        mk_16k_accuracy=mk_16k_accuracy,
        vt_4k_accuracy=vt_4k_accuracy,
        gate_mk_min=gate_mk_min,
        gate_vt_min=gate_vt_min,
        passed_gate=passed_gate,
        elapsed_s=time.perf_counter() - t0,
    )
