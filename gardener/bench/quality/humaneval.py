"""HumanEval-Lite — 20 curated HumanEval problems with exec+assert scoring.

Reference map (NOT imported): omlx/bench/hypercar_bench.py:1422-1620.
The HUMANEVAL_LITE list is a verbatim port of the upstream 20 problems.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .common import per_item_cleanup


HUMANEVAL_LITE: list[dict] = [
    {
        "task_id": "HE/0",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """Check if any two numbers in the list are closer than threshold."""\n',
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HE/1",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list[str]:\n    """Separate groups of balanced parentheses into individual strings."""\n',
        "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HE/2",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Return the decimal part of a positive float."""\n',
        "test": "assert abs(truncate_number(3.5) - 0.5) < 1e-6",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HE/3",
        "prompt": 'def below_zero(operations: list[int]) -> bool:\n    """Check if a bank account starting at 0 goes below zero after operations."""\n',
        "test": "assert below_zero([1, 2, -3, 1, 2, -4, 5, 6, -1, 2, -3, 5, -22]) == True\nassert below_zero([1, 2, 3]) == False",
        "entry_point": "below_zero",
    },
    {
        "task_id": "HE/4",
        "prompt": 'def mean_absolute_deviation(numbers: list[float]) -> float:\n    """Compute mean absolute deviation around the mean."""\n',
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
        "entry_point": "mean_absolute_deviation",
    },
    {
        "task_id": "HE/5",
        "prompt": 'def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n    """Insert delimeter between every two consecutive elements."""\n',
        "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
        "entry_point": "intersperse",
    },
    {
        "task_id": "HE/6",
        "prompt": 'def parse_nested_parens(paren_string: str) -> list[int]:\n    """Return the max nesting depth for each group of parentheses."""\n',
        "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]",
        "entry_point": "parse_nested_parens",
    },
    {
        "task_id": "HE/7",
        "prompt": 'def filter_by_substring(strings: list[str], substring: str) -> list[str]:\n    """Filter strings that contain the given substring."""\n',
        "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']",
        "entry_point": "filter_by_substring",
    },
    {
        "task_id": "HE/8",
        "prompt": 'def sum_product(numbers: list[int]) -> tuple[int, int]:\n    """Return a tuple of (sum, product) of all numbers in the list."""\n',
        "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)",
        "entry_point": "sum_product",
    },
    {
        "task_id": "HE/9",
        "prompt": 'def rolling_max(numbers: list[int]) -> list[int]:\n    """Return the running maximum at each position."""\n',
        "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]",
        "entry_point": "rolling_max",
    },
    {
        "task_id": "HE/10",
        "prompt": 'def is_palindrome(string: str) -> bool:\n    """Check if a string is a palindrome."""\n',
        "test": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('abc') == False",
        "entry_point": "is_palindrome",
    },
    {
        "task_id": "HE/11",
        "prompt": 'def string_xor(a: str, b: str) -> str:\n    """Perform XOR on two binary strings."""\n',
        "test": "assert string_xor('010', '110') == '100'",
        "entry_point": "string_xor",
    },
    {
        "task_id": "HE/12",
        "prompt": 'def longest(strings: list[str]) -> str | None:\n    """Return the longest string, or None if empty."""\n',
        "test": "assert longest([]) is None\nassert longest(['a', 'bb', 'ccc']) == 'ccc'",
        "entry_point": "longest",
    },
    {
        "task_id": "HE/13",
        "prompt": 'def greatest_common_divisor(a: int, b: int) -> int:\n    """Compute the GCD of two integers."""\n',
        "test": "assert greatest_common_divisor(3, 5) == 1\nassert greatest_common_divisor(25, 15) == 5",
        "entry_point": "greatest_common_divisor",
    },
    {
        "task_id": "HE/14",
        "prompt": 'def all_prefixes(string: str) -> list[str]:\n    """Return all prefixes from shortest to longest."""\n',
        "test": "assert all_prefixes('abc') == ['a', 'ab', 'abc']",
        "entry_point": "all_prefixes",
    },
    {
        "task_id": "HE/15",
        "prompt": 'def string_sequence(n: int) -> str:\n    """Return a space-separated string of numbers from 0 to n."""\n',
        "test": "assert string_sequence(0) == '0'\nassert string_sequence(5) == '0 1 2 3 4 5'",
        "entry_point": "string_sequence",
    },
    {
        "task_id": "HE/16",
        "prompt": 'def count_distinct_characters(string: str) -> int:\n    """Count distinct characters (case insensitive)."""\n',
        "test": "assert count_distinct_characters('xyzXYZ') == 3\nassert count_distinct_characters('Jerry') == 4",
        "entry_point": "count_distinct_characters",
    },
    {
        "task_id": "HE/17",
        "prompt": 'def parse_music(music_string: str) -> list[int]:\n    """Parse music notation: o=4, o|=2, .|=1 beats."""\n',
        "test": "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]",
        "entry_point": "parse_music",
    },
    {
        "task_id": "HE/18",
        "prompt": 'def how_many_times(string: str, substring: str) -> int:\n    """Count how many times substring occurs in string (overlapping)."""\n',
        "test": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3",
        "entry_point": "how_many_times",
    },
    {
        "task_id": "HE/19",
        "prompt": 'def sort_numbers(numbers: str) -> str:\n    """Sort space-separated number words (zero through nine)."""\n',
        "test": "assert sort_numbers('three one five') == 'one three five'",
        "entry_point": "sort_numbers",
    },
]


@dataclass
class HumanEvalResult:
    n_problems: int
    n_passed: int
    pass_rate: float
    gate_min: float = 0.35
    passed_gate: bool = False
    elapsed_s: float = 0.0
    per_problem: list[dict] = field(default_factory=list)


def run_humaneval_lite(
    model: Any,
    tokenizer: Any,
    *,
    n_problems: int | None = None,
    max_tokens: int = 256,
    gate_min: float = 0.35,
    verbose: bool = False,
) -> HumanEvalResult:
    """Run the HumanEval-Lite gate.

    For each problem: encode the prompt (function signature + docstring),
    greedy-generate up to max_tokens, split the completion at the first
    blank line or new def, exec the function + the test assertions, count
    passes. Per-problem cleanup between items (Task 259 pattern).
    """
    t0 = time.perf_counter()
    problems = HUMANEVAL_LITE if n_problems is None else HUMANEVAL_LITE[:n_problems]
    sampler = make_sampler(temp=0.0)
    per_problem: list[dict] = []
    n_passed = 0

    for prob in problems:
        try:
            text = generate(
                model, tokenizer, prompt=prob["prompt"],
                max_tokens=max_tokens, sampler=sampler, verbose=False,
            )
            # Isolate function body: first \n\n or \ndef ends the function.
            completion = text.split("\n\n")[0].split("\ndef ")[0]
            full_code = prob["prompt"] + completion
            exec_globals: dict = {}
            exec(full_code, exec_globals)        # nosec — test exec required
            exec(prob["test"], exec_globals)
            passed = True
            error = None
        except BaseException as e:
            passed = False
            error = f"{type(e).__name__}: {e}"
        if passed:
            n_passed += 1
        per_problem.append({
            "task_id": prob["task_id"], "passed": passed, "error": error,
        })
        if verbose:
            print(f"  {prob['task_id']}: {'PASS' if passed else 'FAIL'}"
                  + (f"  ({error})" if error else ""))
        per_item_cleanup()

    pass_rate = n_passed / len(problems) if problems else 0.0
    return HumanEvalResult(
        n_problems=len(problems), n_passed=n_passed, pass_rate=pass_rate,
        gate_min=gate_min, passed_gate=(pass_rate >= gate_min),
        elapsed_s=time.perf_counter() - t0, per_problem=per_problem,
    )
