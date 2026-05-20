"""LiveCodeBench quality gate (HX12.4).

Port of hypercar's Phase 3d LiveCodeBench runner.

SECURITY NOTE: This module executes model-generated code on the local machine.
Mitigations: subprocess with wall-time timeout, RLIMIT_AS memory cap, temp file
cleanup. Users must accept this before running the gate.

Reference map:
  omlx/eval/livecodebench.py       — _execute_code, data loading
  omlx/eval/base.py:125-168        — _extract_last_code_block (LAST-match)
  omlx/bench/hypercar_bench.py:1629-1865 — phase3d_livecodebench runner
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("gardener.bench.livecodebench")

# ---------------------------------------------------------------------------
# Constants (match hypercar's Task 254/255/258/261 findings)
# ---------------------------------------------------------------------------

LCB_RETRIES: int = 2          # greedy + 2 retries (Tasks 258/261)
LCB_RETRY_TEMP: float = 0.7   # temp=0.7 beats temp=1.0 for same gain (Task 261)
LCB_TIMEOUT_S: float = 30.0   # wall-time cap per execution
LCB_MEMORY_MB: int = 256      # RLIMIT_AS cap per subprocess

_DATA_PATH = Path(__file__).parent / "data" / "livecodebench.jsonl"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LCBProblem:
    question_id: str
    difficulty: str
    description: str          # natural-language problem text
    inputs: list[str]         # stdin per test case
    outputs: list[str]        # expected stdout per test case
    starter_code: str = ""


@dataclass
class LCBProblemResult:
    question_id: str
    difficulty: str
    passed: bool
    attempts: int             # 1 (greedy worked) … up to 1 + LCB_RETRIES
    error: str | None
    extracted_code: str


@dataclass
class LCBResult:
    n_problems: int
    n_passed: int
    pass_rate: float
    gate_min: float = 0.30
    passed_gate: bool = False
    elapsed_s: float = 0.0
    per_problem: list[LCBProblemResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load newline-delimited JSON file."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def _deterministic_sample(items: list, n: int, seed: int = 42) -> list:
    """Deterministic reservoir sample matching hypercar's deterministic_sample."""
    import random
    rng = random.Random(seed)
    if n >= len(items):
        sampled = list(items)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(items, n)


def load_livecodebench(
    n: int = 10,
    seed: int = 42,
    data_path: str | None = None,
) -> list[LCBProblem]:
    """Load N problems deterministically from the JSONL dataset.

    The dataset ships with the package under data/livecodebench.jsonl.
    """
    path = Path(data_path) if data_path else _DATA_PATH
    raw = _load_jsonl(path)

    problems: list[LCBProblem] = []
    for i, item in enumerate(raw):
        tc_raw = item.get("public_test_cases", "[]")
        if isinstance(tc_raw, str):
            try:
                tc = json.loads(tc_raw)
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            tc = tc_raw

        if not isinstance(tc, list) or not tc:
            continue

        inputs = [t.get("input", "") for t in tc]
        outputs = [t.get("output", "") for t in tc]
        if not inputs or not outputs:
            continue

        problems.append(LCBProblem(
            question_id=item.get("question_id", str(i)),
            difficulty=item.get("difficulty", "unknown"),
            description=item.get("question_content", ""),
            inputs=inputs,
            outputs=outputs,
            starter_code=item.get("starter_code", ""),
        ))

    return _deterministic_sample(problems, n, seed=seed)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(prob: LCBProblem) -> str:
    """CoT prompt template (exact match of hypercar_bench.py:1768-1777).

    Matches omlx/eval/livecodebench.py::format_prompt too — both were
    unified in Task 254 to prevent silent divergence.
    """
    return (
        "Solve the following programming problem in Python. "
        "Read input from stdin and print the output to stdout.\n\n"
        f"Problem:\n{prob.description}\n\n"
        "Think step-by-step: identify the approach, consider edge "
        "cases and complexity, then write the solution. End your "
        "response with the complete, runnable solution in a single "
        "```python code block.\n\n"
        "Solution:"
    )


# ---------------------------------------------------------------------------
# Code extraction — LAST-match semantics
# ---------------------------------------------------------------------------


def extract_last_code_block(response: str) -> str:
    """Extract the LAST code block from model response (not first-match).

    Three-stage fallback (reference: omlx/eval/base.py:125-168):
      1. Last closed ```python ... ``` block  (most common / CoT end)
      2. Last closed ``` ... ``` block        (generic fence)
      3. Unclosed ```python ... to end         (truncation case)
      4. Empty string when no marker is present.

    LAST-match is correct for CoT-style responses where the model emits a
    draft block then a corrected final block. First-match would return the
    wrong (draft) code.
    """
    response = response.strip()

    # 1. All closed python blocks — use LAST
    blocks = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # 2. Generic closed blocks — use LAST
    blocks = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # 3. Unclosed python block (model was truncated mid-code)
    unclosed = re.search(r"```python\s*\n(.*)", response, re.DOTALL)
    if unclosed:
        return unclosed.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Subprocess execution with resource limits
# ---------------------------------------------------------------------------


def _set_resource_limits(memory_bytes: int, timeout_s: float) -> None:
    """Set RLIMIT_AS + RLIMIT_CPU in child process. Called via preexec_fn."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, OSError):
        pass
    cpu_s = int(timeout_s) + 5
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):
        pass


def execute_code(
    code: str,
    stdin_input: str,
    timeout_s: float = LCB_TIMEOUT_S,
    memory_mb: int = LCB_MEMORY_MB,
) -> tuple[str, bool, str | None]:
    """Run code in a subprocess with stdin=stdin_input.

    Returns (stdout_text, success, error_or_None).

    Resource limits applied:
      - RLIMIT_AS  : memory_mb * 1024 * 1024 bytes of virtual address space
      - RLIMIT_CPU : timeout_s + 5 s CPU time (wall-time timeout enforced
                     by subprocess.run's timeout= separately)
      - Wall-time  : timeout_s (subprocess.run timeout)

    Reference: omlx/eval/livecodebench.py:97-147.
    """
    memory_bytes = memory_mb * 1024 * 1024

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as fh:
        fh.write(code)
        tmp_path = fh.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            preexec_fn=lambda: _set_resource_limits(memory_bytes, timeout_s),
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/usr/local/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": "en_US.UTF-8",
            },
        )
        if result.returncode == 0:
            return result.stdout, True, None
        else:
            return result.stdout, False, result.stderr[:500] or f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return "", False, f"timeout after {timeout_s}s killed"
    except Exception as exc:
        return "", False, str(exc)[:500]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


# Module-level name that tests can monkeypatch
def generate(model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> str:  # noqa: ANN401
    """Thin wrapper around mlx_lm.generate (monkeypatched in tests)."""
    from mlx_lm import generate as _gen
    return _gen(model, tokenizer, prompt=prompt, **kwargs)


def _passes_test_cases(
    code: str,
    prob: LCBProblem,
    n: int = 3,
) -> tuple[bool, str | None]:
    """Execute code against the first n test cases of prob.

    Returns (all_passed, last_error_or_None).
    """
    if not code.strip():
        return False, "empty code"

    inputs = prob.inputs[:n]
    outputs = prob.outputs[:n]
    last_error: str | None = None

    for inp, expected_out in zip(inputs, outputs):
        stdin_input = inp if isinstance(inp, str) else str(inp)
        expected = expected_out.strip() if isinstance(expected_out, str) else str(expected_out).strip()
        stdout, success, err = execute_code(code, stdin_input)
        if not success:
            last_error = err
            return False, last_error
        if stdout.strip() != expected:
            last_error = f"wrong output: got {stdout.strip()!r}, expected {expected!r}"
            return False, last_error

    return True, None


def run_livecodebench(
    model: Any,
    tokenizer: Any,
    *,
    n_problems: int = 10,
    max_tokens: int = 1024,
    gate_min: float = 0.30,
    retries: int = LCB_RETRIES,
    retry_temp: float = LCB_RETRY_TEMP,
    seed: int = 42,
    enable_thinking: bool = False,
    problems: list[LCBProblem] | None = None,
    verbose: bool = False,
) -> LCBResult:
    """Run LCB gate.

    For each problem:
      Attempt 0 (greedy, temp=0): generate → extract → execute against first
        3 test cases. If pass: record 1 attempt.
      If fails, attempt 1..retries (temp=retry_temp): generate → extract →
        execute. First retry that passes wins.
      If all attempts fail: record failure with last error.

    Per-problem gc.collect()+mx.clear_cache() (Task 259).

    Parameters
    ----------
    problems:
        Inject a list of LCBProblem objects directly (used in tests to avoid
        hitting the dataset file). If None, loads n_problems from the bundled
        JSONL.
    """
    t0 = time.perf_counter()

    if problems is None:
        problems = load_livecodebench(n=n_problems, seed=seed)

    per_problem: list[LCBProblemResult] = []
    n_passed = 0

    for prob in problems:
        prompt_text = format_prompt(prob)

        # Apply chat template (disable thinking for LCB — consistent with
        # Task 254's finding that thinking burns tokens before code emerges)
        messages = [{"role": "user", "content": prompt_text}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = prompt_text + "\n"
        except Exception:
            prompt = prompt_text + "\n"

        # --- Attempt 0: greedy (temp=0) ---
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.0,
        )
        code = extract_last_code_block(response)
        passed, last_error = _passes_test_cases(code, prob)
        attempts = 1

        # --- Retries with temp>0 if greedy failed ---
        if not passed:
            for retry_i in range(retries):
                r_seed = retry_i + 1
                r_response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=retry_temp,
                    seed=r_seed,
                )
                r_code = extract_last_code_block(r_response)
                r_passed, r_error = _passes_test_cases(r_code, prob)
                attempts += 1
                if r_passed:
                    code = r_code
                    passed = True
                    last_error = None
                    break
                last_error = r_error

        status = "PASS" if passed else "FAIL"
        marker = "" if attempts == 1 else f" (retry {attempts - 1})"
        if verbose:
            logger.info(f"  {status}{marker}: [{prob.difficulty}] {prob.question_id[:60]}")

        if passed:
            n_passed += 1

        per_problem.append(LCBProblemResult(
            question_id=prob.question_id,
            difficulty=prob.difficulty,
            passed=passed,
            attempts=attempts,
            error=last_error if not passed else None,
            extracted_code=code,
        ))

        # Per-problem cleanup (Task 259 — prevents 100× slowdown at high N)
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass

    elapsed = time.perf_counter() - t0
    n_total = len(per_problem)
    pass_rate = n_passed / max(n_total, 1)
    passed_gate = pass_rate >= gate_min

    logger.info(
        f"LiveCodeBench: {n_passed}/{n_total} ({pass_rate * 100:.0f}%) "
        f"— gate {'PASS' if passed_gate else 'FAIL'}"
    )

    return LCBResult(
        n_problems=n_total,
        n_passed=n_passed,
        pass_rate=pass_rate,
        gate_min=gate_min,
        passed_gate=passed_gate,
        elapsed_s=elapsed,
        per_problem=per_problem,
    )
