"""Pytest config + shared fixtures.

MLX availability handling
=========================

The gardener test suite mixes pure-Python tests (parser, config, wizard
flow, journal, knowledge store, etc.) with MLX-backed tests (model load,
real drafter, kv-cache kernels). The pure-Python ones run in CI on Linux;
the MLX-backed ones run locally on Mac.

Three layers of MLX-related test skipping:

  1. `@pytest.mark.model`            — runtime skip via `-m "not model"`
  2. Inline `pytest.importorskip`    — module body skips if mlx missing
  3. `pytest_ignore_collect` (here)  — test FILES with top-level `import mlx`
                                       cannot be *collected* without mlx;
                                       this hook ignores them.

CI runs `pytest -m "not model"` after this hook has already removed
collection-blocking files. The combination keeps the suite runnable on
both Mac (full) and Linux (mock-only).
"""
from __future__ import annotations

import pathlib

import pytest


TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


# Detect MLX availability once at config load.
try:
    import mlx.core  # noqa: F401

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


def pytest_ignore_collect(collection_path: pathlib.Path) -> bool:
    """Skip collection of test files that require mlx at module load.

    Only fires when mlx is not installed (i.e. Linux CI). On Mac, every
    test file is collectable. We detect "requires mlx for collection" by
    searching for a top-level `import mlx` / `from mlx` line — quick,
    string-based, no AST parsing.
    """
    if _HAS_MLX:
        return False
    if not collection_path.is_file():
        return False
    if collection_path.suffix != ".py":
        return False
    # Only test files (not gardener source).
    if "tests" not in collection_path.parts:
        return False
    try:
        text = collection_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    # Top-of-file import lines (allow leading whitespace + future imports).
    # Includes direct mlx imports AND transitively-mlx gardener modules
    # (mlxsuper/sleep/subagents/harness/bench) that import mlx at their
    # own module load, AND examples that wrap MLX usage.
    needs_mlx_prefixes = (
        "import mlx",
        "from mlx ",
        "from mlx.",
        "from mlx_lm",
        "import gardener.mlxsuper",
        "from gardener.mlxsuper",
        "import gardener.sleep",
        "from gardener.sleep",
        "import gardener.subagents",
        "from gardener.subagents",
        "import gardener.harness",
        "from gardener.harness",
        "import gardener.bench",
        "from gardener.bench",
        "from examples.demo_simple_agent",
        "from examples.demo_full",
        "import examples.demo_simple_agent",
        "import examples.demo_full",
    )
    # Scan the WHOLE file, not just the top import block: function-body
    # imports of mlx (e.g. test_phase0_gate.py doing
    # `from gardener.mlxsuper import ...` inside the test body) succeed at
    # collection but explode at execution. Either signal — top-of-file
    # import or body import — means the test needs mlx to do anything
    # useful, so skipping at collection is the right policy.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(stripped.startswith(p) for p in needs_mlx_prefixes):
            return True
        # Module-level pytestmark = pytest.mark.model also means the whole
        # file's tests need MLX; safe to skip collection.
        if stripped.startswith("pytestmark") and "mark.model" in stripped:
            return True
    return False


@pytest.fixture(scope="session")
def loaded_model():
    from mlx_lm import load

    model, tokenizer = load(TEST_MODEL)
    return model, tokenizer
