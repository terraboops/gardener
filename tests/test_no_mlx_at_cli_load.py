"""Regression guard: gardener.cli.main must import without mlx installed.

CI runs on Linux runners where mlx (Mac-only) isn't installed. The CLI
must therefore not pull in mlx at module-load time for any of its
no-MLX subcommands (wizard, list, observe, calibrate-wizard, etc.) —
MLX-only paths (query, cache warm/use, swarm execution) must lazy-import
mlx_lm inside the function body.

This test simulates the missing-mlx environment by blocking the imports
before triggering CLI load. If anyone adds a top-level `from mlx... import`
to a command module on the CLI path, this test fails loudly.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap


def test_cli_main_imports_without_mlx_in_a_fresh_subprocess():
    """Subprocess with sys.modules-blocked mlx — fresh interpreter, no test pollution."""
    code = textwrap.dedent(
        """
        import sys

        # Block mlx imports completely. Any attempted import raises ImportError.
        class _Blocked:
            def __getattr__(self, name):
                raise ImportError(f"mlx is blocked in CI: requested {name!r}")

        for blocked in ("mlx", "mlx.core", "mlx.nn", "mlx_lm",
                        "mlx_lm.sample_utils", "mlx_lm.models",
                        "mlx_lm.models.cache"):
            sys.modules[blocked] = _Blocked()

        # If this import triggers any top-level mlx access, the import
        # itself raises ImportError and the subprocess exits non-zero.
        import gardener.cli.main as m
        # Build the parser too — that's what `gardener --help` does.
        p = m.build_parser()
        # Sanity: the CLI knows about the no-MLX subcommands.
        subcommands = set()
        for a in p._subparsers._actions:
            choices = getattr(a, "choices", None)
            if isinstance(choices, dict):
                subcommands.update(choices.keys())
        for required in ("wizard", "list", "observe", "calibrate-wizard"):
            assert required in subcommands, f"missing subcommand: {required}"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"gardener.cli.main pulled in mlx at module load — see stderr:\n"
        f"{result.stderr}\n"
        f"stdout: {result.stdout}"
    )
    assert "OK" in result.stdout


def test_wizard_subpackage_imports_without_mlx():
    """Same drill for the wizard subpackage (used by calibrate-wizard etc.)."""
    code = textwrap.dedent(
        """
        import sys

        class _Blocked:
            def __getattr__(self, name):
                raise ImportError(f"mlx is blocked: {name!r}")

        for blocked in ("mlx", "mlx.core", "mlx.nn", "mlx_lm",
                        "mlx_lm.sample_utils", "mlx_lm.models",
                        "mlx_lm.models.cache"):
            sys.modules[blocked] = _Blocked()

        # The MLX-backed drafter MAY be imported but it must lazy-load
        # mlx_lm inside __init__, not at module load.
        import gardener.wizard.drafter_mlx
        import gardener.wizard.config
        import gardener.wizard.parser
        import gardener.wizard.drafter
        import gardener.wizard.adversarial
        import gardener.wizard.thresholds
        import gardener.wizard.germination
        import gardener.wizard.cli_flow
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"wizard subpackage triggered mlx at module load:\n{result.stderr}"
    )
    assert "OK" in result.stdout
