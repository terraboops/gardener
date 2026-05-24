"""Top-level Gardener CLI entry point.

   gardener init NAME [--path PATH] [--model HF_ID] [--force]
   gardener list
   gardener query NAME-OR-PATH "QUESTION" [--max-tokens N] [--temperature T]
   gardener update NAME-OR-PATH "NOTE"
   gardener remove NAME
   gardener swarm PIPELINE [--agent NAME ...] [--inputs JSON] [--journal-root DIR]
                           [--max-concurrent N] [--priority P] [--timeout S]
   gardener observe NAME-OR-PATH [--topic TOPIC] [--follow] [-n N]
   gardener cache warm AGENT [--text TEXT | --file FILE | --use-prompt-md] [--name NAME]
   gardener cache list AGENT
   gardener cache rm AGENT CACHE_NAME
   gardener cache use AGENT CACHE_NAME "QUESTION"
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .commands.cache import (
    cmd_cache_list,
    cmd_cache_rm,
    cmd_cache_use,
    cmd_cache_warm,
)
from .commands.calibrate_wizard import cmd_calibrate_wizard
from .commands.germination_status import cmd_germination_status
from .commands.init import cmd_init
from .commands.list import cmd_list_agents
from .commands.observe import cmd_observe
from .commands.query import cmd_query
from .commands.remove import cmd_remove
from .commands.swarm import cmd_swarm
from .commands.update import cmd_update
from .commands.wizard import cmd_wizard


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gardener",
        description="Cultivate long-running persistent agents.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="Create a new agent directory.")
    s.add_argument("name")
    s.add_argument(
        "--path", default=None,
        help="Directory to scaffold (default: ./<name>).",
    )
    s.add_argument(
        "--model", default=None,
        help="HF model id (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit).",
    )
    s.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing directory.",
    )
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("list", help="List registered agents.")
    s.set_defaults(func=cmd_list_agents)

    s = sub.add_parser("query", help="One-shot ask an agent.")
    s.add_argument("agent")
    s.add_argument("question")
    s.add_argument("--max-tokens", type=int, default=None)
    s.add_argument("--temperature", type=float, default=None)
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("update", help="Record a note in an agent's knowledge.")
    s.add_argument("agent")
    s.add_argument("note")
    s.set_defaults(func=cmd_update)

    s = sub.add_parser(
        "remove", help="Unregister an agent (does not delete files)."
    )
    s.add_argument("name")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("swarm", help="Run a pipeline through the scheduler.")
    s.add_argument("pipeline", help="Path to a .prose or .yaml pipeline file.")
    s.add_argument(
        "--agent",
        action="append",
        default=None,
        help=(
            "Preload an agent (repeatable). Useful for warming caches "
            "before a run uses them."
        ),
    )
    s.add_argument(
        "--inputs",
        default=None,
        help='Initial inputs as JSON (e.g. \'{"x":"hello"}\').',
    )
    s.add_argument(
        "--journal-root",
        default=None,
        help="Where to write the journal (default: temp dir).",
    )
    s.add_argument("--max-concurrent", type=int, default=2)
    s.add_argument("--priority", type=float, default=5.0)
    s.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for the pipeline future.",
    )
    s.set_defaults(func=cmd_swarm)

    s = sub.add_parser("observe", help="Print or follow an agent's journal.")
    s.add_argument("agent", help="Registered agent name or directory path.")
    s.add_argument(
        "--topic",
        default="pipeline",
        help="Journal topic to read (default: pipeline).",
    )
    s.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Tail the journal (Ctrl-C to exit).",
    )
    s.add_argument(
        "-n",
        type=int,
        default=20,
        help="Number of trailing events to print (non-follow mode).",
    )
    s.set_defaults(func=cmd_observe)

    s = sub.add_parser(
        "germination-status",
        help="Aggregate T3 germination state across all registered agents.",
    )
    s.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table for humans).",
    )
    s.add_argument(
        "--output",
        default=None,
        help="Write the JSON report to this path (also prints table unless --format json).",
    )
    s.add_argument(
        "--fail-on-t3-exceed",
        type=float,
        default=None,
        dest="fail_on_t3_exceed",
        help=(
            "Exit 1 if the cross-agent T3 germination fail rate exceeds "
            "this fraction (e.g. 0.05 for the locked 5%% threshold)."
        ),
    )
    s.set_defaults(func=cmd_germination_status)

    s = sub.add_parser(
        "calibrate-wizard",
        help="Measure drafter quality against the smoke corpus; emit baseline.json.",
    )
    s.add_argument(
        "--drafter",
        default=None,
        help="HF model id of the drafter (default: from config).",
    )
    s.add_argument(
        "--mock",
        action="store_true",
        help="Use MockDrafter (replays fixtures; no MLX needed). For CI.",
    )
    s.add_argument(
        "--mock-fixtures",
        default=None,
        help="Path to mock_drafter_outputs.yaml (default: tests/fixtures/...).",
    )
    s.add_argument(
        "--corpus",
        default=None,
        help="Path to wizard_smoke_corpus.yaml (default: from config).",
    )
    s.add_argument(
        "--config",
        default=None,
        help="Path to a gardener config YAML (default: ~/.config/gardener/config.yaml or ./gardener.yaml).",
    )
    s.add_argument(
        "--replicas",
        type=int,
        default=1,
        help="Number of replicas per corpus entry (default: 1). Use 4 for 5%% T3 measurement on 10-entry strata.",
    )
    s.add_argument(
        "--output",
        required=True,
        help="Path to write baseline.json artifact.",
    )
    s.add_argument(
        "--t1-fail-max",
        type=float,
        default=None,
        dest="t1_fail_max",
        help="Override T1 parse-fail threshold (default: 0.10).",
    )
    s.add_argument(
        "--t2-fail-max",
        type=float,
        default=None,
        dest="t2_fail_max",
        help="Override T2 acceptance-fail threshold (default: 0.20).",
    )
    s.add_argument(
        "--t3-fail-max",
        type=float,
        default=None,
        dest="t3_fail_max",
        help="Override T3 germination-fail threshold (default: 0.05).",
    )
    s.add_argument(
        "--exit-nonzero-on-flip",
        action="store_true",
        help="Exit 1 when verdict says flip the default drafter (for CI gating).",
    )
    s.add_argument(
        "--t3-from-germination-status",
        default=None,
        dest="t3_from_germination_status",
        help=(
            "Path to a germination-status JSON dump; folds its T3 fail rate "
            "into this calibration's verdict (post-plant data closes the loop)."
        ),
    )
    s.set_defaults(func=cmd_calibrate_wizard)

    s = sub.add_parser("wizard", help="Interactive agent creator.")
    s.add_argument(
        "--inline",
        nargs="*",
        default=None,
        help=(
            "Non-interactive: KEY=VALUE pairs (name, description, "
            "purpose, model, temperature, max_tokens, path, "
            "seed_knowledge, force)."
        ),
    )
    s.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the final confirmation prompt.",
    )
    s.add_argument(
        "--drafter",
        default=None,
        help=(
            "HF model id of a local LLM to draft system prompt + seed "
            "knowledge from the purpose. When unset, uses the static "
            "template (existing Q2 behavior)."
        ),
    )
    s.add_argument(
        "--mock-drafter-fixtures",
        default=None,
        dest="mock_drafter_fixtures",
        help=(
            "Test-only: replay drafter responses from a fixture YAML. "
            "Bypasses MLX inference. Overrides --drafter when set."
        ),
    )
    s.set_defaults(func=cmd_wizard)

    # --- cache subcommands ---------------------------------------------------
    cache_parser = sub.add_parser(
        "cache", help="Manage warm subagent KV caches."
    )
    cache_sub = cache_parser.add_subparsers(dest="cache_cmd", required=True)

    w = cache_sub.add_parser("warm", help="Pre-fill a warm context cache.")
    w.add_argument("agent")
    w.add_argument(
        "--text",
        default=None,
        help="Warmup text to prefill into the cache.",
    )
    w.add_argument(
        "--file",
        default=None,
        help="Read warmup text from a file.",
    )
    w.add_argument(
        "--use-prompt-md",
        action="store_true",
        help="Use the agent's prompt.md as the warmup text (default).",
    )
    w.add_argument(
        "--name",
        default=None,
        help="Cache name. Default: warm-<sha256[:12]>.",
    )
    w.set_defaults(func=cmd_cache_warm)

    ll = cache_sub.add_parser("list", help="List warm caches for an agent.")
    ll.add_argument("agent")
    ll.set_defaults(func=cmd_cache_list)

    r = cache_sub.add_parser("rm", help="Delete a warm cache.")
    r.add_argument("agent")
    r.add_argument("cache_name")
    r.set_defaults(func=cmd_cache_rm)

    u = cache_sub.add_parser(
        "use", help="Run a one-shot query against a warm cache."
    )
    u.add_argument("agent")
    u.add_argument("cache_name")
    u.add_argument("question")
    u.set_defaults(func=cmd_cache_use)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except SystemExit:
        raise
    except Exception as e:
        print(f"gardener: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
