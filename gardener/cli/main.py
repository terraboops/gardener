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
