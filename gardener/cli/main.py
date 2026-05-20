"""Top-level Gardener CLI entry point.

   gardener init NAME [--path PATH] [--model HF_ID] [--force]
   gardener list
   gardener query NAME-OR-PATH "QUESTION" [--max-tokens N] [--temperature T]
   gardener update NAME-OR-PATH "NOTE"
   gardener remove NAME
   gardener swarm PIPELINE [--agent NAME ...] [--inputs JSON] [--journal-root DIR]
                           [--max-concurrent N] [--priority P] [--timeout S]
   gardener observe NAME-OR-PATH [--topic TOPIC] [--follow] [-n N]
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

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
