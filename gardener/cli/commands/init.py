"""gardener init -- create a new agent directory and register it."""
from __future__ import annotations

import sys
from pathlib import Path

from ..agent_dir import DEFAULT_MODEL, scaffold_agent
from ..registry import register


def cmd_init(args) -> int:
    name: str = args.name
    model: str = args.model or DEFAULT_MODEL
    force: bool = args.force

    agent_path = Path(args.path) if args.path else Path.cwd() / name

    try:
        scaffold_agent(agent_path, name=name, model=model, force=force)
    except ValueError as e:
        print(f"gardener init: {e}", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"gardener init: {e}", file=sys.stderr)
        return 1

    register(name, agent_path)
    print(f"Created agent '{name}' at {agent_path.resolve()}")
    return 0
