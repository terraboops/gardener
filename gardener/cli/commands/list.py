"""gardener list -- print registered agents."""
from __future__ import annotations

import sys
from pathlib import Path

from ..agent_dir import load_agent
from ..registry import load_registry


def cmd_list_agents(args) -> int:
    reg = load_registry()
    if not reg:
        print("No agents registered. Run `gardener init <name>` to create one.")
        return 0

    # Print a simple table: name, path, model
    col_name = max(len(n) for n in reg) + 2
    col_name = max(col_name, 8)
    print(f"{'NAME':<{col_name}}  {'MODEL':<40}  PATH")
    print("-" * (col_name + 2 + 40 + 2 + 20))
    for name, path_str in sorted(reg.items()):
        path = Path(path_str)
        try:
            cfg = load_agent(path)
            model = cfg.model
        except (FileNotFoundError, ValueError):
            model = "<missing agent.yaml>"
        print(f"{name:<{col_name}}  {model:<40}  {path_str}")
    return 0
