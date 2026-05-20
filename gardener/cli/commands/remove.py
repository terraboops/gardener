"""gardener remove -- unregister an agent (does not delete files)."""
from __future__ import annotations

import sys

from ..registry import load_registry, unregister


def cmd_remove(args) -> int:
    name: str = args.name
    reg = load_registry()
    if name not in reg:
        print(f"gardener remove: agent '{name}' is not registered.", file=sys.stderr)
        return 1
    path = reg[name]
    unregister(name)
    print(f"Unregistered agent '{name}' (files at {path} are untouched).")
    return 0
