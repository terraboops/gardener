"""gardener update -- record a note as a KnowledgeObject in an agent's store."""
from __future__ import annotations

import sys

from ..agent_dir import load_agent
from ..registry import resolve


def cmd_update(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener update: {e}", file=sys.stderr)
        return 1

    try:
        cfg = load_agent(agent_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"gardener update: {e}", file=sys.stderr)
        return 1

    from ...knowledge import KnowledgeObject, KnowledgeStore

    store = KnowledgeStore(root=agent_path / "knowledge", agent=cfg.name)
    obj = KnowledgeObject(
        predicates=[["note", "from", "human"]],
        insight=args.note,
        justification="recorded via `gardener update`",
        source_agent=cfg.name,
    )
    oid = store.write(obj)
    print(f"Recorded knowledge object {oid} in {store.dir}")
    return 0
