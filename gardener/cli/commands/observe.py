"""gardener observe — print or follow an agent's journal."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ...journal import DLQ_TOPIC, EventJournal
from ..registry import resolve


def cmd_observe(args) -> int:
    try:
        agent_path = resolve(args.agent)
    except KeyError as e:
        print(f"gardener observe: {e}", file=sys.stderr)
        return 1

    journal_root = agent_path / "journal"
    if not journal_root.exists():
        print(f"(no journal at {journal_root} — agent hasn't been run yet)")
        return 0

    journal = EventJournal(journal_root)

    def _dlq_banner() -> None:
        depth = journal.dlq_depth()
        print(f"[dlq depth: {depth}]", flush=True)

    topic = args.topic
    path = journal_root / f"{topic}.jsonl"

    if args.follow:
        # Tail-mode: print all existing lines, then poll for new ones.
        _dlq_banner()
        last_size = 0
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        print(line, flush=True)
            last_size = path.stat().st_size
        try:
            while True:
                time.sleep(0.25)
                if not path.exists():
                    continue
                size = path.stat().st_size
                if size > last_size:
                    with path.open(encoding="utf-8") as f:
                        f.seek(last_size)
                        chunk = f.read()
                    for line in chunk.splitlines():
                        if line.strip():
                            print(line, flush=True)
                    last_size = size
        except KeyboardInterrupt:
            return 0
        return 0

    # Non-follow: print last N events.
    _dlq_banner()
    events = list(journal.read(topic))
    tail = events[-args.n :]
    for ev in tail:
        print(json.dumps(ev, sort_keys=True), flush=True)
    return 0
