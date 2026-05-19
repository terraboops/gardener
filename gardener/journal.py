"""Append-only JSONL event journal with DLQ topic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

DLQ_TOPIC = "dead_letter"


class EventJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _topic_path(self, topic: str) -> Path:
        return self.root / f"{topic}.jsonl"

    def append(self, topic: str, event: dict) -> None:
        with self._topic_path(topic).open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def read(self, topic: str) -> Iterable[dict]:
        path = self._topic_path(topic)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def dead_letter(self, item: dict) -> None:
        self.append(DLQ_TOPIC, item)

    def dlq_depth(self) -> int:
        return sum(1 for _ in self.read(DLQ_TOPIC))

    def redrive_dlq(self) -> list[dict]:
        items = list(self.read(DLQ_TOPIC))
        self._topic_path(DLQ_TOPIC).write_text("", encoding="utf-8")
        return items
