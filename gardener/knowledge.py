"""Per-agent knowledge store with empirical-feedback gate.

Reference map (semantics only — defects K2, K10 fixed here; K1/K3-K9
deferred to spec Phase 4 deep rebuild): omlx incubator
trellis/tools/knowledge_io.py + orchestrator/evolution.py."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


def _normalize_triple(t) -> tuple[str, str, str]:
    if not isinstance(t, (list, tuple)) or len(t) != 3:
        raise ValueError(f"predicate must be a 3-tuple, got: {t!r}")
    s, p, o = t
    if not all(isinstance(x, str) and x for x in (s, p, o)):
        raise ValueError(f"predicate elements must be non-empty strs: {t!r}")
    return s, p, o


def semantic_hash(predicates: list[list[str]]) -> str:
    norm = sorted(_normalize_triple(t) for t in predicates)
    return hashlib.sha256(
        json.dumps(norm, sort_keys=True).encode()
    ).hexdigest()[:16]      # 16 chars (K3 wider than trellis's 8)


@dataclass
class KnowledgeObject:
    predicates: list[list[str]]
    insight: str
    justification: str
    source_agent: str
    id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 0.5
    idea_context: list[str] = field(default_factory=list)
    empirical: dict = field(default_factory=lambda: {
        "tested": False, "uses": 0, "helped": 0, "last_validated_at": None,
    })

    def __post_init__(self):
        # K2: generate ID if not provided (validation deferred to write time)
        if not self.id:
            try:
                self.id = semantic_hash(self.predicates)
            except ValueError:
                # If predicates are malformed, defer to write() for loud rejection
                self.id = uuid.uuid4().hex[:16]


class KnowledgeStore:
    def __init__(self, root: str | Path, agent: str):
        self.dir = Path(root) / agent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.agent = agent

    def _path(self, oid: str) -> Path:
        return self.dir / f"{oid}.yaml"

    def write(self, obj: KnowledgeObject) -> str:
        # K2: strict validation — re-normalize all predicates (raises on bad)
        [_normalize_triple(t) for t in obj.predicates]
        # Re-construct to ensure all field defaults are consistent
        obj = KnowledgeObject(**asdict(obj))
        obj.updated_at = time.time()
        self._path(obj.id).write_text(
            yaml.safe_dump(asdict(obj), sort_keys=True), encoding="utf-8")
        return obj.id

    def get(self, oid: str) -> KnowledgeObject:
        data = yaml.safe_load(self._path(oid).read_text(encoding="utf-8"))
        return KnowledgeObject(**data)

    def all(self) -> Iterable[KnowledgeObject]:
        for p in sorted(self.dir.glob("*.yaml")):
            try:
                yield KnowledgeObject(**yaml.safe_load(p.read_text()))
            except (ValueError, TypeError, yaml.YAMLError):
                # K2: malformed entries are SKIPPED on read (logged later)
                continue

    def search(self, predicates: list[list[str]]) -> list[KnowledgeObject]:
        want = {tuple(_normalize_triple(t)) for t in predicates}
        out = []
        for obj in self.all():
            have = {tuple(_normalize_triple(t)) for t in obj.predicates}
            if want & have:
                out.append(obj)
        return out

    def mark_tested(self, oid: str, *, helped: bool) -> None:
        obj = self.get(oid)
        obj.empirical["tested"] = True
        obj.empirical["uses"] = obj.empirical.get("uses", 0) + 1
        if helped:
            obj.empirical["helped"] = obj.empirical.get("helped", 0) + 1
        obj.empirical["last_validated_at"] = time.time()
        obj.updated_at = time.time()
        self._path(oid).write_text(
            yaml.safe_dump(asdict(obj), sort_keys=True), encoding="utf-8")

    def consolidatable(self) -> Iterable[KnowledgeObject]:
        """K10: only tested knowledge is eligible for weight-tier consolidation."""
        for obj in self.all():
            if obj.empirical.get("tested"):
                yield obj
