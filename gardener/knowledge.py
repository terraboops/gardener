"""Per-agent knowledge store with empirical-feedback gate.

Reference map (semantics only — defects K2, K10 fixed here; K1/K4/K5/K8/K9
fixed in MVP+ Phase 1; K3/K6/K7 deferred to P2): omlx incubator
trellis/tools/knowledge_io.py + orchestrator/evolution.py."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
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


def _validate_idea_context(seq) -> None:
    """K9: every element must be a non-empty str."""
    for item in seq:
        if not isinstance(item, str):
            raise ValueError(
                f"idea_context elements must be str, got: {type(item).__name__!r}"
            )
        if not item:
            raise ValueError("idea_context elements must be non-empty strings")


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
        # K9: validate idea_context elements
        _validate_idea_context(self.idea_context)


class KnowledgeStore:
    def __init__(self, root: str | Path, agent: str, max_entries: int = 1000):
        self.dir = Path(root) / agent
        self.dir.mkdir(parents=True, exist_ok=True)
        self.agent = agent
        self.max_entries = max_entries          # K1
        self._lock = threading.Lock()           # K8

    def _path(self, oid: str) -> Path:
        return self.dir / f"{oid}.yaml"

    def _atomic_write(self, target: Path, content: str) -> None:
        """K5: write via tempfile + os.replace for atomicity."""
        fd, tmp = tempfile.mkstemp(prefix=f".{target.stem}.", dir=self.dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, target)
        except Exception:
            # Clean up the temp file if anything went wrong
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _evict_to_cap(self, incoming_id: str) -> None:
        """K1: delete oldest entries (by updated_at) until count < max_entries."""
        yaml_files = list(self.dir.glob("*.yaml"))
        # +1 because we are about to add one more (possibly)
        if len(yaml_files) < self.max_entries:
            return
        # Load (oid, updated_at) pairs for sorting; fall back to mtime on error
        entries: list[tuple[float, Path]] = []
        for p in yaml_files:
            if p.stem == incoming_id:
                # Treat the entry we're about to overwrite as if it already
                # exists — don't evict it; it will be overwritten in place.
                continue
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                ts = float(data.get("updated_at", p.stat().st_mtime))
            except Exception:
                ts = p.stat().st_mtime
            entries.append((ts, p))
        entries.sort(key=lambda x: x[0])
        # Delete oldest until we have room for the new entry
        excess = len(yaml_files) - self.max_entries + 1
        for _, p in entries[:excess]:
            try:
                p.unlink()
            except OSError:
                pass

    def write(self, obj: KnowledgeObject) -> str:
        with self._lock:                        # K8
            # K2: strict validation — re-normalize all predicates (raises on bad)
            [_normalize_triple(t) for t in obj.predicates]
            # Re-construct to ensure all field defaults are consistent
            obj = KnowledgeObject(**asdict(obj))
            obj.updated_at = time.time()
            # K1: evict before writing
            self._evict_to_cap(obj.id)
            # K5: atomic write
            self._atomic_write(
                self._path(obj.id),
                yaml.safe_dump(asdict(obj), sort_keys=True),
            )
            return obj.id

    def get(self, oid: str) -> KnowledgeObject:
        data = yaml.safe_load(self._path(oid).read_text(encoding="utf-8"))
        return KnowledgeObject(**data)

    def get_strict(self, oid: str) -> KnowledgeObject:
        """K4: load and validate; raises ValueError on any malformed content."""
        try:
            data = yaml.safe_load(self._path(oid).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("YAML root is not a mapping")
            obj = KnowledgeObject(**data)
            # Explicitly normalize all predicates — raises ValueError on bad triples.
            [_normalize_triple(t) for t in obj.predicates]
            return obj
        except (ValueError, TypeError, yaml.YAMLError, FileNotFoundError) as e:
            raise ValueError(f"malformed knowledge object {oid}: {e}") from e

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
        with self._lock:                        # K8
            obj = self.get(oid)
            obj.empirical["tested"] = True
            obj.empirical["uses"] = obj.empirical.get("uses", 0) + 1
            if helped:
                obj.empirical["helped"] = obj.empirical.get("helped", 0) + 1
            obj.empirical["last_validated_at"] = time.time()
            obj.updated_at = time.time()
            # K5: atomic write
            self._atomic_write(
                self._path(oid),
                yaml.safe_dump(asdict(obj), sort_keys=True),
            )

    def consolidatable(self) -> Iterable[KnowledgeObject]:
        """K10: only tested knowledge is eligible for weight-tier consolidation."""
        for obj in self.all():
            if obj.empirical.get("tested"):
                yield obj
