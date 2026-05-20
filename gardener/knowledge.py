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
            # K3: collision detection — if same id exists but different content, raise
            target = self._path(obj.id)
            if target.exists():
                try:
                    existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
                    existing_norm = sorted(
                        _normalize_triple(t) for t in (existing.get("predicates") or [])
                    )
                    incoming_norm = sorted(_normalize_triple(t) for t in obj.predicates)
                    if existing_norm != incoming_norm:
                        raise RuntimeError(
                            f"hash collision on {obj.id}: existing predicates "
                            f"{existing_norm!r} differ from incoming {incoming_norm!r}"
                        )
                except RuntimeError:
                    raise
                except Exception:
                    pass  # malformed existing file — allow overwrite
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

    def merge(self, ids: list[str], into_predicates: list[list[str]]) -> str:
        """K7: merge multiple knowledge objects into a single fresh one.

        The new object's id is recomputed from ``into_predicates`` (may or may
        not coincide with a source id — that is fine; if it does the source
        entry is simply overwritten with the merged content).  Source files are
        deleted after a successful write.
        """
        if len(ids) < 2:
            raise ValueError(f"merge requires at least 2 ids, got {ids!r}")
        with self._lock:
            sources: list[KnowledgeObject] = []
            for oid in ids:
                p = self._path(oid)
                if not p.exists():
                    raise ValueError(f"merge: unknown id {oid!r}")
                sources.append(KnowledgeObject(**yaml.safe_load(p.read_text(encoding="utf-8"))))

            # K7: combine provenance fields
            seen_insights: list[str] = []
            seen_justs: list[str] = []
            for src in sources:
                if src.insight not in seen_insights:
                    seen_insights.append(src.insight)
                if src.justification not in seen_justs:
                    seen_justs.append(src.justification)

            idea_ctx: list[str] = []
            seen_ctx: set[str] = set()
            for src in sources:
                for item in src.idea_context:
                    if item not in seen_ctx:
                        seen_ctx.add(item)
                        idea_ctx.append(item)
            idea_ctx = sorted(idea_ctx)

            max_conf = max(src.confidence for src in sources)
            emp_tested = any(src.empirical.get("tested") for src in sources)
            emp_uses = sum(src.empirical.get("uses", 0) for src in sources)
            emp_helped = sum(src.empirical.get("helped", 0) for src in sources)
            lv_candidates = [
                src.empirical.get("last_validated_at")
                for src in sources
                if src.empirical.get("last_validated_at") is not None
            ]
            emp_lv = max(lv_candidates) if lv_candidates else None

            merged = KnowledgeObject(
                predicates=into_predicates,
                insight=" | ".join(seen_insights),
                justification=" | ".join(seen_justs),
                source_agent=self.agent,
                idea_context=idea_ctx,
                confidence=max_conf,
                empirical={
                    "tested": emp_tested,
                    "uses": emp_uses,
                    "helped": emp_helped,
                    "last_validated_at": emp_lv,
                },
            )
            # id is computed fresh from into_predicates by __post_init__
            new_id = merged.id
            merged.updated_at = time.time()

            # Write merged object (K5 atomic; K3 collision check runs inside write
            # but we hold the lock so call _atomic_write directly to avoid deadlock)
            self._evict_to_cap(new_id)
            self._atomic_write(
                self._path(new_id),
                yaml.safe_dump(asdict(merged), sort_keys=True),
            )

            # Delete source files (skip if source id == new_id — already overwritten)
            for oid in ids:
                if oid != new_id:
                    try:
                        self._path(oid).unlink()
                    except OSError:
                        pass

            return new_id

    def apply_curation(self, actions: list[dict]) -> None:
        """K6: apply a validated list of curator actions.

        Each action is a dict with ``action`` ∈ {keep, drop, merge}.

        * ``keep``  — requires ``id`` (str); no-op.
        * ``drop``  — requires ``id`` (str); deletes the file.
        * ``merge`` — requires ``ids`` (list[str], ≥2) and
          ``into_predicates`` (list of triples); calls ``self.merge()``.

        Raises ``ValueError`` on any validation failure.  Never silently skips.
        """
        # Collect all known ids upfront for unknown-id checks
        known_ids = {p.stem for p in self.dir.glob("*.yaml")}

        for action_dict in actions:
            if not isinstance(action_dict, dict):
                raise ValueError(f"action must be a dict, got: {action_dict!r}")
            action_type = action_dict.get("action")
            if action_type not in ("keep", "drop", "merge"):
                raise ValueError(
                    f"unknown action {action_type!r} in: {action_dict!r}; "
                    "must be one of keep/drop/merge"
                )

            if action_type in ("keep", "drop"):
                if "id" not in action_dict:
                    raise ValueError(f"action {action_type!r} requires 'id': {action_dict!r}")
                oid = action_dict["id"]
                if not isinstance(oid, str):
                    raise ValueError(f"'id' must be str, got {type(oid).__name__!r}: {action_dict!r}")
                if oid not in known_ids:
                    raise ValueError(f"unknown id {oid!r} in action: {action_dict!r}")
                if action_type == "drop":
                    try:
                        self._path(oid).unlink()
                    except OSError:
                        pass
                    known_ids.discard(oid)
                # keep → no-op

            else:  # merge
                if "ids" not in action_dict:
                    raise ValueError(f"action 'merge' requires 'ids': {action_dict!r}")
                if "into_predicates" not in action_dict:
                    raise ValueError(f"action 'merge' requires 'into_predicates': {action_dict!r}")
                merge_ids = action_dict["ids"]
                if not isinstance(merge_ids, list) or len(merge_ids) < 2:
                    raise ValueError(
                        f"'ids' must be a list of ≥2 str ids: {action_dict!r}"
                    )
                for mid in merge_ids:
                    if not isinstance(mid, str):
                        raise ValueError(
                            f"each element of 'ids' must be str, got {type(mid).__name__!r}: {action_dict!r}"
                        )
                    if mid not in known_ids:
                        raise ValueError(f"unknown id {mid!r} in merge action: {action_dict!r}")
                new_id = self.merge(merge_ids, action_dict["into_predicates"])
                # Update known_ids to reflect the merge
                for mid in merge_ids:
                    known_ids.discard(mid)
                known_ids.add(new_id)

    def consolidatable(self) -> Iterable[KnowledgeObject]:
        """K10: only tested knowledge is eligible for weight-tier consolidation."""
        for obj in self.all():
            if obj.empirical.get("tested"):
                yield obj
