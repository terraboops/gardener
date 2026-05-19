"""Persistent / forkable / rewindable sessions over stock mlx_lm caches.

Reference map (semantics only): omlx/turboquant_kv.py:1624 fork,
:1648 rewind_to, :1670 save_to_disk, :1722 load_from_disk;
omlx/hypercar_server.py:410-680 session model. Built on stock
mlx_lm.models.cache (make_prompt_cache/save_prompt_cache/
load_prompt_cache/KVCache.trim). No omlx import."""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any, Callable

from .observability import counter, timer


class SessionPool:
    """In-memory registry of named KV-cache sessions.

    A session's value is a `list[cache]` — one cache object per model
    layer (the stock mlx_lm prompt-cache shape). `cache_factory` builds
    a fresh empty cache list (real use: a closure over the model).
    """

    def __init__(self, cache_factory: Callable[[], list[Any]]):
        self._cache_factory = cache_factory
        self._sessions: dict[str, list[Any]] = {}
        self._parent: dict[str, str] = {}

    def create(self, name: str | None = None) -> str:
        """Create a new empty session with optional name."""
        sid = name or uuid.uuid4().hex[:8]
        if sid in self._sessions:
            raise KeyError(f"session exists: {sid}")
        self._sessions[sid] = self._cache_factory()
        counter("session.created").add(1)
        return sid

    def get(self, sid: str) -> list[Any]:
        """Retrieve a session's cache list by ID."""
        if sid not in self._sessions:
            raise KeyError(f"unknown session: {sid}")
        return self._sessions[sid]

    def parent_of(self, sid: str) -> str | None:
        """Return the parent session ID if this is a fork, else None."""
        return self._parent.get(sid)

    def fork(self, sid: str, new_name: str | None = None) -> str:
        """Independent deep copy — both branches generate without
        corrupting each other (semantics from turboquant_kv.fork)."""
        src = self.get(sid)
        nid = new_name or uuid.uuid4().hex[:8]
        if nid in self._sessions:
            raise KeyError(f"session exists: {nid}")
        with timer("session.fork"):
            self._sessions[nid] = copy.deepcopy(src)
        self._parent[nid] = sid
        counter("session.forked").add(1)
        return nid

    def rewind(self, sid: str, target_offset: int) -> int:
        """O(1) tail drop via stock cache.trim(); returns dropped count
        (semantics from turboquant_kv.rewind_to)."""
        caches = self.get(sid)
        offset = caches[0].offset
        n = max(0, offset - max(0, min(target_offset, offset)))
        with timer("session.rewind"):
            for c in caches:
                c.trim(n)
        counter("session.rewound").add(1)
        return n

    def save(self, sid: str, path: str, metadata: dict | None = None) -> str:
        """Freeze to disk via stock mlx_lm.save_prompt_cache."""
        from mlx_lm.models.cache import save_prompt_cache

        p = str(Path(path))
        with timer("session.save"):
            save_prompt_cache(p, self.get(sid), metadata or {})
        counter("session.saved").add(1)
        return p

    def load(self, path: str, name: str | None = None) -> str:
        """Resume from disk via stock mlx_lm.load_prompt_cache —
        no re-prefill (semantics from turboquant_kv.load_from_disk)."""
        from mlx_lm.models.cache import load_prompt_cache

        sid = name or uuid.uuid4().hex[:8]
        if sid in self._sessions:
            raise KeyError(f"session exists: {sid}")
        with timer("session.load"):
            self._sessions[sid] = load_prompt_cache(path)
        counter("session.loaded").add(1)
        return sid
