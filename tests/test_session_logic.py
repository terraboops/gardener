from gardener.mlxsuper.session import SessionPool


class FakeCache:
    """Mimics mlx_lm KVCache trim/offset semantics with a python list."""
    def __init__(self):
        self.tokens = []
        self.offset = 0

    def append(self, t):
        self.tokens.append(t)
        self.offset += 1

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        del self.tokens[self.offset:]
        return n


def _mk_pool():
    return SessionPool(cache_factory=lambda: [FakeCache()])


def test_create_and_fork_are_independent():
    pool = _mk_pool()
    a = pool.create("a")
    pool.get(a)[0].append("x")
    pool.get(a)[0].append("y")
    b = pool.fork(a, "b")
    pool.get(b)[0].append("z")          # diverge child
    pool.get(a)[0].append("w")          # diverge parent
    assert pool.get(a)[0].tokens == ["x", "y", "w"]
    assert pool.get(b)[0].tokens == ["x", "y", "z"]
    assert pool.parent_of(b) == a


def test_rewind_drops_tail_o1():
    pool = _mk_pool()
    s = pool.create("s")
    for t in "abcde":
        pool.get(s)[0].append(t)
    dropped = pool.rewind(s, 2)
    assert dropped == 3
    assert pool.get(s)[0].tokens == ["a", "b"]
    assert pool.get(s)[0].offset == 2


def test_unknown_session_raises():
    pool = _mk_pool()
    import pytest
    with pytest.raises(KeyError):
        pool.get("nope")
