import pytest
from gardener.knowledge import KnowledgeStore, KnowledgeObject


def test_write_then_read(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    obj = KnowledgeObject(
        predicates=[["sky", "is", "blue"]],
        insight="visible-light scattering",
        justification="Rayleigh scattering",
        source_agent="alpha",
    )
    oid = s.write(obj)
    loaded = s.get(oid)
    assert loaded.insight == "visible-light scattering"
    assert loaded.empirical["tested"] is False


def test_malformed_predicates_rejected_loudly(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    bad = KnowledgeObject(
        predicates=[["only", "two"]],  # length 2, not triple
        insight="x", justification="y", source_agent="alpha",
    )
    with pytest.raises(ValueError, match="predicate"):
        s.write(bad)


def test_search_by_predicate_overlap(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="x",
                            justification="y", source_agent="alpha"))
    s.write(KnowledgeObject(predicates=[["c", "r", "d"]], insight="z",
                            justification="y", source_agent="alpha"))
    hits = s.search([["a", "r", "b"]])
    assert len(hits) == 1 and hits[0].insight == "x"


def test_empirical_gate_only_tested_consolidates(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = s.write(KnowledgeObject(
        predicates=[["a", "r", "b"]], insight="i1", justification="j",
        source_agent="alpha"))
    b = s.write(KnowledgeObject(
        predicates=[["c", "r", "d"]], insight="i2", justification="j",
        source_agent="alpha"))
    s.mark_tested(a, helped=True)
    consolidatable = list(s.consolidatable())
    assert [c.id for c in consolidatable] == [a]


def test_K1_cap_evicts_oldest_by_updated_at(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha", max_entries=3)
    # Write 4 with deterministic, increasing updated_at via direct file writes.
    ids = []
    for i, sub in enumerate(["a", "b", "c", "d"]):
        ids.append(s.write(KnowledgeObject(
            predicates=[[sub, "r", "x"]], insight=f"i{i}",
            justification="j", source_agent="alpha")))
    surviving = {obj.id for obj in s.all()}
    # oldest (first written) should have been evicted
    assert ids[0] not in surviving
    assert len(surviving) == 3


def test_K4_get_strict_raises_on_malformed_file(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    bad_path = s.dir / "deadbeef00000000.yaml"
    bad_path.write_text("predicates: [[only, two]]\ninsight: x\n"
                        "justification: y\nsource_agent: alpha\nid: deadbeef00000000\n")
    with pytest.raises(ValueError):
        s.get_strict("deadbeef00000000")


def test_K5_atomic_write_no_partial_reads(tmp_path):
    # Smoke: simply verify the atomic temp file is gone after a successful write.
    s = KnowledgeStore(tmp_path, agent="alpha")
    oid = s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                                   justification="j", source_agent="alpha"))
    leftover = list(s.dir.glob(".*"))   # no hidden tempfiles left
    assert leftover == []
    # And the final file is present.
    assert (s.dir / f"{oid}.yaml").exists()


def test_K8_concurrent_writers_no_lost_update(tmp_path):
    import threading
    s = KnowledgeStore(tmp_path, agent="alpha", max_entries=10_000)
    written = []
    def worker(i):
        oid = s.write(KnowledgeObject(
            predicates=[[f"k{i}", "r", "v"]], insight=f"i{i}",
            justification="j", source_agent="alpha"))
        written.append(oid)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(written) == 20
    assert len({obj.id for obj in s.all()}) == 20  # no lost update


def test_K9_idea_context_validated(tmp_path):
    with pytest.raises(ValueError):
        KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                        justification="j", source_agent="alpha",
                        idea_context=[""])     # empty string rejected
    with pytest.raises(ValueError):
        KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                        justification="j", source_agent="alpha",
                        idea_context=[123])    # non-str rejected
    # Round-trip preserves order:
    s = KnowledgeStore(tmp_path, agent="alpha")
    oid = s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                                   justification="j", source_agent="alpha",
                                   idea_context=["one", "two", "three"]))
    assert s.get(oid).idea_context == ["one", "two", "three"]
