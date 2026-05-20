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


def test_K3_collision_with_different_content_raises(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = KnowledgeObject(predicates=[["x", "r", "y"]], insight="A",
                        justification="j", source_agent="alpha")
    s.write(a)
    # Force a collision: same id, different predicates.
    b = KnowledgeObject(predicates=[["different", "r", "thing"]],
                        insight="B", justification="j",
                        source_agent="alpha", id=a.id)
    with pytest.raises(RuntimeError, match="collision"):
        s.write(b)


def test_K3_same_content_is_upsert(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = KnowledgeObject(predicates=[["x", "r", "y"]], insight="A",
                        justification="j", source_agent="alpha")
    oid = s.write(a)
    # Same predicates → same hash → upsert allowed.
    a2 = KnowledgeObject(predicates=[["x", "r", "y"]], insight="A updated",
                         justification="j", source_agent="alpha")
    assert s.write(a2) == oid
    assert s.get(oid).insight == "A updated"


def test_K6_invalid_action_raises_loud(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                                 justification="j", source_agent="alpha"))
    with pytest.raises(ValueError):
        s.apply_curation([{"action": "obliterate", "id": a}])  # bad action
    with pytest.raises(ValueError):
        s.apply_curation([{"action": "drop"}])                  # missing id
    with pytest.raises(ValueError):
        s.apply_curation([{"action": "merge", "ids": [a], "into_predicates": [["a", "r", "b"]]}])  # only 1 id


def test_K6_valid_actions_apply(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = s.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="A",
                                 justification="j", source_agent="alpha"))
    b = s.write(KnowledgeObject(predicates=[["c", "r", "d"]], insight="B",
                                 justification="j", source_agent="alpha"))
    s.apply_curation([{"action": "drop", "id": a}])
    assert {obj.id for obj in s.all()} == {b}


def test_K7_merge_recomputes_id_and_combines_provenance(tmp_path):
    s = KnowledgeStore(tmp_path, agent="alpha")
    a = s.write(KnowledgeObject(predicates=[["x", "is", "y"]], insight="A",
                                 justification="ja", source_agent="alpha",
                                 idea_context=["idea-1"], confidence=0.6))
    b = s.write(KnowledgeObject(predicates=[["x", "was", "y"]], insight="B",
                                 justification="jb", source_agent="alpha",
                                 idea_context=["idea-2"], confidence=0.8))
    new_id = s.merge([a, b], into_predicates=[["x", "relates_to", "y"]])
    # Sources gone; new id distinct.
    assert new_id != a and new_id != b
    surviving = {obj.id for obj in s.all()}
    assert surviving == {new_id}
    merged = s.get(new_id)
    assert "A" in merged.insight and "B" in merged.insight
    assert set(merged.idea_context) == {"idea-1", "idea-2"}
    assert merged.confidence == 0.8


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
