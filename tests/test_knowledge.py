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
