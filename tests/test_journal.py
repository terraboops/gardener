from gardener.journal import EventJournal


def test_append_read_roundtrip(tmp_path):
    j = EventJournal(tmp_path)
    j.append("main", {"k": 1})
    j.append("main", {"k": 2})
    assert list(j.read("main")) == [{"k": 1}, {"k": 2}]


def test_dead_letter_routes_to_dlq_topic(tmp_path):
    j = EventJournal(tmp_path)
    j.dead_letter({"node": "n1", "reason": "timeout"})
    items = list(j.read("dead_letter"))
    assert items == [{"node": "n1", "reason": "timeout"}]
    assert j.dlq_depth() == 1


def test_redrive_returns_and_clears_dlq(tmp_path):
    j = EventJournal(tmp_path)
    j.dead_letter({"id": "a"})
    j.dead_letter({"id": "b"})
    drained = j.redrive_dlq()
    assert [d["id"] for d in drained] == ["a", "b"]
    assert j.dlq_depth() == 0
