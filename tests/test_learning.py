import time
from gardener.knowledge import KnowledgeObject, KnowledgeStore
from gardener.learning import CultivationHook


class FakeEngine:
    def __init__(self):
        self.fed = []

    def feedback(self, cid, reward, signal="solution", metadata=None):
        self.fed.append((cid, reward, signal))


def test_hook_marks_knowledge_tested_and_feeds_ttt(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(
        predicates=[["x", "is", "y"]], insight="i",
        justification="j", source_agent="alpha"))
    eng = FakeEngine()
    hook = CultivationHook(store=store, ttt=eng)
    hook.record_success(candidate_id="c1", knowledge_ids=[oid])
    assert store.get(oid).empirical["tested"] is True
    assert eng.fed == [("c1", 1.0, "solution")]
