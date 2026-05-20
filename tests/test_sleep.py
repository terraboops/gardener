import pytest

from gardener.knowledge import KnowledgeObject, KnowledgeStore
from gardener.sleep import SleepCycle, SleepResult


# --- pure-logic tests (no model) ---------------------------------------------

class _FakeTokenizer:
    def encode(self, s): return [ord(c) % 256 for c in s][:32]

class _FakeEngine:
    def __init__(self, *, drift_after=0.1, raise_train=False):
        self.tokenizer = _FakeTokenizer()
        self.candidates = {}
        self.checkpointed = False
        self.rewound = False
        self._drift = drift_after
        self._raise = raise_train
        # Fake "model" callable: returns a fixed tensor pre/post,
        # changed after training to simulate drift.
        import mlx.core as mx
        self._t = mx.array([[1.0, 2.0, 3.0]])
    def model(self, ids):
        import mlx.core as mx
        return self._t
    def save_checkpoint(self): self.checkpointed = True
    def rewind(self): self.rewound = True
    def train_step(self, *, use_oplora=False):
        if self._raise:
            raise RuntimeError("boom")
        import mlx.core as mx
        # Simulate post-train shift on the probe tensor.
        self._t = self._t + self._drift
        return None


def test_sleep_no_op_when_no_consolidatable(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    store.write(KnowledgeObject(predicates=[["a", "r", "b"]], insight="i",
                                 justification="j", source_agent="alpha"))
    eng = _FakeEngine()
    cycle = SleepCycle(eng, store, probe_prompts=["hi"])
    res = cycle.run()
    assert res.state == "no_op"
    assert res.trained == 0
    assert eng.checkpointed is False

def test_sleep_promoted_when_drift_under_threshold(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(predicates=[["a", "r", "b"]],
                                       insight="i", justification="j",
                                       source_agent="alpha"))
    store.mark_tested(oid, helped=True)
    eng = _FakeEngine(drift_after=0.01)
    cycle = SleepCycle(eng, store, probe_prompts=["hi"], max_drift=0.5)
    res = cycle.run()
    assert res.state == "promoted"
    assert res.trained == 1
    assert eng.checkpointed is True
    assert eng.rewound is False

def test_sleep_rewound_when_drift_over_threshold(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(predicates=[["a", "r", "b"]],
                                       insight="i", justification="j",
                                       source_agent="alpha"))
    store.mark_tested(oid, helped=True)
    eng = _FakeEngine(drift_after=10.0)
    cycle = SleepCycle(eng, store, probe_prompts=["hi"], max_drift=0.5)
    res = cycle.run()
    assert res.state == "rewound"
    assert eng.rewound is True

def test_sleep_rewinds_on_train_crash(tmp_path):
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(predicates=[["a", "r", "b"]],
                                       insight="i", justification="j",
                                       source_agent="alpha"))
    store.mark_tested(oid, helped=True)
    eng = _FakeEngine(raise_train=True)
    cycle = SleepCycle(eng, store, probe_prompts=["hi"])
    res = cycle.run()
    assert res.state == "rewound"


# --- model integration -------------------------------------------------------

pytest_model = pytest.mark.model

@pytest_model
def test_sleep_terminal_state_on_real_model(loaded_model, tmp_path):
    from gardener.mlxsuper.ttt import TTTEngine
    model, tok = loaded_model
    eng = TTTEngine(model, tok, rank=4, lr=1e-4, num_lora_layers=1)
    store = KnowledgeStore(tmp_path, agent="alpha")
    oid = store.write(KnowledgeObject(
        predicates=[["sky", "appears", "blue"]],
        insight="The sky appears blue.",
        justification="Rayleigh scattering of sunlight.",
        source_agent="alpha"))
    store.mark_tested(oid, helped=True)
    cycle = SleepCycle(eng, store, probe_prompts=["The sky is"],
                       max_drift=1e6)   # very loose so we usually promote
    res = cycle.run()
    assert res.state in {"promoted", "rewound"}
    assert res.trained == 1
    assert isinstance(res.drift, float)
