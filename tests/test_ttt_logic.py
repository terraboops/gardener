from gardener.mlxsuper.ttt import Candidate, TrainStats, TTTEngine


def test_candidate_and_feedback_roundtrip():
    eng = TTTEngine.__new__(TTTEngine)      # no model needed for this logic
    eng.candidates = {}
    c = Candidate(id="c1", prompt="p", completion="x", tokens=[1, 2])
    eng.candidates["c1"] = c
    TTTEngine.feedback(eng, "c1", reward=1.0, signal="solution",
                       metadata={"passed": True})
    assert eng.candidates["c1"].reward == 1.0
    assert eng.candidates["c1"].signal == "solution"
    assert eng.candidates["c1"].metadata["passed"] is True


def test_feedback_unknown_candidate_is_safe():
    eng = TTTEngine.__new__(TTTEngine)
    eng.candidates = {}
    TTTEngine.feedback(eng, "missing", reward=1.0)   # must not raise


def test_trainstats_defaults():
    s = TrainStats()
    assert s.loss == 0.0 and s.num_positive == 0 and s.num_negative == 0
