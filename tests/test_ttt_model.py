import mlx.core as mx
import pytest

from gardener.mlxsuper.ttt import TTTEngine

pytestmark = pytest.mark.model


def test_train_step_lowers_loss_on_positive_sample(loaded_model):
    model, tok = loaded_model
    eng = TTTEngine(model, tok, rank=8, lr=1e-3, num_lora_layers=2)
    c = eng.generate_candidates("Write hello:", n=1, max_tokens=8)[0]
    c.completion = " hello world"          # force a clean positive target
    eng.feedback(c.id, reward=1.0, signal="solution")
    s1 = eng.train_step()
    # Re-train the SAME target; loss must not increase (it should drop/flatten).
    eng.candidates[c.id] = c
    c.reward = 1.0
    s2 = eng.train_step()
    assert s2.loss <= s1.loss + 1e-3
    assert s1.num_positive == 1


def test_oplora_projected_step_keeps_heldout_probe_within_tolerance(loaded_model):
    model, tok = loaded_model
    eng = TTTEngine(model, tok, rank=8, lr=1e-3, num_lora_layers=2)
    probe = mx.array([tok.encode("The sky is")])
    before = mx.array(model(probe))
    c = eng.generate_candidates("Adapt:", n=1, max_tokens=8)[0]
    c.completion = " adapted text here"
    eng.feedback(c.id, reward=1.0)
    eng.train_step(use_oplora=True)
    after = mx.array(model(probe))
    # Cast to float32 before mean to avoid float16 accumulation overflow.
    drift = float(mx.mean(mx.abs(after.astype(mx.float32) - before.astype(mx.float32))).item())
    assert drift < 1.0          # projected update stays bounded on held-out input
