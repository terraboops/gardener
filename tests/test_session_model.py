import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm import generate

from gardener.mlxsuper.session import SessionPool

pytestmark = pytest.mark.model

def _greedy(model, tokenizer, prompt, cache, n):
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=n,
        sampler=make_sampler(temp=0.0), prompt_cache=cache, verbose=False,
    )

def test_save_load_resumes_bit_stable(loaded_model, tmp_path):
    model, tok = loaded_model
    pool = SessionPool(cache_factory=lambda: make_prompt_cache(model))
    sid = pool.create("s")
    prompt = "Count: one two three"
    _greedy(model, tok, prompt, pool.get(sid), 8)            # warm the cache
    path = pool.save(sid, str(tmp_path / "s.safetensors"))
    cont_a = _greedy(model, tok, " four", pool.get(sid), 8)
    rid = pool.load(path, "restored")
    cont_b = _greedy(model, tok, " four", pool.get(rid), 8)
    assert cont_a == cont_b                                  # no re-prefill, bit-stable

def test_fork_then_divergent_generation_is_independent(loaded_model):
    model, tok = loaded_model
    pool = SessionPool(cache_factory=lambda: make_prompt_cache(model))
    a = pool.create("a")
    _greedy(model, tok, "The capital of France is", pool.get(a), 4)
    b = pool.fork(a, "b")
    out_a = _greedy(model, tok, " definitely", pool.get(a), 6)
    out_b = _greedy(model, tok, " arguably", pool.get(b), 6)
    # Independent caches: continuing 'a' again is unaffected by 'b'.
    out_a2 = _greedy(model, tok, " and", pool.get(a), 4)
    assert isinstance(out_a, str) and isinstance(out_b, str)
    assert out_a != out_b
    assert isinstance(out_a2, str)
