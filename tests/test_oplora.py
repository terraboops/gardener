import mlx.core as mx
from gardener.mlxsuper.oplora import project_lora_grads, _truncated_svd


def test_truncated_svd_shapes():
    W = mx.random.normal((12, 9))
    U_k, V_k = _truncated_svd(W, k=4)
    assert U_k.shape == (12, 4)
    assert V_k.shape == (9, 4)


def test_projection_removes_top_subspace_component():
    mx.random.seed(0)
    m, n, r, k = 16, 10, 3, 4
    W = mx.random.normal((m, n))
    U_k, V_k = _truncated_svd(W, k)
    A = mx.random.normal((r, n))
    B = mx.random.normal((m, r))
    dA = mx.random.normal((r, n))
    dB = mx.random.normal((m, r))
    dA_s, dB_s = project_lora_grads(W, A, B, dA, dB, {"U_k": U_k, "V_k": V_k})
    # Projected grads must be orthogonal to the removed subspace.
    assert float(mx.max(mx.abs(dA_s @ V_k))) < 1e-4
    assert float(mx.max(mx.abs(U_k.T @ dB_s))) < 1e-4
    # And idempotent: projecting again is a no-op.
    dA_s2, dB_s2 = project_lora_grads(W, A, B, dA_s, dB_s,
                                      {"U_k": U_k, "V_k": V_k})
    assert float(mx.max(mx.abs(dA_s2 - dA_s))) < 1e-4
    assert float(mx.max(mx.abs(dB_s2 - dB_s))) < 1e-4
