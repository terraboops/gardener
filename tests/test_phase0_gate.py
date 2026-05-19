def test_public_surface_imports():
    from gardener.mlxsuper import (
        SessionPool, TTTEngine, Candidate, TrainStats,
        compute_layer_lrs, project_lora_grads, compute_svd_cache,
        timer, counter, registry, median_of_n,
    )
    assert callable(compute_layer_lrs)
    assert callable(project_lora_grads)
