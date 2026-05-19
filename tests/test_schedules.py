import math
import pytest
from gardener.mlxsuper.schedules import compute_layer_lrs

def test_uniform_is_flat():
    assert compute_layer_lrs(1e-4, n_layers=8, schedule="uniform") == [1e-4] * 8

def test_reservoir_is_monotone_nondecreasing_zero_first_base_last():
    lrs = compute_layer_lrs(1.0, n_layers=10, schedule="reservoir", gamma=1.5)
    assert lrs[0] == 0.0
    assert math.isclose(lrs[-1], 1.0, rel_tol=1e-9)
    assert all(b >= a for a, b in zip(lrs, lrs[1:]))

def test_cosine_is_u_shaped_edges_high_middle_low_min_10pct():
    lrs = compute_layer_lrs(1.0, n_layers=11, schedule="cosine")
    assert lrs[0] > lrs[len(lrs) // 2] < lrs[-1]
    assert min(lrs) >= 0.1 - 1e-9

def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        compute_layer_lrs(1e-4, n_layers=4, schedule="bogus")
