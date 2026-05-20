import pytest
from examples.demo_full import main

pytestmark = pytest.mark.model

def test_demo_full_runs_end_to_end():
    assert main() == 0
