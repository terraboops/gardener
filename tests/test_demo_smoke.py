import pytest
from examples.demo_simple_agent import main

pytestmark = pytest.mark.model


def test_demo_runs_end_to_end():
    assert main() == 0
