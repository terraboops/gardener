import pytest

TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

@pytest.fixture(scope="session")
def loaded_model():
    from mlx_lm import load
    model, tokenizer = load(TEST_MODEL)
    return model, tokenizer
