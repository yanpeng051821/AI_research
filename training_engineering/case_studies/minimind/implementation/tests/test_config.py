import pytest

from minimind_lab.models.config import ModelConfig

@pytest.mark.parametrize(
    "field",
    ["vocab_size", "hidden_size", "intermediate_size", "rms_norm_eps"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_model_config_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


def test_model_config_valid_values():
    config = ModelConfig()
    assert config.vocab_size == 6400
    assert config.hidden_size == 64
    assert config.intermediate_size == 128
    assert config.rms_norm_eps == 1e-5

    personal_config = ModelConfig(
        vocab_size=20,
        hidden_size=8,
        intermediate_size=4,
        rms_norm_eps=1e-6,
    )
    assert personal_config.vocab_size == 20
    assert personal_config.hidden_size == 8
    assert personal_config.intermediate_size == 4
    assert personal_config.rms_norm_eps == 1e-6


    

