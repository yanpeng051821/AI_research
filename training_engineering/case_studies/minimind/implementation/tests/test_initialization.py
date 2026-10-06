
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.embedding import TokenEmbedding

from torch import nn

import torch, pytest

def test_initialization():
    linear = nn.Linear(4, 3, bias=True)
    with torch.no_grad():
        linear.bias.fill_(2.0)
        initialize_weights(linear)
    assert torch.equal(linear.bias, torch.zeros_like(linear.bias))

    rms_norm = RMSNorm(4)
    with torch.no_grad():
        rms_norm.weight.fill_(2.0)
        initialize_weights(rms_norm)
    assert torch.equal(rms_norm.weight, torch.ones_like(rms_norm.weight))



@pytest.mark.parametrize(
    "module",
    [nn.Linear(4, 3, bias=False), TokenEmbedding(6, 4)]
)
def test_each_module_normal_initialization(module):
    expected = torch.empty_like(module.weight)

    torch.manual_seed(42)
    nn.init.normal_(expected, mean=0.0, std=0.02)

    torch.manual_seed(42)
    initialize_weights(module)

    torch.testing.assert_close(module.weight, expected)

