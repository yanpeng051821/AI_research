from minimind_lab.models.feedforward import SwiGLU
from torch import nn

import torch

def test_swiglu_forward_matches_manual():
    ffd = SwiGLU(2, 3)

    gate_weight = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
    ])
    up_weight = torch.tensor([
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    down_weight = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0],
    ])

    with torch.no_grad():
        ffd.gate_proj.weight.copy_(gate_weight)
        ffd.up_proj.weight.copy_(up_weight)
        ffd.down_proj.weight.copy_(down_weight)


    x = torch.tensor([[[1.0, 2.0]]], requires_grad=True)

    output = ffd(x)

    # 手工计算路径
    gate_output = nn.SiLU()(x @ gate_weight.T)
    up_output = x @ up_weight.T
    expected_output = (gate_output * up_output) @ down_weight.T

    torch.testing.assert_close(output, expected_output)


def test_swiglu_gradients_match_reference():
    ffd = SwiGLU(2, 3)
    gate_weight = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, -1.0],
    ])
    up_weight = torch.tensor([
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    down_weight = torch.tensor([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, -1.0],
    ])

    gate_reference = gate_weight.detach().clone().requires_grad_(True)
    up_reference = up_weight.detach().clone().requires_grad_(True)
    down_reference = down_weight.detach().clone().requires_grad_(True)

    with torch.no_grad():
        ffd.gate_proj.weight.copy_(gate_reference)
        ffd.up_proj.weight.copy_(up_reference)
        ffd.down_proj.weight.copy_(down_reference)


    x = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)

    silu_output = ffd(x)

    # 参考路径计算
    gate_output = x_reference @ gate_reference.T
    gate_act = gate_output * torch.sigmoid(gate_output)
    up_output = x_reference @ up_reference.T
    expected_output = (gate_act * up_output) @ down_reference.T

        # 测试反向
    silu_output.sum().backward()
    expected_output.sum().backward()

    torch.testing.assert_close(silu_output, expected_output)
    torch.testing.assert_close(x.grad, x_reference.grad)
    torch.testing.assert_close(ffd.gate_proj.weight.grad, gate_reference.grad)
    torch.testing.assert_close(ffd.up_proj.weight.grad, up_reference.grad)
    torch.testing.assert_close(ffd.down_proj.weight.grad, down_reference.grad)


def test_swiglu_does_not_mix_positions():
    ffd = SwiGLU(4, 8)

    x = torch.randn(2, 3, 4)

    x_changed = x.clone()
    x_changed[0, 1] += 10

    output = ffd(x)
    output_changed = ffd(x_changed)

    torch.testing.assert_close(output[0, 0], output_changed[0, 0])
    torch.testing.assert_close(output[0, 2], output_changed[0, 2])
    torch.testing.assert_close(output[1], output_changed[1])





