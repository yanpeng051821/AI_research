from minimind_lab.models.rmsnorm import RMSNorm
from torch import nn


import torch, math


def test_rmsnorm_forward_matches_manual():
    resnorm = RMSNorm(2)
    with torch.no_grad():
        resnorm.weight.copy_(torch.tensor([2.0, 0.5]))

    x = torch.tensor([[[3.0, 4.0], [30.0, 40.0]]])

    resnorm_output = resnorm(x)
    print(f"resnorm_output: {resnorm_output}")
    print(f"resnorm_output.shape: {resnorm_output.shape}")

    assert resnorm_output.shape == (1, 2, 2)
    assert resnorm_output.dtype == torch.float32

    y1 = math.sqrt((3.0**2 + 4.0**2)/2 + 1e-5)
    y2 = math.sqrt((30.0**2 + 40.0**2)/2 + 1e-5)

    manual_output = torch.tensor([[[2.0 * 3.0 / y1, 0.5 * 4.0 / y1], [2 * 30.0 / y2, 0.5 * 40.0 / y2]]])

    torch.testing.assert_close(resnorm_output, manual_output)

def test_rmsnorm_gradients_match_torch():
    manual_resnorm = RMSNorm(2, eps=1e-5)
    torch_resnorm = nn.RMSNorm(2, eps=1e-5)

    with torch.no_grad():
        manual_resnorm.weight.copy_(torch.tensor([2.0, 0.5]))
        torch_resnorm.weight.copy_(torch.tensor([2.0, 0.5]))

    x_manual = torch.tensor([
        [3.0, 4.0], 
        [30.0, 40.0],
    ], requires_grad=True)
    x_torch = x_manual.detach().clone().requires_grad_(True)

    # 分别前向
    manual_output = manual_resnorm(x_manual)
    torch_output = torch_resnorm(x_torch)

    # 分别反向
    manual_output.sum().backward()
    torch_output.sum().backward()

    torch.testing.assert_close(manual_output, torch_output)
    torch.testing.assert_close(manual_resnorm.weight.grad, torch_resnorm.weight.grad)
    torch.testing.assert_close(x_manual.grad, x_torch.grad)

def test_rmsnorm_zero_input_is_finite():
    model = RMSNorm(3)
    x = torch.zeros(2, 4, 3)

    output = model(x)

    assert output.shape == (2, 4, 3)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(x))

def test_rmsnorm_preserves_float16_dtype():
    model = RMSNorm(2)
    x = torch.tensor([
        [
            [3.0, 4.0]
        ]
    ], dtype=torch.float16)

    output = model(x)
    assert output.dtype == torch.float16
    assert output.shape == (1, 1, 2)
    assert torch.isfinite(output).all()

    expected = model(x.float()).to(dtype=x.dtype)
    torch.testing.assert_close(output, expected)


