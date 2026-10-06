import torch

from minimind_lab.models.block import ResidualFeedForward

def test_residual_feed_forward_matches_identity():
    block = ResidualFeedForward(4, 8)
    x = torch.rand(2, 3, 4, requires_grad=True)

    with torch.no_grad():
        block.swiglu.down_proj.weight.copy_(torch.zeros_like(block.swiglu.down_proj.weight))

    output = block(x)
    torch.testing.assert_close(output, x)

    # 验证反向
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_residual_feed_forward_matches_composition():
    block = ResidualFeedForward(4, 8)
    x = torch.rand(2, 3, 4, requires_grad=True)

    # 分步计算
    before = x.detach().clone()
    normalized = block.rms_norm(before)
    delta = block.swiglu(normalized)
    expected = before + delta

    output = block(x)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x, before)









