import pytest
import torch

from minimind_lab.models.block import DecoderBlock


@pytest.mark.parametrize("causal", [True, False])
def test_decoder_forward_backward(causal):
    torch.manual_seed(42)
    block = DecoderBlock(16, 32, 4, 2).double()
    x = torch.randn((2, 4, 16), dtype=torch.float64, requires_grad=True)

    output, weights = block(x, causal=causal)

    assert output.shape == x.shape
    assert weights.shape == (2, 4, 4, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert block.attention_norm.weight is not block.feed_forward.rms_norm.weight

    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("zeroed_branch", ["both", "attention", "feed_forward"])
def test_decoder_residuals_preserve_branch_inputs(zeroed_branch):
    torch.manual_seed(42)
    block = DecoderBlock(16, 32, 4, 2).double()
    x = torch.randn((2, 4, 16), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        if zeroed_branch in ("both", "attention"):
            block.attention.o_proj.weight.zero_()
        if zeroed_branch in ("both", "feed_forward"):
            block.feed_forward.swiglu.down_proj.weight.zero_()

    output, _ = block(x)
    if zeroed_branch == "both":
        expected = x
    elif zeroed_branch == "attention":
        expected = block.feed_forward(x)
    else:
        attention_output, _ = block.attention(block.attention_norm(x))
        expected = x + attention_output
    torch.testing.assert_close(output, expected, atol=1e-10, rtol=1e-10)

    if zeroed_branch == "both":
        # With both updates zeroed, the remaining identity path has unit derivative.
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_decoder_causal_output_ignores_future_input():
    torch.manual_seed(42)
    block = DecoderBlock(16, 32, 4, 2).double()
    x = torch.randn((2, 4, 16), dtype=torch.float64)
    changed = x.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 100

    original_output, _ = block(x)
    changed_output, _ = block(changed)

    torch.testing.assert_close(
        original_output[:, :2], changed_output[:, :2], atol=1e-10, rtol=1e-10,
    )


@pytest.mark.parametrize("causal", [True, False])
def test_decoder_padding_matches_unpadded_samples(causal):
    torch.manual_seed(42)
    block = DecoderBlock(16, 32, 4, 2).double()
    x = torch.randn((2, 4, 16), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    output, weights = block(x, causal=causal, attention_mask=mask)
    for sample, length in enumerate([4, 2]):
        single_output, single_weights = block(x[sample:sample + 1, :length], causal=causal)
        torch.testing.assert_close(
            output[sample:sample + 1, :length], single_output,
            atol=1e-10, rtol=1e-10,
        )
        torch.testing.assert_close(
            weights[sample:sample + 1, :, :length, :length], single_weights,
            atol=1e-10, rtol=1e-10,
        )

    changed = x.detach().clone()
    changed[1, 2:] = torch.randn_like(changed[1, 2:]) * 100
    changed_output, _ = block(changed, causal=causal, attention_mask=mask)
    torch.testing.assert_close(
        output[mask.bool()], changed_output[mask.bool()], atol=1e-10, rtol=1e-10,
    )
    torch.testing.assert_close(weights[1, ..., 2:], torch.zeros_like(weights[1, ..., 2:]))

    output[mask.bool()].square().mean().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad[1, 2:], torch.zeros_like(x.grad[1, 2:]))
