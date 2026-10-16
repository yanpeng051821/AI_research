import pytest
import torch
from torch.nn import functional as F

from minimind_lab.models.attention import (
    GroupQueryAttention,
    scaled_dot_product_attention,
)


@pytest.mark.parametrize("causal", [True, False])
def test_padding_and_causal_masks_match_known_values(causal):
    q = torch.zeros((1, 3, 2), dtype=torch.float64)
    k = torch.zeros_like(q)
    v = torch.tensor([[[10.0], [20.0], [1000.0]]], dtype=q.dtype)
    mask = torch.tensor([[1, 1, 0]])

    output, weights = scaled_dot_product_attention(
        q, k, v, causal=causal, attention_mask=mask,
    )

    expected_weights = torch.tensor(
        [[[1.0, 0.0, 0.0] if causal else [0.5, 0.5, 0.0],
          [0.5, 0.5, 0.0],
          [0.5, 0.5, 0.0]]],
        dtype=q.dtype,
    )
    expected_output = torch.tensor(
        [[[10.0 if causal else 15.0], [15.0], [15.0]]],
        dtype=q.dtype,
    )
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("multihead", [True, False])
def test_padding_attention_matches_torch_forward_and_gradients(causal, multihead):
    torch.manual_seed(42)
    shape = (2, 3, 4, 2) if multihead else (2, 4, 2)
    q = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    k = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    v = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    actual, weights = scaled_dot_product_attention(
        q, k, v, causal=causal, attention_mask=mask,
    )

    # Explicit batch/head/query/key axes make this reference independent of unsqueeze.
    reference_mask = mask[:, None, None, :].bool()
    if causal:
        reference_mask = reference_mask & torch.ones((4, 4), dtype=torch.bool).tril()
    reference = F.scaled_dot_product_attention(
        rq if multihead else rq[:, None],
        rk if multihead else rk[:, None],
        rv if multihead else rv[:, None],
        attn_mask=reference_mask,
        dropout_p=0.0,
    )
    if not multihead:
        reference = reference[:, 0]

    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)),
    )
    torch.testing.assert_close(weights[1, ..., 2:], torch.zeros_like(weights[1, ..., 2:]))
    if causal:
        torch.testing.assert_close(weights.triu(1), torch.zeros_like(weights))

    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (reference * upstream).sum().backward()
    for actual_input, reference_input in [(q, rq), (k, rk), (v, rv)]:
        assert actual_input.grad is not None
        assert reference_input.grad is not None
        assert torch.isfinite(actual_input.grad).all()
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, atol=1e-10, rtol=1e-10,
        )


@pytest.mark.parametrize("causal", [True, False])
def test_gqa_padding_matches_unpadded_samples_and_ignores_pad_content(causal):
    torch.manual_seed(42)
    model = GroupQueryAttention(16, 4, 2).double()
    x = torch.randn((2, 4, 16), dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    output, weights = model(x, causal=causal, attention_mask=mask)

    assert output.shape == (2, 4, 16)
    assert weights.shape == (2, 4, 4, 4)
    for sample, length in enumerate([4, 2]):
        single_output, single_weights = model(x[sample:sample + 1, :length], causal=causal)
        torch.testing.assert_close(
            output[sample:sample + 1, :length], single_output,
            atol=1e-10, rtol=1e-10,
        )
        torch.testing.assert_close(
            weights[sample:sample + 1, :, :length, :length], single_weights,
            atol=1e-10, rtol=1e-10,
        )

    changed = x.detach().clone()
    changed[1, 2:] = torch.randn_like(changed[1, 2:]) * 1000
    changed_output, _ = model(changed, causal=causal, attention_mask=mask)
    torch.testing.assert_close(output[0], changed_output[0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(output[1, :2], changed_output[1, :2], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(weights[1, :, :, 2:], torch.zeros_like(weights[1, :, :, 2:]))

    # Only real query positions contribute to this loss.
    output[mask.bool()].square().sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad[1, 2:], torch.zeros_like(x.grad[1, 2:]))
    assert torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("mask_shape", [(4,), (2, 3)])
def test_padding_mask_rejects_wrong_shape(mask_shape):
    q = torch.zeros((2, 3, 4, 2))
    mask = torch.ones(mask_shape)

    with pytest.raises(ValueError, match="attention_mask must have shape"):
        scaled_dot_product_attention(q, q, q, attention_mask=mask)


@pytest.mark.parametrize("multihead", [False, True])
@pytest.mark.parametrize(
    "causal, mask",
    [
        (True, [[1, 1, 1, 1], [0, 0, 0, 0]]),
        (False, [[1, 1, 1, 1], [0, 0, 0, 0]]),
        (True, [[1, 1, 1, 1], [0, 1, 1, 0]]),
    ],
    ids=["empty-causal", "empty-noncausal", "left-padding-causal"],
)
def test_padding_attention_rejects_queries_without_readable_keys(multihead, causal, mask):
    shape = (2, 3, 4, 2) if multihead else (2, 4, 2)
    q = torch.zeros(shape, dtype=torch.float64)

    # Even one unreadable query row is outside the current right-padding contract.
    with pytest.raises(ValueError, match="at least one readable key"):
        scaled_dot_product_attention(
            q, q, q, causal=causal, attention_mask=torch.tensor(mask),
        )
