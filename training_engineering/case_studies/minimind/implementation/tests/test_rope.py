import pytest
import torch

from minimind_lab.models.rope import (
    apply_rope,
    build_rope_angles,
    rotate_features,
    rotate_features_with_half,
    rotate_half,
)


def test_build_rope_angles_matches_known_values():
    angles = build_rope_angles(
        seq_len=3,
        head_dim=4,
        rope_base=1e6,
        dtype=torch.float64,
    )

    expected = torch.tensor(
        [
            [0.0, 0.000],
            [1.0, 0.001],
            [2.0, 0.002],
        ],
        dtype=torch.float64,
    )

    assert angles.shape == (3, 2)
    assert angles.dtype == torch.float64
    torch.testing.assert_close(angles, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    ("seq_len", "head_dim", "rope_base"),
    [
        (0, 4, 1e6),
        (3, 0, 1e6),
        (3, 3, 1e6),
        (3, 4, 0.0),
    ],
)
def test_build_rope_angles_rejects_invalid_configuration(
    seq_len: int,
    head_dim: int,
    rope_base: float,
):
    with pytest.raises(ValueError):
        build_rope_angles(seq_len, head_dim, rope_base)


def test_rotate_half_matches_known_feature_layout():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    actual = rotate_half(x)
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0], dtype=torch.float64)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor(1.0),
        torch.empty(0),
        torch.empty(3),
    ],
)
def test_rotate_half_rejects_invalid_last_dimension(x: torch.Tensor):
    with pytest.raises(ValueError):
        rotate_half(x)


def test_direct_and_rotate_half_forms_match_forward_and_backward():
    torch.manual_seed(42)
    angles = build_rope_angles(
        seq_len=3,
        head_dim=4,
        dtype=torch.float64,
    )
    source = torch.randn(2, 4, 3, 4, dtype=torch.float64)
    direct_input = source.detach().clone().requires_grad_(True)
    half_input = source.detach().clone().requires_grad_(True)

    direct_output = rotate_features(direct_input, angles)
    half_output = rotate_features_with_half(half_input, angles)

    torch.testing.assert_close(
        half_output,
        direct_output,
        atol=1e-10,
        rtol=1e-10,
    )

    upstream_gradient = torch.randn_like(direct_output)
    (direct_output * upstream_gradient).sum().backward()
    (half_output * upstream_gradient).sum().backward()

    assert direct_input.grad is not None
    assert half_input.grad is not None
    torch.testing.assert_close(
        half_input.grad,
        direct_input.grad,
        atol=1e-10,
        rtol=1e-10,
    )


def test_apply_rope_preserves_bfloat16_dtype_and_gradients_with_fp32_angles():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 260, 4, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 1, 260, 4, dtype=torch.bfloat16, requires_grad=True)
    angles = build_rope_angles(260, 4, dtype=torch.float32)

    q_rotated, k_rotated = apply_rope(q, k, angles)

    assert q_rotated.dtype == q.dtype
    assert k_rotated.dtype == k.dtype
    assert angles[257, 0].item() == 257
    expected = torch.tensor([torch.cos(torch.tensor(257.0)), 0.0,
                             torch.sin(torch.tensor(257.0)), 0.0]).to(q.dtype)
    probe = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    probe = probe.expand(1, 1, 260, 4)
    probe_rotated, _ = apply_rope(probe, probe, angles)
    torch.testing.assert_close(probe_rotated[0, 0, 257], expected, rtol=0, atol=0)

    (q_rotated.float().square().mean() + k_rotated.float().square().mean()).backward()
    for tensor in (q, k):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_apply_rope_supports_different_query_and_key_head_counts():
    torch.manual_seed(42)
    q = torch.randn(2, 4, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    angles = build_rope_angles(
        seq_len=3,
        head_dim=4,
        dtype=torch.float64,
    )

    q_rotated, k_rotated = apply_rope(q, k, angles)

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape
    torch.testing.assert_close(q_rotated[:, :, 0, :], q[:, :, 0, :])
    torch.testing.assert_close(k_rotated[:, :, 0, :], k[:, :, 0, :])
    torch.testing.assert_close(
        q_rotated.square().sum(dim=-1),
        q.square().sum(dim=-1),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        k_rotated.square().sum(dim=-1),
        k.square().sum(dim=-1),
        atol=1e-10,
        rtol=1e-10,
    )

    (q_rotated.sum() + k_rotated.sum()).backward()

    assert q.grad is not None
    assert k.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "angles_shape"),
    [
        ((2, 4, 3), (2, 2, 3, 4), (3, 2)),
        ((2, 4, 3, 4), (3, 2, 3, 4), (3, 2)),
        ((2, 4, 3, 4), (2, 2, 4, 4), (3, 2)),
        ((2, 4, 3, 4), (2, 2, 3, 6), (3, 2)),
        ((2, 4, 3, 3), (2, 2, 3, 3), (3, 1)),
        ((2, 4, 3, 4), (2, 2, 3, 4), (3, 4)),
    ],
)
def test_apply_rope_rejects_invalid_contracts(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    angles_shape: tuple[int, ...],
):
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    angles = torch.randn(angles_shape)

    with pytest.raises(ValueError):
        apply_rope(q, k, angles)
