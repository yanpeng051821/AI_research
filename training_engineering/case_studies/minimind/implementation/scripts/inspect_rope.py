
import torch

from minimind_lab.models.rope import rotate_features, build_rope_angles, apply_rope, rotate_features_with_half, rotate_half

def rotate_2d(
    x: torch.Tensor,
    angle: torch.Tensor,
):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    new_x0 = x[0]*cos_angle - x[1]*sin_angle
    new_x1 = x[0]*sin_angle + x[1]*cos_angle

    return torch.stack([new_x0, new_x1], dim=0)


x = torch.tensor([3.0, 4.0], dtype=torch.float64)
angle_1 = torch.tensor(0.0, dtype=torch.float64)
angle_2 = torch.tensor(torch.pi /2 , dtype=torch.float64)

rotated_1 = rotate_2d(x, angle_1)
rotated_2 = rotate_2d(x, angle_2)

assert rotated_1.shape == (2, )
assert rotated_2.shape == (2, )

torch.testing.assert_close(rotated_1, torch.tensor([3.0,4.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)
torch.testing.assert_close(rotated_2, torch.tensor([-4.0,3.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)

torch.testing.assert_close(rotated_1.square().sum(), x.square().sum(), atol=1e-10, rtol=1e-10)
torch.testing.assert_close(rotated_2.square().sum(), x.square().sum(), atol=1e-10, rtol=1e-10)



q = torch.tensor([1.0, 2.0], dtype=torch.float64)
k = torch.tensor([3.0, 4.0], dtype=torch.float64)
omega = torch.tensor(0.5, dtype=torch.float64)

def position_score(m: int, n:int):
    q_rotated = rotate_2d(q, m*omega)
    k_rotated = rotate_2d(k, n*omega)

    return torch.dot(q_rotated, k_rotated)


score_a = position_score(2, 5)
score_b = position_score(9, 12)
score_c = position_score(2, 6)

torch.testing.assert_close(
    score_a, score_b, atol=1e-10, rtol=0
)
assert not torch.allclose(score_a, score_c)


features = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
angles = torch.tensor([0.0, torch.pi / 2], dtype=torch.float64)

rotated = rotate_features(features, angles)

torch.testing.assert_close(
    rotated,
    torch.tensor([1.0, -4.0, 3.0, 2.0], dtype=torch.float64),
    atol=1e-10,
    rtol=0,
)


### 验证位置和频率
head_dim = 4
rope_base = 1e6
position = 3

feature_indices = torch.arange(0, head_dim, 2, dtype=torch.float64)
# [0, 2]

frequencies = 1.0 / (
    rope_base ** (feature_indices / head_dim)
)

position_angles = position * frequencies

torch.testing.assert_close(
    frequencies,
    torch.tensor([1.0, 0.001], dtype=torch.float64),
    atol=1e-12,
    rtol=0,
)

torch.testing.assert_close(
    position_angles,
    torch.tensor([3.0, 0.003], dtype=torch.float64),
    atol=1e-12,
    rtol=0,
)


position_rotated = rotate_features(features, position_angles)

assert position_rotated.shape == features.shape

torch.testing.assert_close(
    position_rotated.square().sum(),
    features.square().sum(),
    atol=1e-10,
    rtol=1e-10,
)


seq_len = 3
positions = torch.arange(
    seq_len,
    dtype=torch.float64,
)

position_column = positions[:, None]
frequency_row = frequencies[None, :]

sequence_angles = position_column * frequency_row

assert sequence_angles.shape == (3, 2)

torch.testing.assert_close(
    sequence_angles,
    torch.tensor(
        [
            [0.0, 0.000],
            [1.0, 0.001],
            [2.0, 0.002],
        ],
        dtype=torch.float64,
    ),
    atol=1e-12,
    rtol=0,
)

sequence_features = features.repeat(seq_len, 1)
sequence_rotated = rotate_features(
    sequence_features,
    sequence_angles,
)
assert sequence_rotated.shape == sequence_features.shape

torch.testing.assert_close(
    sequence_rotated[0],
    sequence_features[0],
    atol=1e-10,
    rtol=1e-10,
)

torch.testing.assert_close(
    sequence_rotated.square().sum(dim=-1),
    sequence_features.square().sum(dim=-1),
    atol=1e-10,
    rtol=1e-10,
)


generated_angles = build_rope_angles(
    seq_len=seq_len,
    head_dim=head_dim,
    rope_base=rope_base,
    dtype=torch.float64,
)

torch.testing.assert_close(
    generated_angles,
    sequence_angles,
    atol=1e-12,
    rtol=0,
)


batch_size = 2
num_heads = 3
seq_len = 3
head_dim = 4

q = torch.arange(
    batch_size * num_heads * seq_len * head_dim,
    dtype=torch.float64,
).reshape(
    batch_size,
    num_heads,
    seq_len,
    head_dim,
)

q.requires_grad_(True)

q_angles = build_rope_angles(
    seq_len=seq_len,
    head_dim=head_dim,
    rope_base=rope_base,
    dtype=q.dtype,
    device=q.device,
)

q_rotated = rotate_features(q, q_angles)

assert q_angles.shape == (seq_len, head_dim // 2)
assert q_rotated.shape == q.shape


torch.testing.assert_close(
    q_rotated.square().sum(dim=-1),
    q.square().sum(dim=-1),
    atol=1e-9,
    rtol=1e-10,
)


q_rotated.sum().backward()

assert q.grad is not None
assert torch.isfinite(q.grad).all()

torch.testing.assert_close(
    q_rotated[:, :, 0, :],
    q[:, :, 0, :],
    atol=1e-10,
    rtol=1e-10,
)



torch.manual_seed(42)

gqa_q = torch.randn(
    2, 4, seq_len, head_dim,
    dtype=torch.float64,
    requires_grad=True,
)

gqa_k = torch.randn(
    2, 2, seq_len, head_dim,
    dtype=torch.float64,
    requires_grad=True,
)

gqa_q_rotated, gqa_k_rotated = apply_rope(
    gqa_q,
    gqa_k,
    generated_angles,
)

assert gqa_q_rotated.shape == gqa_q.shape
assert gqa_k_rotated.shape == gqa_k.shape

torch.testing.assert_close(
    gqa_q_rotated[:, :, 0, :],
    gqa_q[:, :, 0, :],
    atol=1e-10,
    rtol=1e-10,
)

torch.testing.assert_close(
    gqa_k_rotated[:, :, 0, :],
    gqa_k[:, :, 0, :],
    atol=1e-10,
    rtol=1e-10,
)

torch.testing.assert_close(
    gqa_q_rotated.square().sum(dim=-1),
    gqa_q.square().sum(dim=-1),
    atol=1e-10,
    rtol=1e-10,
)

torch.testing.assert_close(
    gqa_k_rotated.square().sum(dim=-1),
    gqa_k.square().sum(dim=-1),
    atol=1e-10,
    rtol=1e-10,
)


loss = gqa_q_rotated.sum() + gqa_k_rotated.sum()
loss.backward()

assert gqa_q.grad is not None
assert gqa_k.grad is not None
assert torch.isfinite(gqa_q.grad).all()
assert torch.isfinite(gqa_k.grad).all()

bad_angles = torch.zeros(
    seq_len,
    head_dim,
    dtype=torch.float64,
)

try:
    apply_rope(gqa_q, gqa_k, bad_angles)
except ValueError:
    pass
else:
    raise AssertionError("invalid angle shape was not rejected")


half_rotated = rotate_half(
    torch.tensor(
        [1.0, 2.0, 3.0, 4.0],
        dtype=torch.float64,
    )
)

torch.testing.assert_close(
    half_rotated,
    torch.tensor(
        [-3.0, -4.0, 1.0, 2.0],
        dtype=torch.float64,
    ),
)

q_direct = rotate_features(
    gqa_q,
    generated_angles,
)

q_with_half = rotate_features_with_half(
    gqa_q,
    generated_angles,
)

torch.testing.assert_close(
    q_with_half,
    q_direct,
    atol=1e-10,
    rtol=1e-10,
)


gradient_source = torch.randn(
    2,
    4,
    seq_len,
    head_dim,
    dtype=torch.float64,
)

direct_input = (
    gradient_source
    .detach()
    .clone()
    .requires_grad_(True)
)

half_input = (
    gradient_source
    .detach()
    .clone()
    .requires_grad_(True)
)



direct_output = rotate_features(
    direct_input,
    generated_angles,
)

half_output = rotate_features_with_half(
    half_input,
    generated_angles,
)

torch.testing.assert_close(
    half_output,
    direct_output,
    atol=1e-10,
    rtol=1e-10,
)

upstream_gradient = torch.randn_like(
    direct_output
)

direct_loss = (
    direct_output * upstream_gradient
).sum()

half_loss = (
    half_output * upstream_gradient
).sum()


direct_loss.backward()
half_loss.backward()

assert direct_input.grad is not None
assert half_input.grad is not None

torch.testing.assert_close(
    half_input.grad,
    direct_input.grad,
    atol=1e-10,
    rtol=1e-10,
)