
import torch


def rotate_2d(
    x: torch.Tensor,
    angle: torch.Tensor,
):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    new_x0 = x[0]*cos_angle - x[1]*sin_angle
    new_x1 = x[0]*sin_angle + x[1]*cos_angle

    return torch.stack([new_x0, new_x1], dim=0)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    
    head_dim = x.shape[-1]

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head dimension must be a positive even integer")

    half = x.shape[-1] // 2
    a = x[...,: half]
    b = x[..., half:]

    return torch.cat([-b, a], dim=-1)


def rotate_features(
    x: torch.Tensor,
    angles: torch.Tensor,
)-> torch.Tensor:
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    new_a = a*cos_angles - b*sin_angles
    new_b = a*sin_angles + b*cos_angles

    return torch.cat([new_a, new_b], dim=-1)

def rotate_features_with_half(
    x: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    cos_full = torch.cat([cos_angles, cos_angles], dim=-1)
    sin_full = torch.cat([sin_angles, sin_angles], dim=-1)

    return (x*cos_full + rotate_half(x)*sin_full)


def build_rope_angles(
    seq_len:int,
    head_dim: int,
    rope_base: float=1e6,
    *,
    dtype:torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")

    if rope_base <= 0:
        raise ValueError("rope_base must be positive")

    positions = torch.arange(seq_len, dtype=dtype, device=device)
    position_columns = positions[:, None] # 增加一个空维度

    frequencies_indices = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    frequencies = 1.0 / (rope_base ** (frequencies_indices / head_dim))
    frequency_row = frequencies[None, :] # 增加一个空维度

    sequence_angles = position_columns * frequency_row
    return sequence_angles

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, H, T, d]")

    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same batch size")

    if q.shape[-2] != k.shape[-2]:
        raise ValueError("q and k must have the same sequence length")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")

    head_dim = q.shape[-1]

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head dimension must be a positive even integer")

    expected_angles_shape = (
        q.shape[-2],
        head_dim // 2,
    )

    if angles.shape != expected_angles_shape:
        raise ValueError(
            "angles must have shape "
            f"{expected_angles_shape}, got {tuple(angles.shape)}"
        )

    q_rotated = rotate_features(q, angles).to(dtype=q.dtype)
    k_rotated = rotate_features(k, angles).to(dtype=k.dtype)

    return q_rotated, k_rotated
