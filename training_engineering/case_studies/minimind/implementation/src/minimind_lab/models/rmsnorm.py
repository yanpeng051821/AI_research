import torch

from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        scale_x = x_float / torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = self.weight * scale_x
        return output.to(dtype=x.dtype)


