from torch import nn

from minimind_lab.models.feedforward import SwiGLU
from minimind_lab.models.rmsnorm import RMSNorm


import torch

class ResidualFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps:float = 1e-5
    ):
        super().__init__()
        self.swiglu = SwiGLU(hidden_size, intermediate_size)
        self.rms_norm = RMSNorm(hidden_size, eps)
        

    def forward(self, x: torch.Tensor):
        return self.swiglu(self.rms_norm(x)) + x



