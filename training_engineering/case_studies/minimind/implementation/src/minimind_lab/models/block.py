from torch import nn

from minimind_lab.models.feedforward import SwiGLU
from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.attention import GroupQueryAttention


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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        kv_heads: int,
        eps: float = 1e-5,
        rope_base : float = 1e6,
        ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps)
        self.attention = GroupQueryAttention(hidden_size, num_heads, kv_heads, rope_base)
        self.feed_forward = ResidualFeedForward(hidden_size, intermediate_size, eps)

    def forward(
        self,
        x:torch.Tensor,
        causal:bool=True,
        attention_mask: torch.Tensor | None = None,
    )-> tuple[torch.Tensor, torch.Tensor]:
        x_normed = self.attention_norm(x)
        x_atten, weights = self.attention(x_normed, causal, attention_mask)
        x_ff_in = x + x_atten
        output = self.feed_forward(x_ff_in)
        return output, weights


