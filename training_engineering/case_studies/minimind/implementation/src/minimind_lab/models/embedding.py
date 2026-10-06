import torch

from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(vocab_size, embedding_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]











