import torch

from torch import nn

embedding = nn.Embedding(6, 3)

input_ids = torch.tensor([
    [1, 2, 1], [3, 0, 2]
])

hidden = embedding(input_ids)

print(f"hidden: {hidden}")
print(f"hidden.shape: {hidden.shape}")

torch.testing.assert_close(hidden[0, 0], hidden[0, 2])


loss = hidden.sum()
loss.backward()

print(f"weight.grad: {embedding.weight.grad}")
print(f"weight.shape: {embedding.weight.shape}")

