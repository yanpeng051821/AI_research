
import torch

q = torch.tensor([[1.0, 0.]])
k = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 1.],
])

v = torch.tensor([
    [10., 0.],
    [0., 20.],
    [30., 30.],
])

scores = (q @ k.T) / q.shape[-1] ** 0.5
print(f"score.shape:{scores.shape}")

weights = torch.softmax(scores, dim=-1)
print(f"weights.shape:{weights.shape}")

output = weights @ v
print(f"output.shape:{output.shape}")





