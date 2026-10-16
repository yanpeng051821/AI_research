
import torch

q = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 1.],
])


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

print("scores:", scores)
print("weights:", weights)
print("output:", output)

torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
torch.testing.assert_close(output[0], torch.tensor([16.0445, 15.9889]))


# 因果掩码
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
output = weights @ v

print(f"allowed: {allowed}, allowed.shapo:{allowed.shape}")
print(f"masked_scores: {masked_scores}, masked_scores.shape:{masked_scores.shape}")
print(f"weights: {weights}, weights.shape:{weights.shape}")
print(f"output: {output}, output.shape:{output.shape}")

torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
assert torch.count_nonzero(weights[~allowed]).item() == 0
torch.testing.assert_close(output[0], v[0])


# 未来扰动实验
original_output = output.clone()
k_changed = k.clone()
v_changed = v.clone()

k_changed[2] = torch.tensor([100., -100.])
v_changed[2] = torch.tensor([-500., 700.])
changed_scores = ( q @ k_changed.T ) / q.shape[-1] ** 0.5
masked_changed_scores = changed_scores.masked_fill(~allowed, float("-inf"))
changed_weights = torch.softmax(masked_changed_scores, dim=-1)
changed_output = changed_weights @ v_changed

torch.testing.assert_close(
    changed_output[:2], original_output[:2]
)

assert not torch.allclose(
    changed_output[2], original_output[2]
)









