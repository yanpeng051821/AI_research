
import torch

from minimind_lab.models.attention import scaled_dot_product_attention

q = torch.zeros(1, 3, 2, dtype=torch.float64)
k = torch.zeros(1, 3, 2, dtype=torch.float64)

v = torch.tensor([[
    [10.0],
    [20.0],
    [1000.0]
]], dtype=torch.float64)

attention_mask = torch.tensor([[1, 1, 0]])

unmasked_output, unmasked_weights = scaled_dot_product_attention(q, k, v, causal=False)

masked_output, masked_weights = scaled_dot_product_attention(
    q, k, v,
    causal=False,
    attention_mask=attention_mask,
)

print("unmasked weights:", unmasked_weights)
print("unmasked output:", unmasked_output)
print("masked weights:", masked_weights)
print("masked output:", masked_output)

torch.testing.assert_close(
    masked_output,
    torch.full_like(masked_output, 15.0),
)

torch.testing.assert_close(
    masked_weights[..., 2],
    torch.zeros_like(masked_weights[..., 2]),
)


combined_output, combined_weights = scaled_dot_product_attention(
    q, k, v,
    attention_mask=attention_mask,
    causal=True
)

print("combined weights:", combined_weights)
print("combined output:", combined_output)

torch.testing.assert_close(
    combined_weights,
    torch.tensor(
        [[[1.0, 0.0, 0.0],
          [0.5, 0.5, 0.0],
          [0.5, 0.5, 0.0]]],
        dtype=torch.float64,
    ),
)

torch.testing.assert_close(
    combined_output,
    torch.tensor(
        [[[10.0], [15.0], [15.0]]],
        dtype=torch.float64,
    ),
)