
import torch

from minimind_lab.models.attention import scaled_dot_product_attention

B, T, H, d = 1, 3, 2, 2

q_flat = torch.arange(12, dtype=torch.float32).reshape(B, T, H*d)
print(f"q_flat.shape: {q_flat.shape}")

# 拆头
q_split = q_flat.reshape(B, T, H, d)
print(f"q_split.shape:{q_split.shape}")

# 交换位置轴与头轴
q_heads = q_split.transpose(1, 2)
print(f"q_heads.shape:{q_heads.shape}")

# 实际数据打印与推演预算
print(f"q_flat: {q_flat}")
print(f"q_split:{q_split}")
print(f"head 0: {q_heads[0, 0]}")
print(f"head 1: {q_heads[0, 1]}")

q_inferred = q_flat.reshape(B, T, -1, d)
torch.testing.assert_close(q_inferred, q_split)

# 逆向换头
q_by_position = q_heads.transpose(1, 2)
q_merged = q_by_position.reshape(B, T, H*d)
torch.testing.assert_close(q_flat, q_merged)

# bad case
wrong_merged = q_heads.reshape(B, T, H*d)
assert not torch.allclose(q_flat, wrong_merged)

# 验证分别计算头的注意力与同时计算两个头的注意力的效果是一致的
k_heads = q_heads.clone()
v_heads = q_heads + 10.0
heads_output, heads_weights = scaled_dot_product_attention(q_heads, k_heads, v_heads)

assert heads_output.shape == (B, H, T, d)
assert heads_weights.shape == (B, H, T, T)

# 遍历两个头
for h in range(H):
    single_output, single_weights = scaled_dot_product_attention(q_heads[:,h], k_heads[:, h], v_heads[:, h])
    torch.testing.assert_close(single_output, heads_output[:, h]) 
    torch.testing.assert_close(single_weights, heads_weights[:, h]) 

# 合并多头
merged_output = heads_output.transpose(1, 2).reshape(B, T, H*d)
assert merged_output.shape == (B, T, H*d)

expected_merged = torch.cat(
    [heads_output[:, h] for h in range(H)],
    dim=-1
)

torch.testing.assert_close(merged_output, expected_merged)

