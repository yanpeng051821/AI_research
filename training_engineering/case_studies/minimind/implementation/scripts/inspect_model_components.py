
from minimind_lab.models.block import ResidualFeedForward
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.embedding import TokenEmbedding
from minimind_lab.models.initialization import initialize_weights

import torch

config = ModelConfig(20, 8, 16)

embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
block = ResidualFeedForward(config.hidden_size, config.intermediate_size, config.rms_norm_eps)

input_ids = torch.tensor([[1, 2, 3], [4, 2, 5]])
print(f"input_ids.shape: {input_ids.shape}")

# 初始化模块参数权重
embedding.apply(initialize_weights)
block.apply(initialize_weights)

# forward
embeded = embedding(input_ids)
print(f"embeded: {embeded}")
print(f"embeded.shape: {embeded.shape}")

output = block(embeded)
print(f"output: {output}")
print(f"output.shape: {output.shape}")  

loss = output.square().mean()
loss.backward()

for name, param in embedding.named_parameters():
    print(f"{name}: shape: {param.shape} counts: {param.numel()}")
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

total_embedding_params = sum(param.numel() for param in embedding.parameters())
print(f"total_embedding_params: {total_embedding_params}")

for name, param in block.named_parameters():
    print(f"{name}: shape: {param.shape} counts: {param.numel()}")
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

total_block_params = sum(param.numel() for param in block.parameters())
print(f"totaol_block_params: {total_block_params}")

expected_embedding = config.vocab_size * config.hidden_size
expected_block = (
    config.hidden_size
    + 3 * config.hidden_size * config.intermediate_size
)

assert total_embedding_params == expected_embedding
assert total_block_params == expected_block