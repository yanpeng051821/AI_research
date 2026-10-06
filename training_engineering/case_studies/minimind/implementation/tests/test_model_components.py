from minimind_lab.models.config import ModelConfig
from minimind_lab.models.embedding import TokenEmbedding
from minimind_lab.models.block import ResidualFeedForward
from minimind_lab.models.initialization import initialize_weights

import torch

def test_components_forward_backward():
    config = ModelConfig(vocab_size=8, hidden_size=8, intermediate_size=16)
    embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
    block = ResidualFeedForward(config.hidden_size, config.intermediate_size, config.rms_norm_eps)

    embedding.apply(initialize_weights)
    block.apply(initialize_weights)

    x = torch.tensor([[1,2,3],[4,2,5]])

    embeded = embedding(x)
    output = block(embeded)

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()

    # 反向传播
    loss = output.square().mean()
    loss.backward()

    embed_param_counts = 0
    block_param_counts = 0

    for param in embedding.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        embed_param_counts += param.numel()

    for param in block.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        block_param_counts += param.numel()
    
    assert embed_param_counts == config.vocab_size * config.hidden_size
    assert block_param_counts == (
        config.hidden_size
        + 3 * config.hidden_size * config.intermediate_size
    )





