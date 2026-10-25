
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM

config = ModelConfig(
    vocab_size=6400,
    hidden_size=768,
    intermediate_size=2432,
    num_layers=8,
    num_heads=8,
    kv_heads=4,
    rms_norm_eps=1e-6,
    rope_base=1e6,
    qk_norm=True,
)

model = MinimindForCausalLM(config)

assert len(model.backbone.blocks) == 8
assert model.backbone.blocks[0].attention.head_dim == 96
assert sum(p.numel() for p in model.parameters()) == 63912192
assert model.lm_head.weight is model.backbone.embedding.weight


weight = model.backbone.embedding.weight.detach()
print(
    f"embedding: mean={weight.mean().item():.8f}, "
    f"std={weight.std().item():.8f}"
)









