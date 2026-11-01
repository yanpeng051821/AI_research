
import torch

from torch import nn
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.embedding import TokenEmbedding
from minimind_lab.models.block import DecoderBlock
from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.initialization import initialize_weights

class MinimindBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size=config.hidden_size, 
                intermediate_size=config.intermediate_size, 
                num_heads=config.num_heads, 
                kv_heads=config.kv_heads,
                eps=config.rms_norm_eps,
                rope_base=config.rope_base,
                qk_norm=config.qk_norm,
                ) 
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, 
        input_ids: torch.Tensor,
        causal: bool = True,
        attention_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states, _ = block(hidden_states, causal, attention_mask)
        output = self.final_norm(hidden_states)
        return output


class MinimindForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = MinimindBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight
        self.apply(initialize_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.backbone(input_ids, causal, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits