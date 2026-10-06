from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.embedding import TokenEmbedding

from torch import nn

def initialize_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias)
        
    elif isinstance(module, TokenEmbedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
    