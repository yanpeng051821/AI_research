
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 64
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5
    qk_norm: bool = False

    num_layers: int = 2
    num_heads: int = 4
    kv_heads: int = 2
    rope_base: float = 1e6
    
    

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")

    





