
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 64
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

    





