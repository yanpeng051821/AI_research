
import torch

from torch.utils.data import Sampler
from collections.abc import Iterator

class StatefulSampler(Sampler[int]):
    def __init__(self, size: int, seed: int=42) -> None:
        super().__init__()
        if size < 0:
            raise ValueError("sampler size must be at least 0")

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.indices = torch.randperm(size, generator=self.generator).tolist()

        self.position = 0
        self.committed_position = 0

    def __iter__(self)-> Iterator[int]:
        while self.position < len(self.indices):
            index = self.indices[self.position]
            self.position += 1
            yield index
    
    def commit(self, count: int):
        if count < 0:
            raise ValueError("commit count must be at least 0")

        if count + self.committed_position > self.position:
            raise ValueError("count + committed_position must be less than or equal to position")
        
        self.committed_position += count

    def state_dict(self) -> dict:
        return {
            "indices": self.indices.copy(),
            "committed_position": self.committed_position,
        }

    def load_state_dict(self, sampler_state: dict):
        indices = sampler_state["indices"].copy()
        committed_position = sampler_state["committed_position"]
        if len(indices) != len(self.indices):
            raise ValueError("indices length must match")

        if any(type(index) is not int for index in indices):
            raise ValueError("indices must be a list of integers")
        
        sorted_indices = sorted(indices)
        if sorted_indices != list(range(len(indices))):
            raise ValueError("indices must be a permutation of 0 to len(indices) - 1")

        if type(committed_position) is not int or committed_position < 0 or committed_position > len(indices):
            raise ValueError("committed_position must be an integer between 0 and len(indices)")


        self.indices = sampler_state["indices"].copy()
        self.committed_position = sampler_state["committed_position"]
        self.position = self.committed_position



