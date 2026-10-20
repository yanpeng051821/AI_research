
import torch
from torch.utils.data import DataLoader

from minimind_lab.data.sampler import StatefulSampler

dataset = torch.arange(5)
sampler = StatefulSampler(size=len(dataset), seed=42)

loader_generator = torch.Generator()
loader_generator.manual_seed(123)

loader  = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler,
    num_workers=0,
    drop_last=False,
    generator=loader_generator,
)

print("计划顺序:", sampler.indices)


for batch in loader:
    print(f"batch:",batch.tolist())
    print(f"sampler.position: {sampler.position}")
    print(f"sampler.committed_position: {sampler.committed_position}")





