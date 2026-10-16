
import torch

from torch import nn

parameter = nn.Parameter(torch.tensor([1.0, 2.0]))

coefficient = torch.tensor([3.0, 4.0])

loss = (parameter * coefficient).sum()
loss.backward()

print(f"before:grad={parameter.grad},norm={parameter.grad.norm()}")

norm_before = torch.nn.utils.clip_grad_norm_(
    [parameter],
    max_norm=1.0,
    error_if_nonfinite=True,
)
print(f"norm_before:{norm_before}")
print(f"after:grad={parameter.grad},norm={parameter.grad.norm()}")




