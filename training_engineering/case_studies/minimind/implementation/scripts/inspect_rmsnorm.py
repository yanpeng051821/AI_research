
import torch

x_original = torch.tensor([
    [3.0, 4.0],
    [30.0, 40.0],
])

x = x_original.pow(2).mean(dim=-1, keepdim=True)

sqrt_x = torch.sqrt(x + 1e-5)
scale_x = x_original / sqrt_x 

print(f"x: {x_original}")
print(f"sqrt_x: {sqrt_x}")
print(f"scale_x: {scale_x}")
