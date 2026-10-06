import torch

x = torch.tensor([-1.0, 2.0], requires_grad=True)
x_residual = torch.tensor([-1.0, 2.0], requires_grad=True)

with torch.no_grad():
    x_residual.copy_(x)

plain_output = 2 * x
residual_output = x_residual + 2 * x_residual

plain_output.sum().backward()
residual_output.sum().backward()

print(f"plain_output: {plain_output}")
print(f"residual_output: {residual_output}")
print(f"x.grad: {x.grad}")
print(f"x_residual.grad: {x_residual.grad}")





