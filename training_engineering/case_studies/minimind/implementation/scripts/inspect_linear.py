import torch

from torch import nn

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 7.0],
])

weight = torch.tensor([
    [1.0, 2.0, 0.0],
    [0.0, -1.0, 1.0],
])

bias = torch.tensor([0.5, -0.5])

manual_output = x @ weight.T + bias


layer = nn.Linear(3, 2)

with torch.no_grad():
    layer.weight.copy_(weight)
    layer.bias.copy_(bias)
    
layer_output = layer(x)

torch.testing.assert_close(manual_output, layer_output)

probe = torch.tensor([-1.0, 0.0, 1.0])

### 多层线性层可叠加成一层
# 第一层
hidden = 2 * probe + 1

# 第二层
linear_output = 3 * hidden - 2

# 直接计算
output_manual = 6 * probe + 1

assert torch.equal(output_manual, linear_output)

### 加入非线性层后，无法叠加
non_linear_output = 3 * torch.relu(hidden) - 2
assert not torch.equal(linear_output, non_linear_output)








