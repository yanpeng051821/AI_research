
import torch

from torch import nn

parameter = nn.Parameter(torch.tensor(2.0))
reference_param = nn.Parameter(torch.tensor(2.0))

optimizer = torch.optim.SGD([parameter], lr=0.1)
reference_optimizer = torch.optim.SGD([reference_param], lr=0.1)

targets = [0.0, 1.0]

optimizer.zero_grad(set_to_none=True)
for target in targets:
    loss = (parameter - target).square()
    loss.backward()

    print(f"parameter: {parameter.item()}, accumulated grad: {parameter.grad.item()}")

# 归一化前的累积梯度
print(f"accumulated grad before div: {parameter.grad.item()}")

# 计算两个样本的平均梯度
parameter.grad.div_(len(targets))
print(f"归一化后的梯度:{parameter.grad.item()}")

# 更新参数
optimizer.step()
print(f"updated parameter: {parameter.item()}\n\n")



### 参考路径
print("### 参考路径")
reference_optimizer.zero_grad(set_to_none=True)
loss_sum = sum(
    (reference_param - target).square()
    for target in targets
)

loss_avg = loss_sum / len(targets)
loss_avg.backward()
print(f"parameter: {reference_param.item()}, accumulated grad: {reference_param.grad.item()}")

reference_optimizer.step()
print(f"reference updated parameter: {reference_param.item()}")

torch.testing.assert_close(
    parameter.grad,
    reference_param.grad,
)

torch.testing.assert_close(
    parameter.detach(),
    reference_param.detach(),
)







