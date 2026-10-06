import torch

from torch import nn

class ScalarLinear(nn.Module):
    def __init__(self, w: float = 1.0, b: float = 0.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
    
    def forward(self, x):
        return self.w * x + self.b

x = torch.tensor(2.0)
target = torch.tensor(5.0)

model = ScalarLinear()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()

prediction = model(x)
print(f"prediction: {prediction.item()}")

loss = (prediction - target) ** 2
loss.backward()
print(f"w.grad: {model.w.grad}, b.grad: {model.b.grad}")
print(f"更新前 w: {model.w.item()}, b: {model.b.item()}")

optimizer.step()
print(f"更新后 w: {model.w.item()}, b: {model.b.item()}")

with torch.no_grad():
    prediction = model(x)
    print(f"prediction: {prediction.item()}")
    loss = (prediction - target) ** 2
    print(f"loss: {loss.item()}")



# w = torch.tensor(1.0, requires_grad=True)
# b = torch.tensor(0.0, requires_grad=True)

# # 更新前的w和b
# print(f"w: {w}, b: {b}")

# optimizer = torch.optim.SGD([w, b], lr=0.1)
# optimizer.zero_grad()

# # 前向传播
# y = w * x + b
# loss = (y - target) ** 2

# # 尚未反向传播的梯度
# print(f"w.grad :{w.grad}, b.grad: {b.grad}")
# print(f"prediction: {y.item()}, loss: {loss.item()}")

# # 反向传播计算梯度
# loss.backward()

# # 反向传播后的梯度
# print(f"w.grad :{w.grad}, b.grad: {b.grad}")

# # 更新参数
# optimizer.step()

# # 更新后的w和b
# print(f"w: {w}, b: {b}")

# with torch.no_grad():
#     y = w * x + b
#     after_loss = (y - target) ** 2
#     print(f"prediction: {y.item()}")
#     print(f"loss: {after_loss.item()}")





