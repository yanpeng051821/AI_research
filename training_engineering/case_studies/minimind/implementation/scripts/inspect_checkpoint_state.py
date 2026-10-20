
import torch

from torch import nn

model = torch.nn.Linear(2, 1, bias=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=1, 
    gamma=0.5,
)

print(f"model state: {model.state_dict()}")
print(f"optimizer state: {optimizer.state_dict()}")

input_ids = torch.tensor([[1.0, 2.0]])
loss = model(input_ids).square().mean()
loss.backward()

optimizer.step()
scheduler.step()

print(f"optimizer after:{optimizer.state_dict()}")
print(f"scheduler after:{scheduler.state_dict()}")










