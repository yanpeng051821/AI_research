

import torch

from torch import nn


parameter = nn.Parameter(torch.tensor(2.0))

optimizer = torch.optim.SGD(
    [parameter],
    lr=0.1
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5,
)

used_lrs = []
next_lrs = []

for update_step in range(1, 5):
    optimizer.zero_grad(set_to_none=True)

    before = parameter.item()
    loss = parameter.square()
    loss.backward()

    lr_used = optimizer.param_groups[0]["lr"]
    used_lrs.append(lr_used)

    optimizer.step()

    after = parameter.item()

    scheduler.step()

    lr_next = optimizer.param_groups[0]["lr"]
    next_lrs.append(lr_next)

    print(
        f"update_step={update_step}, "
        f"before={before}, "
        f"grad={parameter.grad.item()}, "
        f"lr_used={lr_used}, "
        f"after={after}, "
        f"next_lr={lr_next}"
    )


expected_used_lrs = torch.tensor([
    0.1,
    0.1,
    0.05,
    0.05,
])

expected_next_lrs = torch.tensor([
    0.1,
    0.05,
    0.05,
    0.025,
])

torch.testing.assert_close(
    torch.tensor(used_lrs),
    expected_used_lrs,
)

torch.testing.assert_close(
    torch.tensor(next_lrs),
    expected_next_lrs,
)








