
import torch
from torch import nn




base_lr = 0.1
warm_ups = 2
total_steps = 6

def warmup_decay_multiplier(s: int) -> float:
    if s < warm_ups:
        return ( s + 1 ) / warm_ups
    
    return max(0.0, (total_steps - s) / (total_steps - warm_ups + 1))

for s in range(total_steps + 1):
    multiplier = warmup_decay_multiplier(s)
    lr = base_lr * multiplier
    print(f"s={s}, multiplier={multiplier:.2f}, lr={lr:.2f}")


parameter = nn.Parameter(torch.tensor(2.0))
optimizer = torch.optim.SGD(
    [parameter],
    lr=base_lr
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=warmup_decay_multiplier
)

print(
    f"after_init: index={scheduler.last_epoch}, "
    f"lr={optimizer.param_groups[0]['lr']}"
)

used_lrs = []
next_lrs = []

for step in range(1, total_steps + 1):
    print(f"step={step}")
    optimizer.zero_grad(set_to_none=True)

    before = parameter.item()
    print(f"before={before}")

    loss = parameter.square()
    loss.backward()

    lr_used = optimizer.param_groups[0]["lr"]
    used_lrs.append(lr_used)
    print(f"lr_used={lr_used}")
    optimizer.step()

    after = parameter.item()
    print(f"after={after}")

    scheduler.step()
    lr_next = optimizer.param_groups[0]["lr"]
    next_lrs.append(lr_next)
    print(f"lr_next={lr_next}")

expected_used_lrs = torch.tensor([
    0.05, 0.10, 0.08, 0.06, 0.04, 0.02,
])

expected_next_lrs = torch.tensor([
    0.10, 0.08, 0.06, 0.04, 0.02, 0.00,
])

torch.testing.assert_close(expected_used_lrs, torch.tensor(used_lrs))
torch.testing.assert_close(expected_next_lrs, torch.tensor(next_lrs))


print(f"optimizer.param_groups:{optimizer.param_groups}")



