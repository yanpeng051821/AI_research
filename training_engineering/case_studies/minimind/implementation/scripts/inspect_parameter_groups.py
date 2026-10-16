
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM

config = ModelConfig(
    vocab_size=20,
    hidden_size=16,
    intermediate_size=32,
    num_layers=2,
    num_heads=4,
    kv_heads=2,
)

model = MinimindForCausalLM(config)

decay_names = []
decay_params = []

no_decay_names = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    
    if param.ndim >= 2:
        decay_names.append(name)
        decay_params.append(param)
    else:
        no_decay_names.append(name)
        no_decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.02},
        {"params": no_decay_params, "weight_decay": 0.0}
    ],
    lr=1e-3
)

print("decay names:")
for name in decay_names:
    print("  ", name)

print("no decay names:")
for name in no_decay_names:
    print("  ", name)

for index, group in enumerate(optimizer.param_groups):
    print(
        f"group={index}, "
        f"weight_decay={group['weight_decay']}, "
        f"parameter_count={len(group['params'])}, "
        f"numel={sum(parameter.numel() for parameter in group['params'])}"
    )


trainable_ids = {
    id(parameter)
    for parameter in model.parameters()
    if parameter.requires_grad
}
decay_ids = {id(parameter) for parameter in decay_params}
no_decay_ids = {id(parameter) for parameter in no_decay_params}
# 组里都没有重复参数
assert len(decay_ids) == len(decay_params)
assert len(no_decay_ids) == len(no_decay_params)

# 两个组之间没有交叉
assert decay_ids.isdisjoint(no_decay_ids)

# 两组参数身份的并集等于全部可训练参数，没有遗漏或混入
assert decay_ids | no_decay_ids == trainable_ids
assert len(decay_ids) + len(no_decay_ids) == len(trainable_ids)


before = {
    name: param.detach().clone() 
    for name, param in model.named_parameters()
    }
optimizer.zero_grad(set_to_none=True)

zero_loss = sum(
    (parameter * 0.0).sum()
    for parameter in model.parameters()
)

zero_loss.backward()

# 预期每个参数都是：grad max = 0
for name, param in model.named_parameters():
    assert param.grad is not None
    print(
        f"{name}: "
        f"grad_max={param.grad.abs().max().item()}"
    )

optimizer.step()
for name, param in model.named_parameters():
    max_change = (
        param.detach() -before[name]
    ).abs().max().item()

    group_name = (
        "decay"
        if id(param) in decay_ids
        else "no_decay"
    )

    print(
        f"{group_name}: "
        f"{name}, "
        f"max_change={max_change}"
    )

    if id(param) in decay_ids:
        expected = before[name] * (1.0 - 1e-3 * 0.02)
        torch.testing.assert_close(
            param.detach(),
            expected,
            rtol=1e-5,
            atol=1e-7,
        )
    else:
        torch.testing.assert_close(
            param.detach(),
            before[name],
        )









