# 第 5 周：单设备训练系统实践教程

> 教程重写：2026-10-19。本周把第四周的短循环整理成一次更新和整轮训练两个接口；QA 在文末。

## 1. 本周任务与准备

第四周已经能前向、计算语言模型 loss、反向并更新参数。本周不是再写模型，而是弄清一次更新由什么决定，以及怎样把多个小 batch 正确组织成一轮训练。

~~~text
micro-batch → model → loss_sum → backward
    下一批 → model → loss_sum → backward（参数尚未更新）
                                  ↓
                         按本窗口有效 token 数归一化
                                  ↓
                           梯度裁剪 → optimizer.step
                                  ↓
                         scheduler.step → 更新记录
~~~

### 1.1 环境与接口

命令在 implementation 目录执行，PowerShell，先 uv sync --locked。默认 CPU，小模型与人工 token 即可，不依赖服务器或大语料。

复用第四周的 MinimindForCausalLM 和 causal_lm_loss_sum_and_count。当前 loss.py 已把均值拆成 sum/count；若从早期副本开始，按第四周教程的完整接口补齐。train_update 期望每个 batch 包含 input_ids、attention_mask、labels，且每批 shift 后至少有一个有效目标。模型与张量必须已放在同一设备。

训练核心在自己的练习副本中实现，现有 tests 可用来核对。观察脚本说明“为什么”，正式测试保护后续修改，两者不必逐行重复。


**分步测试约定**：下面的 pytest 命令用于已有完整仓库的复练核验。从空文件逐步实现时，先执行本步给出的手算或最小断言；现有测试文件可能在收集时就导入后续类，-k 不会跳过这些导入。相关模块完成后再运行对应测试文件，不为通过尚未实现的后期测试提前增加功能。

### 1.2 实践顺序

| 步骤 | 本步产物 | 判断标准 |
| --- | --- | --- |
| 2–4. 梯度、AdamW、衰减 | inspect_optimizer.py | 手算和实际更新对齐 |
| 5. 参数组 | inspect_parameter_groups.py | 无遗漏、无重复，衰减按组生效 |
| 6. 调度 | 两个学习率观察脚本 | 当前与下一学习率区分清楚 |
| 7–8. 梯度累积 | 标量和 token 两个对照 | 与同目标的大 batch 一致 |
| 9. 裁剪 | inspect_gradient_clipping.py | 梯度被统一缩放 |
| 10. 单次更新 | training/step.py | 计算、更新、指标一致 |
| 11. 整轮循环 | training/loop.py | 完整与尾部窗口都执行 |
| 12. 验收 | tests/test_training.py | 参数、时序和边界受保护 |

完成标准是数值与行为闭环，不是 loss 必须下降到某个固定值。验证、checkpoint 和样本恢复留到第六周。

## 2. 梯度与更新对象

### 2.1 从 Loss 到导数

L(w)=w²，w=2 时 loss=4，梯度 dL/dw=2w=4，这次只是数值碰巧相等。w=3 时两者是 9 和 6。梯度给出局部变化率：ΔL≈gradient×Δw，不能由一个 loss 数值倒推出梯度。

前向记录运算关系，反向用链式法则传回参数。例如 y=2w、L=(y-1)²，dL/dw=2(y-1)×2。共享路径的贡献相加，多次 backward 默认也累加到 .grad。

### 2.2 参数范围

标量的梯度是一个数；矩阵参数的梯度与参数同形状，保存各元素的一阶偏导；全部参数的这些偏导共同构成模型梯度。“某个参数的梯度”与“完整梯度向量”只是范围不同。

optimizer 持有参数对象集合。即使只有一个 Parameter，也传 [parameter]；不是把参数数值交进去，也不是每次 step 重新创建参数。接下来用同样的梯度比较两种更新规则。

## 3. SGD 与 AdamW

### 3.1 同起点对照

在 scripts/inspect_optimizer.py 先写下面的独立实验。固定起点、目标、学习率，只改变优化器；两条路径使用不同对象，防止先更新的一条影响另一条。

~~~python
import torch
from torch import nn

for kind in ("sgd", "adamw"):
    parameter = nn.Parameter(torch.tensor(2.0))
    if kind == "sgd":
        optimizer = torch.optim.SGD([parameter], lr=0.1)
    else:
        optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
    before = parameter.item()
    optimizer.zero_grad(set_to_none=True)
    loss = parameter.square()
    loss.backward()
    print(kind, "loss:", loss.item(), "gradient:", parameter.grad.item())
    optimizer.step()
    print("before/after:", before, parameter.item())
    if kind == "adamw":
        state = optimizer.state[parameter]
        print({key: value.item() for key, value in state.items()})
~~~

~~~powershell
uv run --locked python scripts/inspect_optimizer.py
~~~

两边梯度均为 4。SGD 得到 2-0.1×4=1.6；AdamW 约得到 1.9。这说明不能从“参数变化 0.001”直接反推梯度接近 1，必须知道更新规则。

### 3.2 递推、修正与更新

AdamW 为每个参数保留 m、v，默认 beta1=0.9、beta2=0.999，初始为零：

~~~text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t²

m_hat = m_t / (1-beta1**t)
v_hat = v_t / (1-beta2**t)

adaptive_update = lr*m_hat/(sqrt(v_hat)+eps)
w_new = w_old - adaptive_update          （本节 weight_decay=0）
~~~

这是三层操作：先更新历史统计，再校正零初始化偏差，最后计算参数变化。m/v 是保存状态，hat 值用于本次计算，不是把历史状态永久改成 hat。

首步 g=4，得到 m=0.4、v=0.016；修正后 m_hat=4、v_hat=16，因此更新约 0.1。m 保留正负方向，v 是平方的移动平均，不直接称为方差。

### 3.3 数值核对

在上面 AdamW 分支的 step 后追加以下代码，注意 before 必须来自更新前：

~~~python
group = optimizer.param_groups[0]
beta1, beta2 = group["betas"]
t = state["step"].item()
m_hat = state["exp_avg"].item() / (1 - beta1 ** t)
v_hat = state["exp_avg_sq"].item() / (1 - beta2 ** t)
update = group["lr"] * m_hat / (v_hat ** 0.5 + group["eps"])
predicted_after = before - update
torch.testing.assert_close(
    parameter.detach(),
    torch.tensor(predicted_after, dtype=parameter.dtype),
)
~~~

忘记 lr 会得到约 1 而不是 1.9；程序正常退出不代表公式正确，所以需要与实际参数对照。历史梯度参与统计可以理解为惯性，但不保证参数每步都小幅变化或训练必然稳定。

## 4. 解耦权重衰减

### 4.1 隔离实验

新建一个初始值仍为 2 的参数，AdamW 使用 lr=0.1、weight_decay=0.2，目标仍为 w²。不要在已更新的 1.9 上接着做“同起点对照”。

~~~python
parameter = nn.Parameter(torch.tensor(2.0))
optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
before = parameter.item()
optimizer.zero_grad(set_to_none=True)
parameter.square().backward()
optimizer.step()
state = optimizer.state[parameter]
group = optimizer.param_groups[0]
beta1, beta2 = group["betas"]
t = state["step"].item()
m_hat = state["exp_avg"].item() / (1 - beta1 ** t)
v_hat = state["exp_avg_sq"].item() / (1 - beta2 ** t)
adaptive = group["lr"] * m_hat / (v_hat ** 0.5 + group["eps"])
decay = group["lr"] * group["weight_decay"] * before
expected = before - decay - adaptive
torch.testing.assert_close(parameter.detach(), torch.tensor(expected))
print("expected/actual:", expected, parameter.item())
~~~

### 4.2 结果含义

梯度仍为 4，m/v 仍约为 0.4、0.016，衰减量为 0.04，最终约 1.86。衰减没有加入本次 loss 梯度和 m/v，而是另行按旧参数缩小，所以称为解耦。

~~~text
w_new = w_old - lr*weight_decay*w_old - adaptive_update
      = w_old*(1-lr*weight_decay) - adaptive_update
~~~

两种写法只是提取公因式，不是两套更新规则。下一步把同一规则施加到不同参数组。

## 5. 参数分组

### 5.1 规则与引用完整性

本实验将二维及以上参数设 weight_decay=0.02，一维参数设为零。这是当前实验约定，不是所有模型的普遍最优策略。共享 embedding/lm_head 只能进入一次，冻结参数不进入可训练组。

在 scripts/inspect_parameter_groups.py 按以下顺序实现：构造 tiny 模型，按 ndim 分组，检查对象身份集合，创建 AdamW，再做零梯度隔离实验。完整参考：

~~~python
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
~~~

### 5.2 为什么用零梯度

这次从新优化器的零历史开始，构造与每个参数相连但数值为零的 loss，让 .grad 是零张量而不是 None。自适应项因此为零，可以单独核对衰减：

~~~text
decay 参数：before * (1-lr*weight_decay)
不衰减参数：before
~~~

已有历史 m/v 时，当前梯度为零不保证自适应项为零；不要把这个新优化器实验推广到所有后续步。

~~~powershell
uv run --locked python scripts/inspect_parameter_groups.py
~~~

当前配置得到 15 个衰减参数、5 个不衰减参数，共 4928 和 80 个标量。断言既检查名单，也检查实际更新。若数量不符，先查共享对象是否重复、模型配置是否改变，不直接修改预期数凑通过。

## 6. 学习率调度

### 6.1 当前更新与下一更新

optimizer 使用 param_groups 中当前 lr；scheduler 按进度修改它。Warmup 与线性衰减是变化规则，scheduler 是执行规则的对象，不是三套彼此独立的算法。

先在 scripts/inspect_scheduler.py 用 StepLR 做短实验：

~~~python
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
~~~

~~~powershell
uv run --locked python scripts/inspect_scheduler.py
~~~

四次实际使用 [0.1,0.1,0.05,0.05]，更新后准备的下一学习率是 [0.1,0.05,0.05,0.025]。记录 lr_used 要放在 optimizer.step 前，scheduler.step 放在参数更新后。

### 6.2 Warmup 与线性衰减

在 scripts/inspect_warmup_decay.py 规定六次更新、前两次 warmup，基准学习率 0.1。这里采用一份明确的教学端点约定：

~~~text
调度索引 s=0..6
warmup:    multiplier=(s+1)/2                         s<2
decay:     multiplier=max(0,(6-s)/(6-2+1))           s>=2
实际 lr = base_lr * multiplier
~~~

六次更新使用 [0.05,0.10,0.08,0.06,0.04,0.02]，结束后下一 lr 为零。分母加一来自选择的端点：最后一次仍做正步长更新；不是所有线性衰减都必须这样写。

完整观察代码如下：

~~~python
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
~~~

~~~powershell
uv run --locked python scripts/inspect_warmup_decay.py
~~~

LambdaLR 创建时就使用索引 0，所以第一次更新前 lr 已经变为 0.05。只打印调度表不能证明训练发生，循环里必须有 loss 和 backward，参数也应随之变化。

Warmup 控制早期步长，衰减减小后期步长，但曲线是否改善效果仍需实验。这里验证机制，不从六步玩具例子推断最优超参数。

## 7. 标量梯度累积

### 7.1 实验目标

先验证 backward 会相加，而不是立即更新参数。两个样本目标为 0、1，参数 w=2，loss 分别为 (w-target)²。两次梯度贡献是 4、2，累加为 6，除以样本数得到平均梯度 3。

在 scripts/inspect_gradient_accumulation.py 实现两条路径：逐样本 backward 后平均梯度；一次计算平均 loss 后 backward。两边各更新一次。

~~~python
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
~~~

~~~powershell
uv run --locked python scripts/inspect_gradient_accumulation.py
~~~

两边更新后参数约为 1.7。清梯度只在窗口开始，step 只在窗口结束，否则变成两次小更新，不再等价。

### 7.2 从样本平均到 Token 平均

标量样本每条同权，除以 len(targets) 合理。语言模型每个 batch 的有效 token 数不同，这个分母不能直接照搬。下一步把目标明确为一次更新窗口中的所有有效 token 平均。

## 8. 语言模型的全局 Token 平均

### 8.1 目标范围

micro-batch i 的损失和为 S_i，有效目标数为 N_i，本次更新目标是：

~~~text
L = sum(S_i) / sum(N_i)
gradient = sum(gradient_of_S_i) / sum(N_i)
~~~

“全局”在此指一个 optimizer update 的全部累积窗口，不是整轮 epoch。整个 epoch 的训练日志混合不同更新时刻的参数，而验证 NLL 是固定参数的另一个目标统计。

两个 micro-batch 分别有 2、6 个目标时，它们权重应为 2/8、6/8，不是各 1/2。先 backward loss sum，再统一除总 token 数，与先用同一总数归一化各 loss 再 backward 数学上等价；浮点顺序仍允许微小误差。

### 8.2 对照代码

在 scripts/inspect_token_accumulation.py 使用固定权重，先大 batch，再正确累积，最后故意错误地平均 batch 均值：

~~~python
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss_sum_and_count

torch.manual_seed(42)

config = ModelConfig(
    vocab_size=8,
    hidden_size=16,
    intermediate_size=32,
)

model = MinimindForCausalLM(config).double()

input_ids = torch.tensor([
    [1, 3, 2, 0, 0, 0, 0],
    [1, 4, 5, 6, 7, 3, 2],
])

attention_mask = input_ids != 0

labels = input_ids.clone()
labels[~attention_mask] = -100

logits = model(input_ids=input_ids,attention_mask=attention_mask)

loss_sum, token_count = causal_lm_loss_sum_and_count(logits, labels)

mean_loss = loss_sum / token_count
print(f"loss_sum:{loss_sum}, token_count:{token_count}, mean_loss:{mean_loss}")

mean_loss.backward()

reference_grads = {
    name: param.grad.detach().clone()
    for name, param in model.named_parameters()
}

# 实际路径
model.zero_grad(set_to_none=True)

total_tokens = 0
total_loss_sum = 0.0

for i in range(input_ids.shape[0]):
    micro_input_ids = input_ids[i:i+1]
    micro_labels = labels[i:i+1]
    micro_attention_mask = attention_mask[i:i+1]

    logits = model(input_ids=micro_input_ids, attention_mask=micro_attention_mask)
    loss, count = causal_lm_loss_sum_and_count(logits, micro_labels)
    total_loss_sum += loss.item()
    total_tokens += count.item()

    loss.backward()

for param in model.parameters():
    param.grad.div_(total_tokens)


for name, parameter in model.named_parameters():
    torch.testing.assert_close(
        parameter.grad,
        reference_grads[name],
        rtol=1e-5,
        atol=1e-7,
        msg=lambda message: f"{name}: {message}",
    )

model.zero_grad(set_to_none=True)
wrong_mean_loss = 0.0

for i in range(input_ids.shape[0]):
    micro_input_ids = input_ids[i:i+1]
    micro_labels = labels[i:i+1]
    micro_attention_mask = attention_mask[i:i+1]

    logits = model(input_ids=micro_input_ids, attention_mask=micro_attention_mask)
    loss, count = causal_lm_loss_sum_and_count(logits, micro_labels)
    wrong_contribution = loss / count / input_ids.shape[0]
    wrong_mean_loss += wrong_contribution.item()
    wrong_contribution.backward()


max_grad_error = max(
    (parameter.grad - reference_grads[name]).abs().max().item()
    for name, parameter in model.named_parameters()
)

print(f"reference mean loss: {mean_loss.item()}")
print(f"wrong mean loss: {wrong_mean_loss}")
print(f"wrong max gradient error: {max_grad_error}")
~~~

### 8.3 结果解释与边界

~~~powershell
uv run --locked python scripts/inspect_token_accumulation.py
~~~

固定实验参考平均 loss 约 2.90018，错误均值约 2.47768，错误梯度最大差约 0.19432。正确路径逐参数 assert_close；错误路径当前通过打印观察差异，没有单独的反例断言。

本步不执行 optimizer.step，隔离的是梯度目标。模型 .double() 减小部分舍入误差，但 RMSNorm 仍显式用 float32，所以不能声称全部内部计算为双精度；使用 rtol=1e-5、atol=1e-7。

保存参考梯度后必须清零，再重建前向图。只调用 zero_grad 不会改变参数，因而三条路径仍在相同参数点计算。下一步在归一化之后加入裁剪。

## 9. 梯度裁剪

### 9.1 范数与统一缩放

在 scripts/inspect_gradient_clipping.py 使用 L=3w1+4w2，梯度 [3,4]，L2 范数为 5，阈值 1。完整观察：

~~~python
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
~~~

~~~powershell
uv run --locked python scripts/inspect_gradient_clipping.py
~~~

裁剪后约 [0.6,0.8]，保持方向，整体缩小；实际数值因防除零常数可能略小于理想值。它不是把每个分量分别截到 1。

### 9.2 放置顺序

累计梯度 → 按有效 token 总数归一化 → 对全部待更新参数统一裁剪 → optimizer.step。若归一化前先裁剪，阈值相对于目标梯度的意义改变；若各 micro-batch 分别裁剪，通常也不再等于大 batch。

返回值是裁剪前范数，方便观测原始梯度多大。日志显示超过 max_grad_norm 不等于没裁剪，也不直接证明训练失败。

## 10. 单次更新接口

### 10.1 职责和完整实现

新建 src/minimind_lab/training/__init__.py（可为空）与 training/step.py。train_update 接收一个累积窗口，完成该窗口全部计算，返回本次记录：


~~~python
import torch

from torch import nn

from minimind_lab.models.loss import causal_lm_loss_sum_and_count


def train_update(
    model,
    optimizer,
    micro_batches,
    scheduler,
    max_grad_norm=1.0,
):
    if not micro_batches:
        raise ValueError("micro batches must be not empty")

    # 准备累积， 初始化
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_tokens=0
    total_loss_sum=0.0

    # 准备遍历
    for micro_batch in micro_batches:
        logits = model(input_ids=micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"])

        loss_sum, token_count = causal_lm_loss_sum_and_count(logits, micro_batch["labels"])
        total_loss_sum += loss_sum.item()
        total_tokens += token_count.item()

        loss_sum.backward()

    parameters = []
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.div_(total_tokens)
        parameters.append(param)

    grad_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    lr_used = optimizer.param_groups[0]["lr"]

    optimizer.step()
    scheduler.step()

    lr_next = optimizer.param_groups[0]["lr"]

    return {
        "loss": total_loss_sum / total_tokens,
        "valid_tokens": total_tokens,
        "micro_batches": len(micro_batches),
        "grad_norm": grad_norm.item(),
        "lr_used": lr_used,
        "next_lr": lr_next,
    }
~~~

参数梯度为 None 时跳过，支持已经冻结的参数；没有数据时提前拒绝，不能推进 optimizer/scheduler。每个 micro-batch 在 loss 层保证至少一个有效目标，因此当前窗口总 token 数为正。

model.train() 设置模块模式，不解冻参数，也不代替 backward。指标转为 Python 数值，避免日志引用计算图；多参数组时这里仅记录第零组 lr，是当前日志接口的范围。

### 10.2 与大 Batch 的独立对照

~~~powershell
uv run --locked pytest tests/test_training.py -q -k train_update
~~~

参考路径用相同初始模型、同数据和优化器，整批平均 loss backward；待测路径拆成两个不同 token 数的 micro-batch。两边都必须先归一化再裁剪，再更新并推进 scheduler，最后比较参数、loss、token 数、裁剪前 norm 和学习率。

曾经参考路径漏掉裁剪，使两个“看似相同”的更新实际不同。排错时先核对两边操作顺序，不一开始就放宽容差。另检查空输入与冻结共享权重，避免正常路径掩盖 None 梯度错误。

## 11. 整轮循环与尾部窗口

### 11.1 组织职责

train_update 管一个更新窗口；train_epoch 只负责把 DataLoader 交付的 micro-batch 分成窗口，不能在外层重复 backward 或 scheduler.step。

在 training/loop.py 写下面的第五周基础版本。当前仓库已经为第六周加上 sampler=None 和 commit；复练时理解本版本即可，不要覆盖已有恢复逻辑。

~~~python
from minimind_lab.training.step import train_update

def train_epoch(
    model, optimizer, dataloader, scheduler,
    gradient_accumulation_steps=2, max_grad_norm=1.0,
):
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    pending_batches = []
    history = []
    for batch in dataloader:
        pending_batches.append(batch)
        if len(pending_batches) == gradient_accumulation_steps:
            history.append(train_update(
                model, optimizer, pending_batches, scheduler,
                max_grad_norm=max_grad_norm,
            ))
            pending_batches = []
    if pending_batches:
        history.append(train_update(
            model, optimizer, pending_batches, scheduler,
            max_grad_norm=max_grad_norm,
        ))
    return history
~~~

### 11.2 尾部验证

五个 micro-batch，累积步数为 2，应形成 [2,2,1] 三个窗口。最后一个按自己的真实 token 数归一化，不再额外除以计划的 2，也不能丢掉它。

~~~powershell
uv run --locked pytest tests/test_training.py -q -k train_epoch
~~~

参考测试手动调用三次 train_update，与外层循环比较。用尾部前后的参数快照、AdamW step=3、scheduler 推进三次共同证明最后一批实际参与更新。空 epoch 应返回空 history，不改变状态。

## 12. 验收与衔接

### 12.1 回归检查

~~~powershell
uv run --locked pytest tests/test_loss.py tests/test_training.py -q
~~~

| 行为 | 验证理由 |
| --- | --- |
| 整批与累积更新一致 | 同一个 token 平均目标没有被改写 |
| 空列表/None 提前报错 | 无数据时不产生一次更新 |
| 冻结参数不变 | train 模式不是解冻，None 梯度能处理 |
| 尾部实际更新 | 不静默丢弃最后一个窗口 |
| 空 epoch 不变 | 没有输入就没有训练进度 |
| 非法累积步数拒绝 | 错误配置在计算前暴露 |

历史第五周验收是训练测试 8 项、全仓 229 项；当前 test_training.py 还包含第六周 evaluate 测试，全部执行也应通过。测试数量不是算法理解程度的分数。

### 12.2 实践边界

当前是小模型、单设备、batch 已放到正确设备的训练计算；没有 autocast/scaler、分布式同步、数据指纹、周期保存或恢复。训练输入以一个完整窗口的 batch 列表交付，但每次 backward 后不要求保留此前计算图，不能把梯度累积理解为同时保存全部 micro-batch 的图。

核心与观察实验由学习者在提示下完成；补充边界测试由助手按授权整理，不能全部记作学习者独立实现。下一周在这个训练循环上增加验证、checkpoint、随机状态与样本消费位置，不再重新发明参数更新链路。

## 13. 学习问答

### 13.1 为什么参数的梯度也是 Tensor，可以 detach？

Parameter 是 Tensor 的特殊形式，.grad 通常也是 Tensor。两者都可以使用 Tensor 操作。梯度快照 detach().clone() 用于独立保存数值，不是删除参数本身或让未来训练永久停止求导。

### 13.2 id(parameter) 是什么，为什么用集合检查？

id 表示当前对象生命周期中的身份，不是参数数值。set 去重后可检查组内重复；isdisjoint 检查组间无交叉；竖线表示并集。两个参数数值相同也可能是不同对象，优化器是否重复持有同一对象才是这里关注的问题。

### 13.3 param_groups 和 state 是两套参数吗？

不是。param_groups 持有 Parameter 引用和学习率、衰减等分组配置；state 保存该 Parameter 的 m/v、step 等历史。共享配置不等于共享历史。同一个参数对象被两处引用，也不应因此更新两次。

### 13.4 为什么权重衰减写成 before*(1-lr*weight_decay)？

它等于 before-lr*weight_decay*before。只在自适应项为零的隔离实验里，最终参数就是这部分；正常 AdamW 还要减自适应更新，不能漏掉。

### 13.5 Warmup 的 2 是固定算法常数吗？

不是。本例 warmup 两次是为了手算六步计划，真实 warmup 长度属于实验配置。LambdaLR 接收倍率函数，StepLR 有自己的分段规则，“实际学习率=基准×倍率”不决定倍率必须长什么样。

### 13.6 调度曲线正确就能证明训练更好吗？

不能。规则正确是实现验证，效果更好需要与基线比较。学习率形状受架构、优化器、数据和预算影响；本周不将教学端点当成最佳配置，也不新增一轮超参数搜索。

### 13.7 (w-target)² 的导数为什么不是 2w？

对 w 求导得到 2(w-target)。target 为常量意味着其导数为零，不表示表达式中的 target 可以先删掉。展开后交叉项 -2*target*w 的导数仍为 -2*target。

### 13.8 Tensor 为什么能和数字相减，打印都要 item 吗？

Tensor 支持与 Python 标量逐元素运算，结果仍参与计算图。可以直接 print(tensor)；item 只在需要 Python 单元素数值时使用，多元素矩阵不能整体 item。

### 13.9 为什么清梯度后 grad.item 报错？

zero_grad(set_to_none=True) 把梯度设为 None，首次 backward 才创建或填充它。要观察上一轮梯度，应在清理前保存副本，而不是假设 None 就是零张量。

### 13.10 全局 Token 平均是一个 epoch 的平均吗？

这里指一次 optimizer update 的累积窗口。训练 epoch 的日志平均和固定模型的验证 NLL 可以另算，但不能用它们替代本次更新的梯度分母。

### 13.11 为什么 double 后还保留相对和绝对容差？

double 使相应浮点参数转为 float64，但模块内部仍可能显式转为 float32，计算顺序也会引入舍入差。容差需要结合具体实现，而不是因为写了 double 就要求逐位完全相同。

### 13.12 pytest.approx 为什么同时有 rel 和 abs？

对普通有限标量，近似判断允许的差约为 max(rel*abs(expected), abs)。rel 随量级变化，abs 保护接近零的值。torch.testing.assert_close 使用不同的容差组合规则，不要把两个接口的数学条件混用。

### 13.13 grad_norm 为什么返回裁剪前数值？

需要知道原梯度到底多大。若只返回裁剪后数值，原来是 5 还是 1000 都可能显示约 1，异常程度被隐藏。实际梯度已原地修改，必要时可另算裁剪后范数。

### 13.14 model.train 会开启冻结参数吗？

不会。它改变依赖 training 标志的模块行为，如 Dropout、BatchNorm；requires_grad 控制求导，optimizer 的参数集合控制更新对象。三者职责不同。

### 13.15 为什么同一批更新后 loss 没变，或者变大？

先重新 forward，旧 loss 不会自动刷新；再查是否有 backward、参数是否真的被更新、学习率是否为零。单次更新也不保证任意目标都下降，尤其存在较大学习率、随机性或额外衰减时。这里用确定性小实验先排除实现错误，再讨论训练策略。
