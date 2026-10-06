# 07 测试策略与 pytest

## 本次要解决的问题

前六个主题都使用测试证明训练合同，但“能运行现有测试”不等于“能独立设计测试”。本主题
系统学习 pytest 的发现、组织、隔离、断言和失败诊断，并把它们应用到本项目的数据、数值、
状态恢复、CLI 和框架集成测试中。

本主题放在业务代码回顾之后：先知道每条合同为什么重要，再学习如何选择最低成本且足够有力
的测试证明它。目标不是记忆 pytest API，而是能够从风险反推测试层级和断言。

## 学习范围

1. pytest 如何发现测试模块、函数和参数化 case。
2. Arrange-Act-Assert 与“一项测试只证明一个主要行为”。
3. `fixture`、`tmp_path`、`monkeypatch`、`capsys` 和资源清理。
4. `pytest.raises`、`pytest.mark.parametrize` 与失败分支设计。
5. `assert`、`pytest.approx`、`torch.testing.assert_close` 的适用边界。
6. fake、stub、tiny model 与真实集成测试分别能证明什么。
7. 纯函数单测、模块集成、CLI 子进程和端到端 smoke 的成本梯度。
8. `-q`、`-x`、`-k`、节点 ID、collection error 和 traceback 的阅读方法。
9. 如何先写失败测试，再完成修复和回归验证。
10. 如何识别脆弱测试、重复覆盖、错误 mock 和“测试通过但合同未被证明”。

## 本项目的测试地图

| 层级 | 代表文件 | 主要证明内容 |
| --- | --- | --- |
| 纯计算 | `test_sft_loss.py` | shift、mask、reduction、梯度 |
| 模块行为 | `test_sft_data.py`、`test_engine.py` | padding、累积、更新时序和状态 |
| artifact / 状态 | `test_artifact_data.py`、`test_checkpointing.py` | 文件读取、身份和恢复边界 |
| 框架对齐 | `test_reference_alignment.py`、`test_trl_training.py` | 独立实现与锁定框架行为 |
| CLI 集成 | `test_cli.py`、`test_trl_reference_cli.py` | 参数、退出码、文件产物和错误路径 |
| 真实依赖集成 | `test_qwen_tokenizer_integration.py` | 锁定 tokenizer/chat template 的真实行为 |

## 张量测试速查

补充日期：2026-10-01。

测试时先明确要证明什么：形状正确、数值正确、梯度正确，还是参数真的更新了。
`assert` 是 Python 的断言语句，PyTorch 提供的是张量比较与检查工具；二者配合使用。
下面的例子可以放在 pytest 测试函数里运行，不要求背下所有方法。

### 常用语句

| 检查内容 | 常用写法 | 含义 |
| --- | --- | --- |
| 形状 | `assert x.shape == (2, 3)` | 两条样本，每条三个特征 |
| 维数 | `assert x.ndim == 2` | 有两个轴，不是有两个元素 |
| 元素数 | `assert x.numel() == 6` | 总共有六个元素 |
| 类型 | `assert labels.dtype == torch.long` | 标签是整数张量 |
| 设备 | `assert x.device == weight.device` | 输入与参数在同一个设备上 |
| 标量 | `assert loss.ndim == 0` | loss 是零维张量，形状为 `[]` |
| 有限值 | `assert torch.isfinite(x).all().item()` | 没有 NaN、正无穷或负无穷 |
| 全部满足 | `assert (x >= 0).all().item()` | 每个元素都非负 |
| 至少一个满足 | `assert (x > 0).any().item()` | 至少有一个正数 |
| 非零元素数 | `assert torch.count_nonzero(x).item() == 3` | 恰好三个元素不为零 |
| 完全相等 | `assert torch.equal(actual, expected)` | 形状相同，所有元素数值完全相同 |
| 近似相等 | `torch.testing.assert_close(actual, expected)` | 在容差内一致，失败时提供差异诊断 |
| 同一对象 | `assert received is batch["input_ids"]` | 接口收到的就是原来的对象，而非副本 |

`torch.equal` 不单独保证 dtype 相同；整数标签测试如需约束类型，应另外检查 dtype。
`assert_close` 默认还检查 shape、dtype 和 device。跨设备比较应先明确是否允许设备不同，
不要仅为通过测试而关闭这些检查。

### 比较方式

整数 token ID、标签和 mask 通常要求完全相等；浮点 loss、梯度和模型输出通常使用近似比较。
例如下面两份浮点结果不完全相等，但差异处于指定容差内：

```python
import torch

actual = torch.tensor([1.0, 2.000001])
expected = torch.tensor([1.0, 2.0])

assert not torch.equal(actual, expected)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-7)
```

`rtol` 是随参考值大小变化的相对容差，`atol` 是绝对容差。这里可理解为：

```text
允许误差 = atol + rtol * abs(expected)
```

`torch.allclose` 返回布尔值，`torch.testing.assert_close` 在不满足要求时抛出断言错误。
pytest 中优先用后者，便于定位差异。容差应根据 dtype、计算路径和要验证的合同选择，
不是越宽松越好；上面的数值不是所有训练测试的统一标准。

另外，`is` 检查对象身份，不检查形状或数值：

```python
x = torch.tensor([1, 2, 3])
alias = x
copied = x.clone()

assert alias is x
assert copied is not x
assert torch.equal(copied, x)
```

### 布尔索引

张量的 `==`、`!=` 会逐元素比较，产生布尔张量，不能把多元素结果直接当成一个真假值。
我们在 SFT 中使用的有效标签统计就是这种写法：

```python
labels = torch.tensor([[-100, 21, 22, 99], [-100, 31, -100, -100]])
valid = labels != -100

assert valid.dtype == torch.bool
assert valid.sum().item() == 4
assert valid.any().item()
assert not valid.all().item()
assert torch.equal(labels[valid], torch.tensor([21, 22, 99, 31]))
```

`labels[valid]` 取出对应位置为 True 的元素。这里返回一维张量，因为它是在收集所有选中元素。
`.all()` 将多个真假值归约为“全部满足”，`.any()` 归约为“至少一个满足”。`.item()`
把单元素张量转换为 Python 值；多元素张量不能直接调用它。

实际 SFT 的有效 token 数应在 causal shift 后统计：`(labels[:, 1:] != -100).sum()`。
上面的例子仅演示布尔索引，不代替完整的 shift 合同。

### 梯度与更新

下面是一个完整的小测试：先验证 forward，再验证 backward，最后验证 optimizer.step。
初始预测为 2，目标为 5，因此 loss 为 9，参数梯度分别为 -12 和 -6。

```python
def test_linear_backward_and_step():
    x = torch.tensor(2.0)
    target = torch.tensor(5.0)
    w = torch.nn.Parameter(torch.tensor(1.0))
    b = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([w, b], lr=0.1)

    before_w = w.detach().clone()
    before_b = b.detach().clone()
    optimizer.zero_grad(set_to_none=True)

    prediction = w * x + b
    loss = (prediction - target).square()

    assert loss.ndim == 0
    assert loss.requires_grad
    assert loss.grad_fn is not None
    torch.testing.assert_close(loss, torch.tensor(9.0))

    loss.backward()

    assert w.grad is not None
    assert b.grad is not None
    assert torch.isfinite(w.grad).all().item()
    assert torch.isfinite(b.grad).all().item()
    torch.testing.assert_close(w.grad, torch.tensor(-12.0))
    torch.testing.assert_close(b.grad, torch.tensor(-6.0))
    torch.testing.assert_close(w.detach(), before_w)
    torch.testing.assert_close(b.detach(), before_b)

    optimizer.step()

    torch.testing.assert_close(w.detach(), torch.tensor(2.2))
    torch.testing.assert_close(b.detach(), torch.tensor(0.6))
    assert not torch.equal(w.detach(), before_w)

    with torch.no_grad():
        after_prediction = w * x + b
        after_loss = (after_prediction - target).square()

    assert not after_prediction.requires_grad
    assert after_prediction.grad_fn is None
    assert after_loss.item() < loss.item()
```

`detach().clone()` 是保存独立的参数快照：detach 让快照不连接计算图，clone 让它不与原参数
共享存储。它不会取消原参数的梯度能力。只保存 `before_w = w`，参数更新后看到的仍是同一个
对象，无法证明更新前后发生了什么。

梯度检查通常针对叶子张量或 `nn.Parameter`。中间输出默认不保留 `.grad`，如确实需要查看，
应在 backward 前调用 `output.retain_grad()`；不能仅凭它的 `.grad is None` 判断梯度没有传播。
`torch.no_grad()` 阻止其中的新计算建立计算图，不会清空此前已经存在的参数梯度。

“梯度非零”“参数发生变化”“单步 loss 下降”也不是所有输入下都应成立的通用要求。
这里选用了已知非零梯度的小例子；真实训练还受学习率、随机性和优化器状态影响。

### Mask 检查

本项目的 `test_sft_loss.py` 检查了被忽略目标对应的 logits 梯度。常用写法为：

```python
masked_grad = torch.tensor([0.0, 0.0])
torch.testing.assert_close(
    masked_grad,
    torch.zeros_like(masked_grad),
    rtol=0,
    atol=0,
)
```

这要求严格为零。实际测试应把 `masked_grad` 替换成 backward 后选出的对应 logits 梯度。
需要区分：忽略某个目标的 loss，并不意味着对应输入位置的 embedding 或 hidden state
一定没有梯度；后续受监督 token 仍可能通过 attention 使用该位置的信息。

### 异常检查

除了正常数值，还要验证不合规输入是否按合同被拒绝。pytest 的异常断言可这样使用：

```python
import pytest

with pytest.raises(RuntimeError, match="cannot be multiplied"):
    torch.ones(2, 3) @ torch.ones(2, 3)
```

`pytest.raises` 检查异常类型，`match` 按正则表达式匹配错误信息。
项目接口的测试更应检查自己的 ValueError 与稳定错误说明，而不是依赖第三方错误全文。

### 常见误区

- 不写 `assert actual == expected` 来直接比较多元素张量，改用 `torch.equal` 或 `assert_close`。
- 不把 `tensor([1.0])` 与零维 `tensor(1.0)` 混为一谈；元素数都为一，形状却不同。
- 不用浮点逐位相同代替数值正确性；也不为掩盖误差随意放宽容差。
- 不只检查 loss 能否下降，还要检查 shift、mask、归约方式与参考梯度。
- 不把 `.item()` 后的 Python 数值作为 backward 的对象；反向传播需要连接计算图的 loss 张量。
- 固定 `torch.manual_seed(...)` 可减少随机干扰，但不能保证跨设备、跨版本逐位复现。
- `.item()` 在 GPU 上可能触发同步，适合小测试，不宜作为每个 micro-batch 的高频检查。
- `assert` 用于测试；生产代码的输入校验应显式抛出异常，因为 Python `-O` 可禁用 assert。

## 计划中的动手练习

1. 从一个现有失败日志判断它属于收集失败、测试准备失败、行为失败还是断言设计错误。
2. 为纯函数补一个参数化边界测试，并解释每个 case 为什么必要。
3. 使用 `tmp_path` 测试一次 artifact 或 checkpoint 写入，不污染真实运行目录。
4. 使用 fake/stub 验证调用时序，再说明它不能替代哪一项真实集成测试。
5. 为一个 CLI 门禁先写 RED 测试，检查退出码、stderr 和“不应产生的文件”。
6. 对同一合同设计单元测试与集成测试，比较证据强度和运行成本。

## 验收标准

- [x] 能解释 pytest 从命令到收集、fixture、执行、teardown 和报告的过程。
- [x] 能根据合同选择断言与测试层级，而不是照抄已有测试结构。
- [x] 能独立使用参数化、临时目录、异常断言和 CLI 子进程测试。
- [x] 能区分 fake 对调用合同的证明与真实依赖对行为合同的证明。
- [x] 能从 traceback 找到第一个属于本项目的失败位置。
- [x] 能完成一次 RED-GREEN-REGRESSION，并说明剩余未覆盖风险。

## 快速问答与学习记录

学习日期：2026-09-22。

本轮使用项目现有测试快速建立了测试地图：纯计算测试负责局部数值合同，模块测试负责文件、
状态和协作行为，CLI 子进程测试负责真实入口、退出码、标准流与副作用，真实依赖集成测试负责
验证 fake 无法证明的外部行为。pytest 的执行链路是配置读取、收集、fixture setup、测试执行、
teardown 和报告；collection error 与行为断言失败发生在不同阶段。

快速问答确认：

1. `tmp_path` 是 pytest 提供的独立临时目录，类型为 `pathlib.Path`，避免测试污染正式产物。
2. `monkeypatch` 只证明替换接口下的调用和分支合同，不能代替真实第三方依赖的行为验证。
3. CLI 门禁需同时验证退出码、`stderr` 和不应产生的文件，才能证明失败原因与副作用边界。
4. 参数化 case 可被独立收集、执行和报告；函数内部循环不能提供同等的 case 隔离与节点 ID。
5. 默认配置包含 `-m "not integration"`，因此真实 Qwen tokenizer 测试需要显式选择。

本轮沿用了主题 06 已完成的 RED-GREEN-REGRESSION：先用测试暴露“指标不变时丢失已变化
prediction”的风险，再修正输入构造与实现回归。代表性验证覆盖配置参数化、checkpoint 失败
清理、CLI 子进程门禁和评测回归，共 16 个测试通过。真实 tokenizer integration 本轮未执行；
它仍是验证锁定 Qwen chat template 行为的独立证据层。
