# 第 3 周：模型基础组件实践教程

> 教程整理：2026-10-19。先完成逐位置特征计算，下一周才加入跨位置注意力；QA 在文末。

## 1. 本周任务与准备

第二周已经把文本转成 input_ids、attention_mask 和 labels。模型不会把编号大小当成词义，需要先把 ID 映射为可训练特征，再加工这些特征。本周实现 Embedding、RMSNorm、SwiGLU 和前馈残差块，并补齐配置与初始化。

~~~text
input_ids [B,T] → Embedding [B,T,D]
               → x + SwiGLU(RMSNorm(x)) [B,T,D]
               → 临时标量目标 → backward → 参数梯度
~~~

这不是完整语言模型：没有 Attention、词表输出和 next-token loss。平方均值只是检查梯度能否贯通的目标，不能据此宣称模型学会语言。

### 1.1 环境与完成标准

在 implementation 目录执行 uv sync --locked。终端命令为 PowerShell；观察脚本在 scripts，核心实现放在 src/minimind_lab/models，测试在 tests。每个独立观察脚本顶部按需导入 import torch 和 from torch import nn；正文共享变量只在注明的同一脚本内继续使用。

完成标准：组件的输入输出、参数和梯度都可解释；数值与独立公式或参考实现一致；组合前向与反向通过。复练在自己的副本实现，避免覆盖现有工程。当前 config.py、block.py 已包含第四周扩展，本周只实现或阅读对应的基础部分，不删掉后续代码。


**分步测试约定**：下面的 pytest 命令用于已有完整仓库的复练核验。从空文件逐步实现时，先执行本步给出的手算或最小断言；现有测试文件可能在收集时就导入后续类，-k 不会跳过这些导入。相关模块完成后再运行对应测试文件，不为通过尚未实现的后期测试提前增加功能。

### 1.2 实践顺序

| 步骤 | 产物 | 验证目的 |
| --- | --- | --- |
| 2–3. 自动求导与 Linear | 两个观察脚本 | 区分前向、反向、更新及特征变换 |
| 4. Embedding | TokenEmbedding | 查表和重复 ID 梯度累加 |
| 5. RMSNorm | RMSNorm | 沿特征维归一化与上下游梯度 |
| 6. SwiGLU | 前馈模块 | 两分支计算与逐位置独立性 |
| 7. 残差 | ResidualFeedForward | 原表示与直接梯度路径 |
| 8–9. 配置和初始化 | ModelConfig、初始化函数 | 尺寸和起始参数一致 |
| 10. 组合 | 组件 smoke | 连接正确、梯度存在且有限 |
| 11. 验收 | 对照与边界总结 | 不把局部通过当作完整模型能力 |

### 1.3 组件分工

先用一句话判断当前模块在做什么，再进入它的公式与代码。

| 模块 | 在信息处理中的作用 | 不能据此推断 |
| --- | --- | --- |
| Embedding | 将离散编号映射为可训练的初始特征向量 | 查表本身已经理解了上下文 |
| Linear | 用学习到的权重组合当前向量的特征 | 线性升维会自动增加独立输入信息 |
| 激活 | 使特征响应随输入非线性变化 | 没有可训练参数就没有作用 |
| RMSNorm | 调整每个位置特征向量的整体尺度，再逐特征缩放 | 归一化产生了新语义或使所有向量相同 |
| SwiGLU / FFN | 对当前位置已有的表示进行非线性组合与加工 | 在不同 token 位置之间直接交换信息 |
| Attention | 根据当前表示在允许的位置之间获取、汇总信息 | 它只是拆分最后一维的特征 |
| 残差 | 将分支计算的结果加回原表示，并提供直接梯度路径 | 信息与梯度一定不会受损 |

这里的分工是理解结构的入口，不是“一个模块只会一种能力”的绝对划分。
例如 attention 自身也包含特征投影和非线性，FFN 的输入也可能已经包含上下文。

为了看清它们如何配合，可提前观察 MiniMind 的一个 pre-norm Decoder block：

```text
x -> RMSNorm -> Attention -> 与原 x 相加 -> h
h -> RMSNorm -> SwiGLU   -> 与原 h 相加 -> y
```

Embedding 位于这些 block 之前，每一层继续加工上一层的表示。
同一个 token 的初始查表向量可以相同，但所在上下文、位置和后续计算不同，表示就可能不同。
当前只实现基础组件，这张图用于定位；attention 的实际计算与完整 block 验收留到第四周。

第二张是各组件共同遵循的训练链路：

```text
输入 + 可训练参数 -> forward -> 输出 -> loss
                         <- 梯度 <- backward
参数 + 参数梯度 -> optimizer.step -> 更新后的参数
```

原理复习统一从[模型基础系列](../../../../model_architecture_and_algorithms/01_model_architecture/README.md)进入。
八篇文章依次连接任务、矩阵、激活、Embedding、归一化、SwiGLU、残差与整体结构、自动求导。
其中[整体结构篇](../../../../model_architecture_and_algorithms/01_model_architecture/models/decoder_only_transformer/01_DECODER_ONLY_TRANSFORMER.md)
专门解释归一化、激活和残差的作用与协作；梯度保存和参数更新集中在
[自动求导篇](../../../../model_architecture_and_algorithms/01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)。
本篇仍记录实际操作、测试与故障，不将知识导航变成另一份实践路线。

这里先保留整体结构作定位，再逐项进入实现。必要的数学解释留在步骤中；API 的延伸问题集中到最后。

## 2. 自动求导与最小网络

### 2.1 目的

先用能手算的网络解释参数、预测、loss、梯度和更新，避免把 `backward()` 与训练混为一谈。
这里使用平方误差和 SGD，只为观察共同机制，不代表语言模型的最终 loss 或 optimizer。

### 2.2 理解要点

架构规定“输入怎样通过参数与计算变成输出”，loss 规定“怎样评价输出”，
backward 根据 loss 计算导数，optimizer 再按规则更新参数。这几项相关，但职责不同。
本例的平方误差只决定训练信号，不会把 `w*x+b` 自动变成更复杂的模型结构。

一次前向不等于一次训练，得到梯度也不等于参数已经更新。
同样，某个张量出现在计算图中，不意味着它一定交给 optimizer：后者持有的参数对象决定更新范围。

### 2.3 操作

在 `scripts/inspect_autograd.py` 中，从以下数值开始：

```text
输入 x = 2，目标 target = 5
参数 w = 1，b = 0
预测 prediction = w * x + b
loss = (prediction - target) ** 2
学习率 lr = 0.1
```

先用 `requires_grad=True` 的浮点张量创建 w、b，把它们交给 `torch.optim.SGD`。
随后按顺序执行：

1. `optimizer.zero_grad()` 清理旧梯度。
2. 计算 prediction 和 loss。
3. `loss.backward()`，观察 `.grad` 和参数值。
4. `optimizer.step()`，再观察参数值。
5. 在 `torch.no_grad()` 中重新计算预测与 loss。

可手算的结果为：

| 观察点 | 结果 |
| --- | --- |
| 初始预测、loss | `2`、`9` |
| `backward()` 后的梯度 | `w.grad = -12`、`b.grad = -6` |
| `backward()` 后的参数 | 仍为 `w = 1`、`b = 0` |
| `step()` 后的参数 | `w = 2.2`、`b = 0.6` |
| 更新后的预测、loss | 约 `5`、`0` |

浮点打印可能略有舍入差异。梯度来自：

```text
dL/dw = 2 * (prediction - target) * x
dL/db = 2 * (prediction - target)
SGD 更新：参数 -= lr * 参数梯度
```

### 2.4 完整观察脚本

先实现上述步骤，再与仓库观察脚本核对。为了不依赖对话中的隐式变量，参考完整入口如下：

~~~python
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
~~~

### 2.5 模块封装

观察清楚后，把 w、b 放进 `ScalarLinear(nn.Module)`：

```python
class ScalarLinear(nn.Module):
    def __init__(self, w: float = 1.0, b: float = 0.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        return self.w * x + self.b
```

`nn.Parameter` 把张量注册成模型参数；`model.parameters()` 可以把这些参数交给 optimizer。
调用 `model(x)` 由 `nn.Module` 转入 `forward`，模块本身不会自动计算 loss 或更新参数。

更准确地说，将 Parameter 赋给已初始化的 Module 属性时，它才成为该模块注册的参数；
将另一个 Module 赋给属性则注册为子模块。`super().__init__()` 先建立这些管理机制，
因此应在注册参数和子模块前调用。
Python 的继承让我们复用 nn.Module 提供的参数遍历、子模块管理和调用机制，
不是 dataclass 或普通 Tensor 自动提供了这些行为。

### 2.6 排错

最初把前向与 `backward()` 放在 `no_grad()` 内，导致新计算没有建立所需计算图。
修正方法是让训练前向和 backward 在正常梯度模式下执行，仅把更新后观察放进 `no_grad()`。
`no_grad()` 不会清掉旧 `.grad`，也不会修改参数的 `requires_grad` 标志。

另一个问题是用默认 Tensor 作为构造函数参数。默认对象可能被多个实例共享，
包装为 Parameter 也不应当作自动复制存储。当前采用标量默认值，在每次初始化时新建张量。

运行 `uv run python scripts/inspect_autograd.py`，检查上述顺序和数值。
参考：[当前脚本](../implementation/scripts/inspect_autograd.py)。

## 3. 线性变换与激活

### 3.1 目的

从标量扩展到特征向量，理解线性层处理哪一维、参数怎样共享，以及非线性增加了什么。

### 3.2 理解要点

对 `[B,T,D]` 来说，最后一维 D 是每个位置的特征，不是固定的“第二维”。
Linear 在最后一维重组特征，前几维保留位置关系；它不需要按 batch 和 token 写 Python 循环。
共享权重指所有位置使用同一套变换，不代表把这些位置的输入混起来。

激活的作用不能简化成“修剪不重要的 token”。ReLU 对中间特征逐元素计算，
可以理解为根据当前值决定该通道是否传递信号；SiLU 则是连续调节，不是严格的开关。
这个解释针对计算通道，不等于某个通道已经有固定的人类可读语义。

### 3.3 线性实验

在 `scripts/inspect_linear.py` 中构造：

```python
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]])
weight = torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0]])
bias = torch.tensor([0.5, -0.5])
manual_output = x @ weight.T + bias
```

再创建 `nn.Linear(3, 2)`，在 `no_grad()` 中用 `copy_` 写入相同的 weight、bias，
调用该层，并用 `torch.testing.assert_close` 与手算结果比较。

```text
x [2, 3] @ weight.T [3, 2] + bias [2]
-> [[5.5, 0.5], [14.5, 1.5]]，形状 [2, 2]
```

两行输入使用同一组参数，但各自计算，线性层不会把两行输入的信息混在一起。
对 `[B, T, D_in]` 的输入也是如此：最后一维经过相同变换，输出 `[B, T, D_out]`。
可按逐位置计算理解，但实际由批量张量运算执行，不需要 Python 循环。

### 3.4 激活实验

接着构造 `probe = torch.tensor([-1.0, 0.0, 1.0])`，比较两条路径：

```python
hidden = 2 * probe + 1
linear_output = 3 * hidden - 2
merged_output = 6 * probe + 1
nonlinear_output = 3 * torch.relu(hidden) - 2
```

| 路径 | 结果 |
| --- | --- |
| 两层线性 | `[-5, 1, 7]` |
| 合并为一层 | `[-5, 1, 7]` |
| 两层中间加入 ReLU | `[-2, 1, 7]` |

这里“线性层”按深度学习惯例指带偏置的仿射变换。
没有其他操作的两层仿射变换仍能合并为一层；ReLU 让不同输入区间的计算斜率不同。
输出不同本身只证明计算改变，“不能整体合并为一层”还要结合函数在不同区间的行为解释。

### 3.5 排错

最初写成 `relu(probe)`，是在原输入上激活，不是在线性层之间激活。
正确比较应使用 `relu(hidden)`，使两条路径只相差中间的激活操作。

运行 `uv run python scripts/inspect_linear.py`。当前脚本主要通过断言检查，没有输出也可能正常完成。
参考：[当前脚本](../implementation/scripts/inspect_linear.py)。


在 inspect_linear.py 顶部导入 torch 与 nn；前向对照的参数复制完整写法如下：

~~~python
layer = nn.Linear(3, 2)
with torch.no_grad():
    layer.weight.copy_(weight)
    layer.bias.copy_(bias)
torch.testing.assert_close(layer(x), manual_output)
~~~

非线性改变特征响应，下一步需要先把离散 token 变成可以接受这些计算的向量。

## 4. Embedding：从编号到向量

### 4.1 目的

token ID 是编号，不是连续特征。Embedding 用一张可训练表，为编号查出特征向量。
它与 tokenizer 分工不同：tokenizer 决定编号，Embedding 决定该编号当前的向量表示。

### 4.2 理解要点

ID=5 不意味着比 ID=2 更大、更重要或语义更接近某个词；编号只用于定位参数表中的一行。
查表把离散输入接到可训练的连续计算上，loss 的梯度可以更新被查询的参数行。
因此 Embedding 不只是预先准备好的静态字典，也不需要在 forward 时重新执行分词。

同一个 ID 共享参数行，但不同位置经过后续 attention 等计算，可以获得不同上下文表示。
词表中的一行也不必等于一个完整汉字或词，它对应的是 tokenizer 定义的 token。

### 4.3 操作

在 `scripts/inspect_embedding.py` 中创建：

```python
embedding = nn.Embedding(6, 3)
input_ids = torch.tensor([[1, 2, 1], [3, 0, 2]])
hidden = embedding(input_ids)
```

先打印参数表与输出形状，再比较 `hidden[0, 0]`、`hidden[0, 2]`：

```text
参数表 [V, D] = [6, 3]
输入 [B, T] = [2, 3]
输出 [B, T, D] = [2, 3, 3]
```

两处 ID 都是 1，所以查同一参数行，初始向量相同。
它们共享参数，不应据此推断输出存储地址相同；加入上下文计算后，两个位置的表示也可以不同。
随机初始化的具体向量不固定，观察的是查表关系与形状。

再执行：

```python
hidden.sum().backward()
print(embedding.weight.grad)
```

预期参数梯度为：

```text
ID 0：[1, 1, 1]
ID 1：[2, 2, 2]
ID 2：[2, 2, 2]
ID 3：[1, 1, 1]
ID 4：[0, 0, 0]
ID 5：[0, 0, 0]
```

这里每次查询贡献 `[1, 1, 1]`，是因为求和对每个输出元素的导数为 1。
真实 loss 下，各位置传回的梯度可能大小、方向不同。共享参数梯度会累加，
但出现次数更多不保证最终梯度更大，也不保证更新后的向量更大。

运行 `uv run python scripts/inspect_embedding.py`。
参考：[当前脚本](../implementation/scripts/inspect_embedding.py)。

### 4.4 合同

在 `src/minimind_lab/models/embedding.py` 实现 `TokenEmbedding`：

| 项目 | 当前约定 |
| --- | --- |
| 构造参数 | `vocab_size`、`embedding_size` |
| 参数 | `weight`，形状 `[V, D]`，通过 `nn.Parameter` 注册 |
| 输入 | 有效范围内的 token ID，练习使用 `torch.long` |
| 输出 | `weight[input_ids]`，形状为输入形状再追加 `D` |
| 暂不支持 | 特殊 padding 行、稀疏梯度等扩展选项 |

这张量索引会逐个查出编号对应的整行，自动保留输入的前几维，不需要遍历 batch 或位置。
当前实现用 `torch.rand` 初始化，值在 `[0, 1)`；`torch.randn` 则来自标准正态分布。
这不影响查表合同，但不代表与官方初始化相同，后续模型装配时统一初始化策略。

### 4.5 核心实现

在 models/embedding.py 写入：

~~~python
import torch

from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(vocab_size, embedding_size))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]
~~~

### 4.6 测试

在 `tests/test_embedding.py` 中完成两项测试：

1. `test_embedding_forward_matches_torch`：创建自己的实现和 `nn.Embedding(10, 3)`，
   在 `no_grad()` 中统一参数；输入 `[[1,2,3],[2,3,4]]`，检查 `[2,3,3]` 与前向数值。
2. `test_embedding_gradients_match_torch`：统一参数，输入 `[[1,2,1],[3,0,2]]`，
   分别对输出求和并 backward；比较参数梯度，再比较独立的十行预期梯度。

第二项中第 0、1、2、3 行分别全为 1、2、2、1，余下六行全为 0。
这种“参考实现 + 独立预期”既检查计算一致，也检查梯度累加与未查询行的行为。

### 4.7 排错

曾在测试中访问 `manual_output.grad`，得到 None，传给 `torch.equal` 后报错。
输出是中间结果，默认不保存 `.grad`；这里要检查的是 `model.weight.grad`。
中间结果没有保存 `.grad`，不等于反向传播没有经过它。

若确实需要观察中间结果梯度，可在 backward 前调用 `retain_grad()`，但本项参数测试不需要。

运行 `uv run pytest -q tests/test_embedding.py`，对应查表与梯度两类测试。
参考：[实现](../implementation/src/minimind_lab/models/embedding.py)、
[测试](../implementation/tests/test_embedding.py)。

## 5. RMSNorm：控制特征尺度

### 5.1 目的

Embedding 输出向量后，归一化调整向量的数值尺度。
RMSNorm 沿最后一维计算，不改变输入形状，也不减均值或把值限制在 `[0, 1]`。

### 5.2 理解要点

这里关心的是“特征向量整体有多大”，不是“这个向量是什么意思”。
例如 `[3,4]` 与 `[30,40]` 比例相同、整体尺度不同，除以各自 RMS 后会几乎重合；
`[3,4]` 与 `[4,3]` 的特征比例不同，不能因为做了归一化就变成同一个向量。

归一化为后续计算提供经过尺度调整的输入，是模型数值与优化设计的一部分，
不是独立的语义识别器，也不保证任意网络都稳定。分母依赖输入，反向时也要经过分母的计算，
不能把它当成与 x 无关的固定常数。乘在后面的 weight 则给模型学习逐特征缩放的空间。

```text
RMS(x) = sqrt(mean(x_i ** 2))
归一化结果 = x / sqrt(mean(x_i ** 2) + eps)
模块输出 = 归一化结果 * weight
```

### 5.3 操作

在 `scripts/inspect_rmsnorm.py` 中构造：

```python
x = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
denominator = torch.sqrt(mean_square + 1e-5)
normalized = x / denominator
```

打印并检查：

| 张量 | 形状 | 数值 |
| --- | --- | --- |
| 输入 | `[2, 2]` | `[[3,4],[30,40]]` |
| 均方 | `[2, 1]` | `[[12.5],[1250]]` |
| 分母 | `[2, 1]` | 约 `[[3.5355],[35.3553]]` |
| 归一化 | `[2, 2]` | 两行均约 `[0.8485,1.1314]` |

`keepdim=True` 保留长度为 1 的最后一维，使每行的两个特征共用自己的分母。
第二行整体放大 10 倍，RMS 也约放大 10 倍，所以归一化后几乎相同。
这是整体倍数的例子，不表示任意两个向量归一化后都会相同；eps 也使严格尺度不变性变成近似。


分母必须保留最后一个长度为 1 的轴：[B,T,1] 才能让每个位置使用自己的分母。广播从右向左对齐，不能把删掉的中间轴自动猜回来。更多 API 解释见文末 QA。

### 5.4 合同

在 `src/minimind_lab/models/rmsnorm.py` 实现：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

初始化保存 eps，并用 `nn.Parameter(torch.ones(dim))` 创建 weight。
eps 不是可训练参数；weight 是每个特征的可训练缩放系数，初始为 1。

前向按照以下顺序实现：

```text
保留输入 x
-> x_float = x.float()
-> 沿最后一维计算平方均值
-> 加 eps 并开平方
-> x_float 除以分母
-> 乘 weight
-> 转回 x.dtype
```

对 `[B,T,D]` 输入，分母为 `[B,T,1]`，weight 为 `[D]`。
分母由每个位置的输入独立计算，weight 则由所有位置共享。这两种缩放不能混为一谈。
内部 float32 计算减少低精度平方与归一化的数值风险，不意味着消除了所有极端输入风险。
当前合同面向 float32/float16 的模型特征，不能据此声称保留 float64 的内部计算精度。

### 5.5 核心实现

在 models/rmsnorm.py 写入：

~~~python
import torch

from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        scale_x = x_float / torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = self.weight * scale_x
        return output.to(dtype=x.dtype)
~~~

### 5.6 手算测试

`test_rmsnorm_forward_matches_manual` 使用：

```python
x = torch.tensor([[[3.0, 4.0], [30.0, 40.0]]])
```

在 `no_grad()` 中用 `weight.copy_(torch.tensor([2.0, 0.5]))` 设置参数。
构造手算参考时用 Python 的 `math.sqrt` 计算：

```text
y1 = sqrt(12.5 + eps)
y2 = sqrt(1250 + eps)
expected = [[[2 * 3 / y1, 0.5 * 4 / y1],
             [2 * 30 / y2, 0.5 * 40 / y2]]]
```

检查 shape `[1,2,2]`、dtype float32，并用 `assert_close` 比较数值。
weight 特意不用全 1，防止实现忘记乘 weight 时仍通过。

### 5.7 梯度测试

`test_rmsnorm_gradients_match_torch` 创建自己的 RMSNorm 和 `nn.RMSNorm`，
两者都显式设置 `eps=1e-5`，并将 weight 统一为 `[2,0.5]`。
不能依赖不同实现的默认 eps 恰好一致。

准备两份独立输入：

```python
x_manual = torch.tensor([[3.0, 4.0], [30.0, 40.0]], requires_grad=True)
x_reference = x_manual.detach().clone().requires_grad_(True)
```

分别前向、求和、backward，然后对比三项：输出、weight.grad、两份输入的 `.grad`。
`detach()` 使参考输入不连接到第一条计算图，`clone()` 分离存储，再开启该输入的梯度观察。
这里不应让两条路径共同往一个叶子输入的 `.grad` 累加。

输入梯度模拟 RMSNorm 向前面的 Embedding 传回训练信号；weight 梯度用于学习该模块的缩放。
测试输入没有交给 optimizer，即使计算了梯度，也不会被更新。

### 5.8 边界测试

| 测试 | 输入 | 检查 |
| --- | --- | --- |
| `test_rmsnorm_zero_input_is_finite` | 全零 `[2,4,3]` | shape 不变、全部有限、数值全零 |
| `test_rmsnorm_preserves_float16_dtype` | float16 的 `[1,1,2]` | shape/dtype 保留、全部有限、与 float32 计算后转回的结果一致 |

全零输入验证 eps 避免 `0/0`。判断有限使用 `torch.isfinite(output).all()`。
零值比较要使用 `assert_close(output, torch.zeros_like(x))`，只比较 shape 不能证明数值为零。

类型测试的参考 `model(x.float()).to(dtype=x.dtype)` 复用了当前实现，只验证类型转换行为，
不作为独立的公式正确性证据。公式与梯度由前面的手算和 PyTorch 对照保护。

### 5.9 排错

本步骤出现过的错误和处理方式如下，复练时可对照定位：

| 错误 | 原因 | 修正 |
| --- | --- | --- |
| `x.dtype = torch.float32` 报只读属性错误 | dtype 属性不能赋值 | `x_float = x.float()`，最后返回原 dtype |
| 单独执行 `x.pow(2).mean(...)` | 没接结果，也没修改 x | 接收结果复用，或删掉重复计算 |
| 标注 `x: torch.tensor` | tensor 是创建函数，不是类型 | 使用 `torch.Tensor` |
| `model.weight = [2,0.5]` 报参数类型错误 | 试图用列表替换注册参数 | `no_grad()` 中用张量 `copy_` |
| 断言输出 shape 为 `[2,2]` | 漏掉输入最外层 batch 维 | 输入三层括号，输出为 `[1,2,2]` |
| 把第一行乘 2、第二行乘 0.5 | 把逐特征参数误当逐位置参数 | 每个位置的第一、第二特征分别乘 2、0.5 |
| 导入 `inspect_linear.manual_output` 失败 | 引入了不需要的脚本和包路径 | 删除导入，在当前测试自己构造预期值 |
| 只比较零输入输出的 shape | 断言没有覆盖数值要求 | 加上与全零张量的数值比较 |

运行 `uv run pytest -q tests/test_rmsnorm.py`，对应前向、梯度、零输入和精度四类测试。
参考：[实现](../implementation/src/minimind_lab/models/rmsnorm.py)、
[测试](../implementation/tests/test_rmsnorm.py)。

## 6. SwiGLU：逐位置特征加工

### 6.1 目的

本步实现逐位置前馈网络。它在激活实验的基础上增加并行分支，
进一步学习特征组合，不负责不同 token 位置之间的信息交流。

先验证计算链，再验证它不会混合位置。前面的 Linear 与激活观察将用于解释这里的两个分支。

### 6.2 信息加工

“SwiGLU 加工当前位置的特征，而跨位置的信息交流留给 Attention”是本节的核心理解。
更准确地说，FFN 不主动读取其他位置的向量，但它处理的当前位置向量已经可能包含上游 attention 汇总的信息。
所以逐位置计算不等于处理孤立的词，也不能把模型能力拆成“attention 只传话、FFN 单独完成所有推理”。

例如“苹果”的向量已经根据上下文获得“吃水果”或“购买手机”的相关信息时，
FFN 可以进一步加工这些特征。这个例子是帮助理解的示意，不代表训练时预先指定了“水果神经元”。
在原始 Transformer 中，FFN 就是对每个位置使用相同参数的非线性网络；
参见[原论文第 3.3 节](https://arxiv.org/html/1706.03762v7#S3.SS3)。

### 6.3 宽度与门控

学习者提出的直觉是：gate、up 升维，可能提供更高维的特征与语义分辨能力。
可以保留“更宽提供更多计算通道”这一点，但需要把升维、非线性、训练三件事分开。

```text
[x1,x2] -> [x1,x2,x1+x2,x1-x2]
```

这次线性升维产生了更多输入组合，却没有凭空获得新输入信息。
如果后面只接线性降维，仍可合并成一个线性层；非线性与门控使不同输入条件产生不同响应，
训练才决定这些通道最终学到哪些有用模式。中间宽度更大也意味着参数和计算开销增加，不保证越宽越好。

在 SwiGLU 中，可把 up 理解为提供中间特征，gate 经 SiLU 后提供输入相关的调节，
两支相乘后再由 down 组合回主表示。这个分工是解释方式，不是人为固定的语义标签。
gate 和 up 都有自己的可训练参数，模型共同学习它们如何配合。
结构定义与实验来源见[GLU 变体论文](https://arxiv.org/abs/2002.05202)。

从 H 返回 D 也不是把加工“撤销”：中间已发生非线性与乘积，返回统一的 D 后，
结果才能继续传给后续模块并与原表示做残差相加。

### 6.4 合同

在 `src/minimind_lab/models/feedforward.py` 实现：

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

使用三层独立的 `nn.Linear`，都设置 `bias=False`，激活使用 `nn.SiLU()`：

```text
                   +-> gate_proj -> SiLU --+
x [B,T,D] ---------+                       * -> down_proj -> [B,T,D]
                   +-> up_proj ------------+

gate_proj：D -> H
up_proj：  D -> H
两分支逐元素相乘：[B,T,H]
down_proj：H -> D
```

gate 与 up 都接原始 x，不要串联。乘积是逐元素相乘，不是 `@` 矩阵乘法。
SiLU 的定义为 `z * sigmoid(z)`，门控输出不限定在 `[0,1]`，不是离散开关。

### 6.5 核心实现

在 models/feedforward.py 写入：

~~~python
from torch import nn

import torch

class SwiGLU(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
~~~

### 6.6 形状检查

在 `scripts/inspect_swiglu.py` 中用 `D=4`、`H=8` 和 `[2,3,4]` 随机输入试跑，输出为 `[2,3,4]`。
运行 `uv run python scripts/inspect_swiglu.py` 已正常退出。
形状通过只说明投影维度可以衔接，不能证明激活与乘积的顺序正确。
当前类接口是教学拆分，不直接依赖官方配置类，不等于官方完整 FeedForward 的直接复刻。

### 6.7 前向测试

在 `tests/test_feedforward.py` 创建 `test_swiglu_forward_matches_manual`，使用 `SwiGLU(2,3)`。
为三层投影设置固定参数，避免用随机参数猜测结果：

```python
gate_weight = torch.tensor([[1.0,0.0],[0.0,1.0],[1.0,-1.0]])
up_weight = torch.tensor([[2.0,0.0],[0.0,1.0],[1.0,1.0]])
down_weight = torch.tensor([[1.0,0.0,1.0],[0.0,1.0,-1.0]])
```

在 `no_grad()` 中逐个 `copy_` 到模型参数，输入 `[[[1.0,2.0]]]`。
不用模型的 forward 构造参考，而是分步骤执行：

```text
gate 投影 = x @ gate_weight.T -> [1,2,-1]
up 投影 = x @ up_weight.T -> [2,2,3]
SiLU(gate) * up -> [a,b,c]
down 投影 -> [a+c,b-c]
```

其中 `a=SiLU(1)*2`、`b=SiLU(2)*2`、`c=SiLU(-1)*3`。
最终输出约为 `[[[0.6553,4.3300]]]`，精确参考应使用实际运算或公式，不能用四位小数强求默认容差通过。

当前前向测试使用 `nn.SiLU()` 计算参考激活，重点验证投影与组合顺序。
这也属于分步骤参考计算，不必改成逐标量计算才称为手动验证。
若需更独立，可用 `gate * torch.sigmoid(gate)` 或标量 `z/(1+math.exp(-z))` 构造参考。

### 6.8 梯度测试

`test_swiglu_gradients_match_reference` 沿用固定权重，同时准备独立参考参数：

```python
gate_reference = gate_weight.detach().clone().requires_grad_(True)
```

up、down 同样处理。两份输入也用 detach、clone 创建独立叶子张量。
模型参数通过 `no_grad()` 中 copy_ 统一数值，不与参考参数共用对象或存储。

模型路径调用 `model(x)`，参考路径使用矩阵乘法、`gate * sigmoid(gate)` 和逐元素乘积。
分别将输出求和并 backward，然后用 `assert_close` 对比：

- 两份前向输出。
- 两份输入的 `.grad`。
- gate、up、down 三组模型参数与参考参数的 `.grad`。

参考权重已经是张量，所以访问 `gate_reference.grad`，不是 `gate_reference.weight.grad`。
当前固定输入和求和目标验证的是这组条件下的梯度，不保证每个参数元素都非零；
以后扩大覆盖时可加入更多输入和不同的输出加权目标。

这一步先对照前向与梯度，随后补一条位置扰动测试，避免把局部 shape 当成结构行为。

### 6.9 行为测试

`test_swiglu_does_not_mix_positions` 使用同一个 `SwiGLU(4,8)` 处理 `[2,3,4]` 输入。
通过 `x_changed = x.clone()` 复制输入，仅对 `x_changed[0,1]` 加 10，再分别前向。
比较第一个样本的位置 0、2，以及第二个样本的全部输出，均与原输入结果一致。
两次前向之间不更新参数，也不要求被修改位置一定产生不同输出。

本项验证 FFN 的逐位置计算特征，而不是上下文模型的因果性：FFN 输入仍可能包含上游 attention 的信息。
运行 uv run --locked pytest tests/test_feedforward.py -q，三类行为都应通过。

### 6.10 排错

第一次将激活写为 `SiLU(gate_proj(x) * up_proj(x))`，相当于先相乘再激活。
SwiGLU 应为 `SiLU(gate_proj(x)) * up_proj(x)`，仅激活 gate 分支后相乘。
这两个表达式的 shape 相同，但数值与梯度通常不同，需要数值测试而不能仅看 shape。

三层 Linear 最初没有设置 `bias=False`，默认会多出偏置参数；当前已与本次合同统一。
参数名从 `immediate_size` 改为 `intermediate_size`，明确指中间层维度。
模块底部的模型创建与打印移入独立观察脚本，避免导入模型类时自动运行实验。

测试中还出现过以下问题：

| 问题 | 原因与修正 |
| --- | --- |
| `nn.SiLU(tensor)` 得到模块对象 | 构造模块不是执行前向；使用 `nn.SiLU()(tensor)` 或先创建实例再调用 |
| down 投影未转置 | 参数为 `[D,H]`，行向量计算应乘 `[H,D]` 的转置 |
| 梯度测试忘记复制参数 | 模型仍是随机初始化，不能与固定参考权重比较 |
| 用 `sigmoid(gate)` 代替 SiLU | 漏乘 gate，正确为 `gate * sigmoid(gate)` |
| 设置 requires_grad 后直接检查梯度 | 还需执行 backward，才会填充叶子张量的 `.grad` |
| 参考参数访问 `.weight.grad` | 参考对象本身是权重张量，直接访问 `.grad` |

参考：[实现](../implementation/src/minimind_lab/models/feedforward.py)、
[观察脚本](../implementation/scripts/inspect_swiglu.py)。
正式测试见 [test_feedforward.py](../implementation/tests/test_feedforward.py)。

## 7. 残差：保留原表示与更新分支

### 7.1 直接路径

残差使用 `x+F(x)`，分支不必从零替换整份表示，而是学习一份叠加到现有表示上的调整。
直接路径与分支路径在前向和反向中共同作用；它并不意味着原输入每个特征都原样保留在最终输出里。
如果分支与输入方向相反，两者仍可能抵消。额外梯度路径也不是梯度永远不会消失或爆炸的保证。

当前先用 `F(x)=2*x` 观察直接路径；后面再装配 `x+SwiGLU(RMSNorm(x))`。
这一结构对应完整 MiniMind block 的前馈残差部分，不等于已经实现完整 Decoder。

### 7.2 数值观察

先在 `scripts/inspect_residual.py` 中做两条独立路径，理解 `y=x+F(x)`。
使用两份相同的浮点输入 `[-1,2]`，分别开启 requires_grad，设教学分支 `F(x)=2*x`：

```text
无残差：y=2*x，输出 [-2,4]
有残差：y=x+2*x，输出 [-3,6]
```

各自用输出求和作为 loss 并 backward，预测无残差输入梯度为 `[2,2]`，有残差为 `[3,3]`。
多出的 1 来自直接的 x 路径。它解释梯度为何有额外通路，不是“所有深层网络的梯度都能稳定”的保证。
本次执行 `uv run python scripts/inspect_residual.py`，输出正是 `[-2,4]`、`[-3,6]`，
输入梯度分别为 `[2,2]`、`[3,3]`。观察脚本与下面的模块测试分别验证直觉和正式装配，不按累计测试数判断理解程度。

最初打印 `plain_output.grad` 得到 None 与非叶子警告。
本实验要观察输入导数，因此改为打印 `x_residual.grad`，不是要求保存输出梯度。
`x` 在此图中是直接创建的叶子；真实模型中的模块输入可能是上游计算产生的非叶子。
详细规则见[通用笔记的梯度部分](../../../../model_architecture_and_algorithms/01_model_architecture/foundations/03_LINEARITY_NONLINEARITY_AND_ACTIVATIONS.md)。
脚本见 [inspect_residual.py](../implementation/scripts/inspect_residual.py)。

### 7.3 前馈残差块

已在 `models/block.py` 实现 `ResidualFeedForward(hidden_size, intermediate_size, eps=1e-5)`。
它复用已验收的 RMSNorm 和 SwiGLU，前向为 `x + SwiGLU(RMSNorm(x))`：

```text
原表示 x -------------------------------+
   |                                    |
   +-> RMSNorm -> SwiGLU -> 分支调整 -----+-> 相加 -> 新表示 y
```

归一化作用在分支输入，直接路径保留原 x，不要把两者都替换成归一化输入。
首先检查 `[B,T,D]` 输入输出形状与原输入内容，再检查分支归零时的恒等映射和直接梯度路径。
这是 MiniMind 前馈残差部分的教学拆分，没有 attention，不是完整 Transformer block。
当前两项专门测试已通过，分别保护直接路径与正常分支连接。

#### 7.3.1 实践与验证

首先构造 `ResidualFeedForward(4, 8)` 和 `[2,3,4]` 的输入。在 `no_grad()` 中将
`block.swiglu.down_proj.weight` 清零。由于最后一个投影无偏置，分支输出为零，
因此应有 `output == x`。再令输入需要梯度，对 `output.sum()` 反向，输入梯度应全为1。
前一个断言验证原表示能向前通过，后一个验证直接路径能传回学习信号；这不要求全部参数梯度为零。

零分支不能验证归一化和前馈的正常连接，所以第二个测试保留正常权重，先保存输入快照，
再分步计算 `normalized = block.rms_norm(before)`、`delta = block.swiglu(normalized)`、
`expected = before + delta`。调用 block 后，分别比较输出与 expected、输入与 before。
此处复用组件作为参考，是在验证组装；组件内部公式由此前独立测试负责。

#### 7.3.2 排错记录

最初将 `before = x.detach().clone()` 放在 `block(x)` 之后，虽然测试通过，
却无法证明输入未被修改：若发生修改，快照已经记录了修改后的值。
修正为在调用之前保存快照，再执行参考计算和待测操作。检查操作是否改变状态，必须先保存旧状态。

代码见 [block.py](../implementation/src/minimind_lab/models/block.py)，
测试见 [test_block.py](../implementation/tests/test_block.py)。

在 models/block.py 中先写下面部分；此时不需要导入第四周的 Attention：

~~~python
import torch
from torch import nn
from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.feedforward import SwiGLU

class ResidualFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps:float = 1e-5
    ):
        super().__init__()
        self.swiglu = SwiGLU(hidden_size, intermediate_size)
        self.rms_norm = RMSNorm(hidden_size, eps)


    def forward(self, x: torch.Tensor):
        return self.swiglu(self.rms_norm(x)) + x
~~~

运行 uv run --locked pytest tests/test_block.py -q。直接路径成立后，再把尺寸和初始数值统一起来。

## 8. 配置：统一组件尺寸

配置负责描述尺寸，组件负责计算。先在
[config.py](../implementation/src/minimind_lab/models/config.py) 中用 dataclass 定义 ModelConfig：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| vocab_size | 6400 | 已加载的官方 tokenizer 词表大小 |
| hidden_size | 64 | 本地教学用隐藏宽度，不是官方模型规模 |
| intermediate_size | 128 | 本地教学用前馈中间宽度 |
| rms_norm_eps | 1e-5 | 当前归一化数值合同 |

dataclass 生成构造方法，不会自动保证值合法。用 `__post_init__` 检查四项是否大于0，
否则抛出包含字段名的 ValueError。这里仅实现约定的非正数检查，不宣称覆盖类型、NaN 等所有非法输入。
不要求 intermediate_size 大于 hidden_size：升维是当前选择，不是矩阵计算成立的必要条件。

[配置测试](../implementation/tests/test_config.py) 对四个字段分别传入0和-1，组成8条参数化用例。
`pytest.raises(ValueError, match=field)` 同时检查异常类型和报错字段；
`ModelConfig(**{field: value})` 将字典展开成对应关键字参数。
正常路径在一个测试函数中验证默认值与自定义值，后者特意使用 hidden_size=8、intermediate_size=4，
保护“不强制升维”的约定。这是第三周最初的测试范围，当前配置测试还包含第四周字段。

最初的 res_norm_eps 拼写已修正为 rms_norm_eps；它是 RMSNorm 的参数，不是残差参数。

本周的最小版本如下。当前仓库已经扩展为八个字段；复练时不要用这个四字段版本覆盖实际文件：

~~~python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 64
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "rms_norm_eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
~~~

在当前仓库运行 uv run --locked pytest tests/test_config.py -q。四字段练习副本仅运行相应基础测试，不把缺少后续字段当成本周失败。

## 9. 初始化：确定参数起点

### 9.1 初始化规则

构造出的参数有初值，但不同组件可能使用不同的默认规则。此步集中定义本次教学初始化，
不把它说成已经对齐官方完整模型：

| 参数 | 规则 | 理由 |
| --- | --- | --- |
| TokenEmbedding 和 Linear 的 weight | 正态分布，mean=0、std=0.02 | 不同通道拥有不同起点，并控制初始尺度 |
| Linear 的 bias（存在时） | 全0 | 初始不额外平移 |
| RMSNorm 的 weight | 全1 | 初始不额外改变归一化后的逐特征缩放 |

0.02 是当前练习约定，不是所有模型的最优值。实现见
[initialization.py](../implementation/src/minimind_lab/models/initialization.py)。
initialize_weights 只按当前模块的类型处理自身参数，不手动深入整个 block。
随后在组件创建后、第一次 forward 前执行：

```python
embedding.apply(initialize_weights)
block.apply(initialize_weights)
```

这段依赖前面已创建的两个组件。apply 递归访问子模块，再访问模块自身，把每个模块交给回调。
它不是专门的初始化机制；类型判断由我们的函数完成，参数数值修改由 nn.init 完成。
传入函数本身而不是 initialize_weights()，是让 apply 在遍历时提供当前模块作为参数。
未匹配的 SwiGLU、ResidualFeedForward 等容器不处理。

nn.init 不是唯一修改参数的方法。`with torch.no_grad(): weight.normal_(...)` 同样可以，
而 nn.init 已在内部处理无梯度修改。重要的是修改已有 Parameter 的数值，不随意替换参数对象。
不要在 forward 或训练循环中重复初始化，也不要在加载训练权重后重置它们。

最初误把 ResidualFeedForward 当作“全部设为1”的对象，还访问了不存在的 swiglu.glu。
修正为只对 RMSNorm 权重设1，三个 Linear 分别保留正态规则。
这也避免 apply 处理子模块后，再被容器分支覆盖。

### 9.2 核心实现

写入 models/initialization.py：

~~~python
from minimind_lab.models.rmsnorm import RMSNorm
from minimind_lab.models.embedding import TokenEmbedding

from torch import nn

def initialize_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, TokenEmbedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
~~~

### 9.3 对照验证

[初始化测试](../implementation/tests/test_initialization.py) 共3条：

1. 先将 Linear bias 与 RMSNorm weight 填成2，再验证初始化后分别为0和1。
2. 固定随机种子，验证无偏置 Linear 的正态权重与参考结果一致。
3. 同样验证 TokenEmbedding 的正态权重。

第一项先设置错误初值，避免 RMSNorm 默认全1让遗漏的初始化蒙混过关。
准备值时在 no_grad 中直接 fill_，不使用绕过部分 autograd 保护的 .data。
固定种子对照时先创建模块，再设置 seed 生成参考值，重新设置同一个 seed 后调用待测函数。
两条路径从相同随机状态开始，验证是否使用约定分布；不要求小矩阵的经验均值和标准差恰好等于分布参数。

运行 uv run --locked pytest tests/test_initialization.py -q。初始化完成后，才能开始组合前向；不要在每次 forward 中重新初始化。
## 10. 组件组合与参数统计

### 10.1 参数预算

在 [inspect_model_components.py](../implementation/scripts/inspect_model_components.py) 中，
用配置 V=20、D=8、H=16 创建组件，从形状推导再用 parameters() 和 numel() 统计：

| 部分 | 公式 | 数量 |
| --- | --- | --- |
| Embedding | V × D | 160 |
| RMSNorm | D | 8 |
| SwiGLU 三个无偏置投影 | 3 × D × H | 384 |
| 前馈残差块 | D + 3 × D × H | 392 |
| 组合 | V × D + D + 3 × D × H | 552 |

残差相加不引入参数，batch 和序列长度也不进入这些参数量公式。
最后分别断言实际数量等于配置公式，而非只写死总数552，这样调整配置后仍能核对结构。
脚本参数全部可训练；未来冻结参数时，总参数量与可训练参数量需要分别统计。

### 10.2 链路

此前的组件测试验证局部公式，这一步验证它们能否在同一配置和初始化规则下协作：

```text
ModelConfig -> 创建组件 -> apply 初始化
input_ids [2,3] -> Embedding [2,3,8] -> ResidualFeedForward [2,3,8]
               -> output.square().mean() -> backward -> 各组参数梯度
```

平方均值只是产生反向信号的测试目标，不是语言模型的 next-token 目标。
没有执行 optimizer.step，没有完成一次真实训练更新。
输入中两次出现的 ID 2 得到相同输出，符合当前没有位置编码和 attention、仅逐位置计算的结构。

### 10.3 自动检查

先跑观察脚本，再将关键预期整理到
[test_model_components.py](../implementation/tests/test_model_components.py)，不复制大量打印：

- 输出 shape 正确，所有元素有限；
- 反向后每个参数的 grad 存在，所有梯度元素有限；
- Embedding 和残差块的实际参数量分别等于配置公式。

正式测试使用 V=8、D=8、H=16，输入 ID 最大为5，因此词表覆盖有效。
它的参数总数为64+392=456，不应套用观察脚本 V=20 时的552。

### 10.4 排错

最初写成 `assert torch.isfinite(param.grad)`。
isfinite 逐元素返回布尔张量，assert 却需要一个明确的真假值，因而报出
“Boolean value of Tensor with more than one value is ambiguous”。

修正为先检查 `param.grad is not None`，再检查 `torch.isfinite(param.grad).all()`。
两者分别保护梯度存在与全部元素有限。零梯度也是有限值，不要求每个元素非零。
前向已经使用了 .all()，此次将同一个聚合原则应用到梯度检查。

### 10.5 边界

smoke 保护组件接线和基本数值通路，不替代前面的独立数值参考测试，
也不证明模型能学习语言、具有上下文能力或训练稳定收敛。
初始化递归已在组合中实际运行，但这项 smoke 没有逐个断言每个子模块的初始化分布。

### 10.6 说明

完整 causal attention、RoPE、GQA、Decoder、语言模型输出头和生成留到第四周。
本周不额外启动预训练，也不把 MLP 当成 attention 的替代品。

### 10.7 完整运行脚本

下列代码写入 scripts/inspect_model_components.py。它把本周所有组件接起来，不依赖前面交互环境里的变量：

~~~python
from minimind_lab.models.block import ResidualFeedForward
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.embedding import TokenEmbedding
from minimind_lab.models.initialization import initialize_weights

import torch

config = ModelConfig(20, 8, 16)

embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
block = ResidualFeedForward(config.hidden_size, config.intermediate_size, config.rms_norm_eps)

input_ids = torch.tensor([[1, 2, 3], [4, 2, 5]])
print(f"input_ids.shape: {input_ids.shape}")

# 初始化模块参数权重
embedding.apply(initialize_weights)
block.apply(initialize_weights)

# forward
embeded = embedding(input_ids)
print(f"embeded: {embeded}")
print(f"embeded.shape: {embeded.shape}")

output = block(embeded)
print(f"output: {output}")
print(f"output.shape: {output.shape}")

loss = output.square().mean()
loss.backward()

for name, param in embedding.named_parameters():
    print(f"{name}: shape: {param.shape} counts: {param.numel()}")
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

total_embedding_params = sum(param.numel() for param in embedding.parameters())
print(f"total_embedding_params: {total_embedding_params}")

for name, param in block.named_parameters():
    print(f"{name}: shape: {param.shape} counts: {param.numel()}")
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()

total_block_params = sum(param.numel() for param in block.parameters())
print(f"totaol_block_params: {total_block_params}")

expected_embedding = config.vocab_size * config.hidden_size
expected_block = (
    config.hidden_size
    + 3 * config.hidden_size * config.intermediate_size
)

assert total_embedding_params == expected_embedding
assert total_block_params == expected_block
~~~


~~~powershell
uv run --locked python scripts/inspect_model_components.py
uv run --locked pytest tests/test_model_components.py -q
~~~

## 11. 验收与衔接

### 11.1 参考边界

本轮参照冻结提交 `f659b55761b754d306bd140573493a6543cafd7f`，
官方相关实现集中于 [model_minimind.py](../../../../../minimind/model/model_minimind.py)。

| 当前内容 | 官方对应 | 对照边界 |
| --- | --- | --- |
| TokenEmbedding | 模型使用 `nn.Embedding` | 当前查表与普通稠密梯度已对齐，整体初始化尚未对齐 |
| RMSNorm | `RMSNorm.norm`、`RMSNorm.forward` | 官方用乘 rsqrt，我们用除 sqrt；都内部 float32，乘 weight 后转回输入类型 |
| SwiGLU | `FeedForward`，默认激活 silu | 三层无 bias 投影和两分支乘积已实现，shape、前向、梯度与逐位置行为通过 |
| 残差 | `MiniMindBlock.forward` 的两处相加 | 数值观察、前馈残差装配与两项测试完成；完整 attention 残差装配留到第四周 |
| 模型配置 | `MiniMindConfig` | 第三周教学 dataclass 起初实现4个字段与非正数检查，不是完整官方配置 |
| 初始化 | 官方模型构造及框架初始化路径 | 当前显式教学规则已实现并测试，未宣称与官方完整模型初始化一致 |

官方 RMSNorm 类的默认 eps 为 `1e-5`，但 `MiniMindConfig.rms_norm_eps` 默认是 `1e-6`。
完整模型对齐时要检查调用位置传入的实际配置，不能只比较构造函数默认值。
官方 FeedForward 的 intermediate_size 来自配置；小实验的 H=8 只用于验证计算。

### 11.2 回归命令


~~~powershell
uv run --locked pytest tests/test_embedding.py tests/test_rmsnorm.py tests/test_feedforward.py tests/test_block.py tests/test_config.py tests/test_initialization.py tests/test_model_components.py -q
~~~

历史第三周完成时全仓 101 项通过。现在代码和用例已扩展，不应把旧数量当作本次命令预期。核心组件及观察实验由学习者在指导下完成，助手协助说明、核验和文档整理。

本周只能确认逐位置组件、参数注册与梯度通路。第四周需要 Attention 让位置间交换信息，再接完整 Decoder、词表输出与预测目标。没有必要在进入下一周前重复所有同类测试。

## 12. 学习问答

### 12.1 为什么有 torch 函数、Tensor 方法和原地操作？

`torch` 提供创建和计算张量的函数，Tensor 自身也有方法。
例如 `torch.square(x)`、`x.square()` 和 `x.pow(2)` 都能表达这里的逐元素平方。
选择函数形式还是方法形式不是“能否训练”的区别，关键是运算语义与计算图是否相同。

`x * y` 通常逐元素相乘，支持广播；`x @ w.T` 则做矩阵乘法，对相接的维度求和。
因此 SwiGLU 两分支用 *，Linear 的特征组合用 @，不能只看两边都是张量就互换。

以下小实验把“返回结果”与“修改原对象”区分开：

```python
import torch

x = torch.tensor([2., 3.])
squared = x.square()
torch.testing.assert_close(x, torch.tensor([2., 3.]))
torch.testing.assert_close(squared, torch.tensor([4., 9.]))

target = torch.zeros_like(x)
target.copy_(squared)
torch.testing.assert_close(target, squared)
assert target.numel() == 2
```

square、pow、sqrt 等通常返回结果，单独调用却不接结果，不会自动改变 x。
copy_、fill_、normal_ 的末尾下划线表示原地操作，修改已有张量的内容。
这个命名提示不能替代具体 API 合同，尤其不要把所有方法都猜成原地行为。
修改需要梯度的叶子参数时，测试准备通常放入 no_grad；正常 forward 不应随意原地改参数。

ones_like、zeros_like 便于按已有张量的 shape、dtype、device 创建参考值。
empty_like 只分配相应存储，不保证初值内容；初始化测试必须随后把所有元素写成有效参考值。

### 12.2 mean、dim=-1、keepdim 与广播是什么关系？

mean 和 sum 属于归约：把选中维度的多个值聚合。
`x.mean()` 不指定维度时对全部元素平均；`x.mean(dim=-1)` 则对每个位置的最后一维平均。
在 [B,T,D] 上，后者得到 [B,T]；加 keepdim=True 后得到 [B,T,1]，
保留的1不是新增特征，而是标明原来被归约的轴所在位置。

广播从末尾对齐维度，对应维度相等或其中一个为1时可以扩展使用。
所以 [B,T,D] / [B,T,1] 让同一个位置的 D 个特征使用自己的分母。
去掉 keepdim 后，[B,T] 不一定能正确广播；即使尺寸碰巧允许，也可能沿错误的轴匹配。
不能仅凭“没报 shape 错误”判断公式正确。

```python
import torch

x = torch.tensor([[3., 4.], [30., 40.]])
mean_square = x.square().mean(dim=-1, keepdim=True)
assert mean_square.shape == (2, 1)
torch.testing.assert_close(mean_square, torch.tensor([[12.5], [1250.]]))
normalized = x / torch.sqrt(mean_square + 1e-5)
assert normalized.shape == x.shape
assert torch.isfinite(normalized).all()
```

运行 `uv run python scripts/inspect_rmsnorm.py`。
参考：[当前脚本](../implementation/scripts/inspect_rmsnorm.py)。

### 12.3 为什么设置参数要放在 no_grad 中？

`no_grad()` 中复制数值是准备测试条件，不是需要求导的模型计算。
叶子参数需要梯度时，普通梯度模式下直接原地 `copy_` 会被 PyTorch 阻止。

```python
with torch.no_grad():
    model.weight.copy_(desired_weight)
```

这保留了 Parameter 对象及其可训练属性，仅不记录本次复制操作。
创建 optimizer 后再替换整个 Parameter 还可能让 optimizer 持有旧对象，所以修改数值优先 copy_。

比较两条计算路径时，设置相同参数的目的不是让结果强行相同，而是排除初始化差异，
把“实现是否正确”作为要检查的变量。参数一致后，计算链路不同仍会得到不同结果，测试才有意义。

### 12.4 输入为什么也要梯度，中间张量为什么没有 grad？

| 对象 | 当前练习的行为 |
| --- | --- |
| 原始 token ID | 整数编号，不需要梯度 |
| 模型 Parameter | 通常需要梯度，并交给 optimizer |
| 模型产生的浮点特征 | 保留计算图，梯度可继续传回上游，默认不保存自身 `.grad` |
| 梯度测试新建的浮点输入 | 主动设置 `requires_grad=True`，作为叶子张量观察 `.grad` |

“求这个对象的梯度”与“训练这个对象”是两件事。
RMSNorm 的真实输入来自 Embedding 或其他模块，不是不可求导的 token ID；
为了让上游参数学习，必须正确计算损失对这份特征的导数。

这里“叶子”由张量的创建来源决定，不由它在文字描述中是不是“中间输入”决定。直接创建的浮点 x 可以是叶子；Embedding 输出的 x 通常不是叶子。非叶子照样传递梯度，只是默认不保存自身 grad；需要观察时先调用 retain_grad。

### 12.5 assert 与 assert_close 如何选择？

| 写法 | 用途 |
| --- | --- |
| `assert output.shape == expected_shape` | 检查单个真假条件 |
| `a == b` | 得到逐元素布尔张量，不是直接可用的多元素 assert 条件 |
| `assert torch.equal(a, b)` | 检查形状与逐元素完全相等，不要求 dtype 相同 |
| `torch.testing.assert_close(a, b)` | 检查容差内接近，默认也检查 dtype/device 等 |

参数梯度通常是浮点数，预期矩阵也应构造为匹配的浮点类型。
`torch.tensor([1,2])` 默认是 int64；`torch.tensor([1.0,2.0])` 通常是 float32。
更稳妥的做法是显式传 `dtype=model.weight.dtype`，设备测试还需对齐 device。
`assert_close` 自己会抛出异常，不需要在它前面再加 assert。

梯度比较时，浮点运算顺序可能导致微小差异，优先用 `assert_close`。
目前 Embedding 求和测试中的整数梯度可精确表示，使用 equal 也能通过，但不能推广成所有梯度都严格相等。

### 12.6 为什么 apply 可以初始化整个模型？

apply 递归把子模块交给回调，回调根据模块类型处理参数。它不是专用初始化算子；我们传入 initialize_weights 函数而不是调用后的返回值。nn.init 内部按无梯度方式修改数值，也可以用 no_grad 配合 normal_ 实现，但不要替换优化器已经持有的参数对象。

### 12.7 参数遍历与模块遍历有何区别？

`module.parameters()` 迭代给出参数对象；`module.named_parameters()` 给出名称与参数组成的二元组，
因此 `for name, param in module.named_parameters()` 是 Python 的二元组解包。
block 中的名称包含 swiglu.gate_proj.weight 等层级，是注册子模块的结构在遍历中的体现。
这里遍历的是参数，不是每次前向产生的激活张量。

`param.numel()` 返回元素总数。例如 [16,8] 的矩阵有128个元素，
而 `len(param)` 只返回第一维长度16，不能用于统计全部参数。
`sum(param.numel() for param in module.parameters())` 是生成器表达式：
逐个取出参数的元素数量，再由 Python sum 累加，不需要先创建数量列表。

apply 与这两个迭代器处理的对象不同：apply 回调收到整个模块，
可以据其类型决定初始化方式；parameters 收到 Tensor/Parameter，
便于统一统计或交给优化器。两者都利用注册结构，但不能混为同一个接口。

### 12.8 RMSNorm 是正则化吗，归一化后所有向量都一样吗？

这里是均方根归一化，不是向 loss 添加惩罚项的正则化。它调整每个位置的整体尺度，保留不同特征之间的相对结构，并用可学习 weight 再缩放。同方向的倍数向量可能近似重合，不同方向不会因此变成同一个向量。

### 12.9 前馈为什么升维后又降回去？

更宽的中间层提供更多组合通道，非线性和门控使其不能简单折叠成一层 Linear。降回 hidden_size 是为了继续沿主干计算并做残差相加，不是在撤销中间加工。更宽不保证效果一定更好。

### 12.10 为什么初始化对照要重复设置 seed？

模块构造也会消耗随机数。先构造对象，再给参考路径和待测初始化各设置相同 seed，才能保证两次抽样起点一致。小矩阵的样本均值不必精确为零，标准差也不必恰好为 0.02。
