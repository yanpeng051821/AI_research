# LoRA：结构、低秩与参数更新

整理：2026-11-07。本文沿“原线性层 → 新增分支 → 秩 → 初始化 → 梯度与更新 → 挂载与工程选择”展开，保留本轮讨论中的疑问和纠正。

[目录](README.md) · 下一篇：[QLoRA](02_QLORA_QUANTIZATION_AND_TRAINING.md)

## 1. 本篇解决什么问题

- 我们已经有一个开源模型，希望用任务数据改变它的行为，但不一定需要直接更新全部原参数。
- LoRA 的方法是冻结选定层的原权重，用新增的低秩参数表示调整量。
- 本篇重点：新增矩阵怎样接入、形状是否变化、梯度怎样计算、rank 和挂载范围怎样影响配方。
- 在研究链路中，这是“训练配方”的一部分；不回答多少测试案例最合适，也不保证训练后就达到业务可用标准。

### 1.1 SFT 与 LoRA 不是二选一

| 维度 | 回答什么 | 例子 |
| --- | --- | --- |
| 模型起点 | 从哪份权重继续训练？ | 指定 revision 的 Base 或 Instruct 模型 |
| 数据与监督 | 输入什么，用什么答案计算 loss？ | case 信息与初始证据作为输入，类别、根因、建议作为监督 |
| 参数更新方式 | 哪些参数允许改变？ | 全量微调或 LoRA |
| 精度与资源 | 参数怎样保存和计算？ | 浮点基模 LoRA，或量化基模 QLoRA |

- SFT 描述示范数据的监督训练；它可以使用全量更新，也可以使用 LoRA。
- LoRA 不是一个必须位于 SFT 后面的独立阶段，也不要求输入模型先完成某个固定阶段。
- 使用哪种起点，仍由目标能力、基线评测和资源约束决定。

## 2. 从一个普通线性层开始

### 2.1 贯穿例子与符号

我们构造一个将每个 token 从 1024 维映射到 2048 维的线性层。这里讨论标准 `nn.Linear`，不是带特殊权重布局的融合算子。

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| `N` | batch 中的样本数 | 标量 |
| `T` | 序列长度 | 标量 |
| `x` | 该线性层的输入 | `[N, T, 1024]` |
| `W` | 原线性层权重 | `[2048, 1024]` |
| `b` | 原偏置，可不存在 | `[2048]` |
| `y` | 输出 | `[N, T, 2048]` |

按 PyTorch 的存储约定，`weight` 是 `[out_features, in_features]`，前向为：

\[
y=xW^T+b
\]

- batch 和序列维度保留；只有最后一个特征维度改变。
- `W` 不必是方阵，LoRA 也不要求输入、输出特征数相同。
- 后文用 `B` 表示 LoRA 矩阵，用 `N` 表示 batch，避免同一符号混用。

## 3. LoRA 怎样新增分支

### 3.1 两个小矩阵

设 `r=8`，新增：

| 参数 | 映射 | 存储形状 | 是否训练 |
| --- | --- | --- | --- |
| `W` | 1024 → 2048 | `[2048, 1024]` | 本文设置下冻结 |
| `A` | 1024 → 8 | `[8, 1024]` | 是 |
| `B` | 8 → 2048 | `[2048, 8]` | 是 |

标准 LoRA 使用缩放系数 `s=alpha/r`：

\[
\Delta W=sBA,\qquad y=xW^T+b+s(xA^T)B^T
\]

我们并没有先对原 `W` 做分解、丢掉它的一部分；原权重保留，分解表示的是新增的调整量。

### 3.2 形状与数据流

```text
x：[N, T, 1024]
├─ 原分支：x @ W.T + b ─────────────── [N, T, 2048]
└─ LoRA：x @ A.T → [N, T, 8]
                 → 再乘 B.T、再缩放 ─ [N, T, 2048]
两路结果逐元素相加 ────────────────── [N, T, 2048]
```

- 两条分支吃同一个输入，最终输出形状一致，因此可以相加。
- 原分支和 LoRA 分支的中间形状不必一致；A、B、W 自身的形状也不一样。
- “并行”指计算结构有两条路径，不保证硬件同时执行。
- 原线性层对外的输入、输出接口没有变化；整个模型不会变成 8 维模型。
- 本节先不使用 dropout。标准 LoRA dropout 作用于新增分支的输入，训练时仍不改变张量形状。

## 4. 低秩实际限制了什么

### 4.1 秩不是行数，也不是非线性

矩阵的秩是线性无关列向量的最大数量，也等于线性无关行向量的最大数量。关键是“线性无关”，不是“非线性”。

自拟例子：

\[
M=\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad
M\begin{bmatrix}u\\v\end{bmatrix}
=(u+2v)\begin{bmatrix}1\\2\end{bmatrix}
\]

- 第二行是第一行的两倍，第二列也只是第一列的两倍。
- 输出有两个坐标，但所有输出都在 `(1, 2)` 张成的一条直线上。
- 例如能得到 `(3, 6)`，不能得到 `(1, 0)`；矩阵的秩是 1，不是 2。

对比单位矩阵 `[[1,0],[0,1]]`：它能产生两个独立方向，秩为 2。

### 4.2 A 的行和 B 的列分别做什么

为讲清方向，这里把单个输入改写成列向量 `x：[1024,1]`。这与第 3 节的行式数据布局等价，不是模型改变了接口。

\[
z=Ax,\qquad Bz=\sum_{j=1}^{r}z_j B_{:,j}
\]

- **A 的每一行**与输入做内积，计算一个系数 `z_j`。
- **B 的每一列**是一个 2048 维方向，输出由这些方向按系数加权组合。
- B 的列只有在线性无关时，才构成其列空间的一组基；不是所有矩阵的所有行或列天然都是基。
- 这些方向由训练学习，不是人工指定的“分类方向”或“根因方向”。

由于输出调整只能由最多 r 个列方向组合：

\[
\operatorname{rank}(BA)\leq
\min(\operatorname{rank}(B),\operatorname{rank}(A))\leq r
\]

### 4.3 配置 rank 与实际秩

| 说法 | 是否准确 | 原因 |
| --- | --- | --- |
| `r=8` 表示中间有 8 个位置 | 是 | A 输出 8 维，B 接收 8 维 |
| `BA` 的秩一定是 8 | 否 | A、B 可能有线性依赖；初始化时 B 为零，实际秩为 0 |
| LoRA 只能改变 8 次 | 否 | 限制的是独立方向数量，不是更新次数 |
| 模型只能拥有 8 种能力 | 否 | 原分支仍存在，多个层也各自有调整参数 |
| 原 W 也被约束成 rank 8 | 否 | 低秩约束针对调整量，不是原权重 |

### 4.4 为什么经常选择较小的 rank

原权重参数量为 `m*n`，新增参数量为 `r*(n+m)`。下表按 `n=1024、m=2048` 计算，不含偏置：

| 参数化 | 可训练参数量 | 占原 W 参数量 |
| --- | --- | --- |
| 直接训练 W | 2,097,152 | 100% |
| LoRA，r=8 | 24,576 | 约 1.17% |
| LoRA，r=16 | 49,152 | 约 2.34% |
| LoRA，r=64 | 196,608 | 约 9.38% |

- 小 r 减少参数、对应梯度和优化器状态，是一种成本与表达范围的取舍。
- LoRA 作者提出“任务适配所需的权重调整可能具有较低内在秩”的假设，并在论文第 7 节做了经验分析。
- 论文部分 GPT-3 任务中很小的 r 已能取得较好结果，不等于所有任务都应该用 r=1、2 或 8。
- r 增大放宽表示约束，不保证训练成绩或泛化成绩单调提升。
- 我们没有确定测试领域的最佳 rank；它应当与挂载范围、数据、学习率等一起记录和验证。

## 5. 初始化：为什么 A 随机、B 为零

### 5.1 两个矩阵都初始化，只是方式不同

- LoRA 原论文第 4.1 节：A 使用随机高斯初始化，B 初始化为零。
- 官方线性层实现及 PEFT 默认：A 使用 Kaiming uniform，B 为零。
- `nn.Linear` 创建时已自动初始化权重；代码只显式写 `zeros_(B.weight)`，不代表 A 没有初始化。
- 这里描述标准线性层 LoRA；其他模块类型或初始化变体不一定采用同一规则。

初始时：

\[
B=0\Rightarrow \Delta W=sBA=0\Rightarrow y=Wx+b
\]

这保证新增分支初始贡献为零。在原模块状态、输入和其他配置不变的前提下，接入本身不改变输出；不意味着以后训练也不会改变其他任务的行为。

### 5.2 B 为零为什么仍能更新

- B 的参数梯度依赖前向输入 `z=Ax`，不要求 B 自己先非零。
- 第一次反向时，A 的任务梯度经过 B，因此为零。
- 当 B 经 optimizer step 更新后，后续反向可以让 A 得到非零任务梯度。
- A 非零不保证 Ax 非零；还需输出梯度非零，且 batch 中的贡献不完全抵消。
- A、B 若都为零，标准任务梯度无法让这条双线性分支自行启动。
- “第一次 A 的任务梯度为零”不等于其数值绝对不变：若优化器应用权重衰减或已有历史状态，还需检查这些额外更新项。

## 6. 完整反向与参数更新

### 6.1 推导前提

这一节是对前向公式的链式法则推导，不是论文实测。

| 对象 | 定义与形状 |
| --- | --- |
| x | 一个 token 的列向量，`[n,1]` |
| W | 冻结权重，`[m,n]` |
| A、B | 可训练矩阵，`[r,n]`、`[m,r]` |
| z | `Ax`，`[r,1]` |
| y | `Wx+sBz`，`[m,1]` |
| g | 下游传回的 `∂L/∂y`，`[m,1]` |

省略偏置和 dropout，s 固定。先算完整前向，再根据前向时的参数值求梯度，最后统一更新参数。

### 6.2 从相加节点回到两条分支

令 `u=Wx`、`v=Bz`，有 `y=u+s*v`：

\[
\frac{\partial L}{\partial u}=g,\qquad
\frac{\partial L}{\partial v}=sg
\]

- 原分支收到 g，未缩放的 LoRA 输出 v 收到 s*g。
- 相加不是把梯度各分一半。
- 之前取 s=1 时，两路收到相同梯度；保留缩放系数后要把它带入反向。

### 6.3 B、A 的梯度

先反向经过 B，再经过 A：

\[
\frac{\partial L}{\partial B}=sgz^T
\quad[m,r]
\]

\[
\frac{\partial L}{\partial z}=sB^Tg
\quad[r,1]
\]

\[
\frac{\partial L}{\partial A}=s(B^Tg)x^T
\quad[r,n]
\]

- B 前向吃 z，因此它的参数梯度包含 z。
- A 前向吃 x，因此它的参数梯度包含 x；它接收的上游梯度先经过 B。
- 共享参数接收多个 token 的贡献时，这些贡献相加；loss 的平均与 masking 决定权重，不能再任意平均一次。

### 6.4 冻结 W 不等于切断原分支

数学上，W 的参数偏导是 `g*x.T`；但本文设置下不保存或使用这个梯度去更新 W。

如果前面的可训练模块需要梯度，传给 x 的两路贡献是：

\[
\frac{\partial L}{\partial x}=W^Tg+sA^TB^Tg
\quad[n,1]
\]

- `requires_grad=False` 禁止对 W 求参数梯度，不会自动禁用经过 W 的输入梯度。
- 不能为省显存而随意把整个原分支包进 `no_grad()`；那可能切断前面 adapter 所需的梯度路径。
- 梯度是否需要继续传到 x，取决于它是否来自还需要训练的上游模块。

### 6.5 optimizer step 在梯度之后

最简单的 SGD 更新为：

\[
A_{new}=A-\eta\nabla_A L,\qquad
B_{new}=B-\eta\nabla_B L
\]

- `backward()` 计算并累积梯度；`optimizer.step()` 才改变参数。
- 两个梯度都按同一次前向的旧 A、B 计算，不是在一次反向中先更新 B 再拿新 B 求 A 的梯度。
- AdamW 还使用历史统计和权重衰减，不能把所有优化器都简化成上式；见[优化算法](../03_optimization/01_SGD_ADAMW_AND_PARAMETER_UPDATES.md)。

### 6.6 手算一轮，再看第二轮

这是教学标量例子：`x=1、w=1、a=2、b=0、s=1`，目标 `t=0`，`L=(y-t)^2/2`，SGD 学习率 0.1，无权重衰减。

| 项目 | 第一次 | 第二次，使用第一次 step 后的参数 |
| --- | --- | --- |
| a、b | 2、0 | 2、-0.2 |
| y=wx+bax | 1 | 0.6 |
| g=y-t | 1 | 0.6 |
| b 的梯度=gax | 2 | 1.2 |
| a 的梯度=bgx | 0 | -0.12 |
| step 后 a、b | 2、-0.2 | 2.012、-0.32 |

第一次 B 已有梯度并更新；第二次 A 的任务梯度也非零。更新发生在当次反向之后，不必等到“下一次才执行 B 的更新”。

## 7. 最小实现：将公式变成模块

下面是可独立阅读的标准 `nn.Linear` 包装示例，不依赖 PEFT。它保留已加载的原层，不重新随机初始化基模。

```python
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        options = {"device": base.weight.device,
                   "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, rank,
                                bias=False, **options)
        self.lora_B = nn.Linear(rank, base.out_features,
                                bias=False, **options)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


torch.manual_seed(0)
base = nn.Linear(4, 6, bias=False)
layer = LoRALinear(base, rank=2, alpha=2.0)
x = torch.randn(2, 3, 4)
target = torch.randn(2, 3, 6)
optimizer = torch.optim.SGD(
    [p for p in layer.parameters() if p.requires_grad], lr=0.1
)
before_base = base.weight.detach().clone()

for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    output = layer(x)
    assert output.shape == (2, 3, 6)
    if step == 0:
        torch.testing.assert_close(output, base(x))
    loss = (output - target).square().mean()
    loss.backward()
    assert base.weight.grad is None
    if step == 0:
        torch.testing.assert_close(
            layer.lora_A.weight.grad,
            torch.zeros_like(layer.lora_A.weight),
        )
    print(step, loss.item(), layer.lora_A.weight.grad.norm().item(),
          layer.lora_B.weight.grad.norm().item())
    optimizer.step()

torch.testing.assert_close(base.weight, before_base)
```

### 7.1 代码对应的职责

- A、B 作为 Module 属性注册，能被 `parameters()` 收集；不是只在函数里临时创建矩阵。
- `base.requires_grad_(False)` 冻结原层；optimizer 只持有可训练参数。
- 原层、新增层的 device 和 dtype 一致；生产环境的混合精度还需单独配置。
- 示例没有 dropout、融合投影、量化权重或分布式，不是完整训练脚本。
- 本轮完成代码语法检查，并用 NumPy 核对公式和手算数值；当前验证环境没有 PyTorch，未实际执行此代码块。这里不报告 PyTorch 运行通过。

## 8. 从单个层回到整个模型

### 8.1 target_modules 决定挂载范围

PEFT 可通过 `target_modules` 指定目标模块。例如模型确实具有这些名称时，`["q_proj", "v_proj"]` 可匹配 Q、V 投影；不是自动给全部线性层添加 LoRA。

```text
Block 0：Q 投影有自己的 A/B，V 投影有自己的 A/B
Block 1：Q 投影有另一组 A/B，V 投影有另一组 A/B
其他模块是否适配，取决于目标配置和实际匹配结果
```

- 标准设置中每个目标层各自学习，不是全模型共用一对 A、B。
- 原投影若是 `[out,in]` 非方阵，A/B 按其真实维度构造即可。
- 各架构命名和融合方式不同；不能把某模型的目标名称不加检查地套给另一模型。
- 所有目标模块的 LoRA 参数量，应逐层求和。

### 8.2 配方中需要记录什么

| 配置 | 控制对象 | 比较时注意 |
| --- | --- | --- |
| target_modules、目标层范围 | 在哪些位置允许调整 | 列出实际匹配的模块，不只记录配置字符串 |
| r | 每处调整的中间维度与秩上限 | 相同 r 不代表总参数量相同 |
| alpha、缩放规则 | 新增分支输出及梯度尺度 | 固定 alpha 改 r 时，标准 alpha/r 也会改变 |
| dropout | 新增分支训练时的随机丢弃 | 不是 rank，也不是量化 |
| bias、额外可训练模块 | A/B 之外是否还有更新 | “只训练 A/B”要由参数清单证实 |
| 学习率与优化器 | 梯度如何转成更新 | 不以 alpha 代替完整学习率策略 |

## 9. 保存、加载与合并

- adapter 文件主要保存新增参数与配置，不是完整基模；还需原模型的精确身份、tokenizer、模板和加载方式。
- 完整断点恢复还需优化器、调度器、随机状态及进度等；adapter 推理文件不能代替恢复 checkpoint。
- 标准浮点 LoRA 在关闭 dropout 等推理条件下，可以形成 `W_merged=W+sBA`，不再保留独立分支。
- 实数数学上合并前后等价；实际浮点运算仍可能有舍入差异，要比较输出。
- 若基模是量化的，合并与再量化要按实现处理，不能直接把浮点矩阵加到 4-bit 编码上；见下一篇。

## 10. 我们保留的判断与问题

| 已理解的机制 | 不能由此直接推出 | 后续验证方向 |
| --- | --- | --- |
| 原 W 冻结 | 通用能力一定不回退 | 回归测试实际需要保留的能力 |
| 小 rank 减少参数 | 数据越少越应该用最小 rank | 固定基座、数据与评测比较候选值 |
| 多处挂载增加调整范围 | 挂所有层必然更好 | 对齐预算，报告匹配范围和实际成本 |
| loss 下降、参数更新 | 学会分析而非记忆案例 | 独立故障组、证据变化与未知场景评测 |
| adapter 可以单独保存 | adapter 能单独推理或完整恢复 | 固定基模身份，区分推理与恢复产物 |

对我们而言，LoRA 提供的是候选更新方式，不是测试专用模型的现成最佳配方。基座能力、数据量、监督质量与真实使用效果仍需要各自的证据。

## 11. 来源与核对范围

| 来源 | 本篇使用的位置 |
| --- | --- |
| [LoRA 原论文](https://arxiv.org/abs/2106.09685)，第 4.1、4.2、7 节 | 低秩参数化、初始化、挂载选择、经验边界 |
| [Microsoft 官方线性层实现](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)，`Linear` | 形状、Kaiming 初始化、alpha/r、合并 |
| [PEFT 官方 LoRA 文档](https://huggingface.co/docs/peft/package_reference/lora) | 初始化、目标模块和额外可训练参数配置 |
| [PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) | 权重布局与输入输出形状 |
| [PyTorch 自动求导](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 冻结参数、计算图和梯度职责 |
| [MIT 线性代数：列空间](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-1-the-column-space-of-a-contains-all-vectors-ax/) | 独立方向、列空间与秩的解释 |

核对日期：2026-11-07。动态文档和 main 分支不是实验依赖锁；正式训练时另固定版本。本篇不把历史 MiniMind 结果转写为新的 LoRA 实测，也不报告未进行的能力评测。
