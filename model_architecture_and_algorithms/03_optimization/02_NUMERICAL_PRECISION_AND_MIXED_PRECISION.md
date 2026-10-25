# 训练数值精度与混合精度

> 更新：2026-10-24。本文沿一次语言模型参数更新，解释参数存储、前向计算、中间结果、梯度和优化器状态的精度如何配合。示例以 PyTorch 2.14 为参照；具体训练配置与验收记录留在工程案例中。

[优化目录](README.md) · 前置：[自动求导](../01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)、[SGD 与 AdamW](01_SGD_ADAMW_AND_PARAMETER_UPDATES.md) · 实践：[MiniMind 第七周](../../training_engineering/case_studies/minimind/practice_notes/week_07_pretrain_experiment.md)

## 1. 精度策略在训练中的作用

一套模型可能有很多权重矩阵，但精度策略通常按张量用途和运算类型组织。例如，所有层的 Q 投影采用相同规则，所有 RMSNorm 采用相同规则，而不是逐个矩阵凭经验指定类型。

沿一次参数更新，可以看到几个不同对象：

```text
保存的模型参数、输入
        ↓
前向计算与中间张量
        ↓
loss 与归约
        ↓
反向传播、参数梯度
        ↓
优化器状态、参数更新
```

这些对象可以采用不同精度。模型参数保存为 FP32，并不要求所有矩阵乘法也以同样方式执行；某个模块输出 BF16，也不代表它内部的归一化统计使用 BF16。先分清对象，才能理解为什么一段代码里同时出现 `.float()`、autocast 和 `.to(dtype=...)`。

选择精度是在计算速度、存储开销和数值误差之间做取舍。大矩阵计算适合使用硬件支持的低精度路径，敏感统计与更新状态需要保留足够精度。实际收益要通过目标设备测量，不能只根据 dtype 名字预测速度或总显存。

## 2. 浮点格式与数值误差

### 2.1 数值范围与分辨能力

浮点数用有限位数表示数值。可以把它理解为带符号的有效数字乘以一个指数：指数决定能够覆盖多大的数量级，有效数字决定同一数量级内能分辨多细的差别。

| 类型 | 每个元素位数 | 指数字段位数 | 尾数字段位数 | 主要特点 |
| --- | ---: | ---: | ---: | --- |
| FP64 | 64 | 11 | 52 | 有效数字较多，常用于高精度数值检查 |
| FP32 | 32 | 8 | 23 | 常用于参数、统计与更新状态 |
| FP16 | 16 | 5 | 10 | 存储较小，但可表示的数值范围较窄 |
| BF16 | 16 | 8 | 7 | 范围与 FP32 接近，有效数字少于 FP16 |

表中尾数字段不包含正规数隐含的首位；不能把尾数字段位数直接当作十进制有效位数。FP16 与 BF16 都是 16 位，但它们对数值范围与分辨能力的分配不同，因此不能简单说其中一个在所有场合都更精确。

### 2.2 创建时的舍入不能事后恢复

BF16 在某些数量级上无法分辨所有连续整数。在当前环境执行：

```python
import torch

positions = torch.arange(255, 260, dtype=torch.bfloat16)
print(positions.float())
# tensor([255., 256., 256., 258., 260.])
```

原来的位置 257 被舍入成了 256。最后的 `.float()` 只是把已经保存的 256 转为 FP32，不会记住创建张量之前的 257。因此，位置或频率需要高精度时，要从创建和计算阶段就使用合适类型。

### 2.3 同一公式也可能出现舍入差

实数运算中的等价表达式，放到有限精度计算中不一定逐位相同。例如：

```python
a = x / torch.sqrt(s)
b = x * torch.rsqrt(s)
```

两者在数学上都表示 x 除以 s 的平方根，但计算路径与舍入可能不同。求和顺序、矩阵计算内核、设备和融合方式也会改变结果。相同 dtype 是比较条件的一部分，不足以保证逐位一致。[PyTorch 数值准确性说明](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

这种差别需要根据规模、预设容差和后续行为判断。微小舍入差并不自动说明模型公式错误；复现时统一表达式，可以帮助排除一个差异来源。

## 3. 一次更新涉及的精度对象

### 3.1 参数与中间结果

以 `q = q_proj(x)` 为例，至少有三个对象：输入 x、保存的 q_proj.weight，以及输出 q。autocast 可以让这次 Linear 使用低精度路径并输出 BF16，同时保留 FP32 的 Parameter 对象。

计算期间可能存在临时转换或缓存，但优化器仍更新原来的参数。它们不是两个分别学习的模型；参数的存储类型与某次计算的类型承担不同职责。

### 3.2 梯度与优化器状态

反向传播把结果累积到参数的 `.grad`。在本文采用的“FP32 参数 + autocast”方案中，参数梯度保存在 FP32 张量中，普通 AdamW 的 exp_avg 和 exp_avg_sq 也保持 FP32。

这里限定了参数类型和优化器实现。不能把它推广为“任何框架的所有梯度与优化器状态永远都是 FP32”；显式低精度参数、低精度优化器和分布式策略可能采用其他方案。

保存为 FP32 的 `.grad` 也不意味着反向传播每一步都在 FP32 中计算。中间反向运算会受到对应前向路径的影响；若中间梯度已经丢失，最终放进 FP32 容器不能补回信息。

### 3.3 整数与布尔张量

input_ids 和 labels 表示词表索引或忽略标记，通常使用 torch.long；mask 可以使用 bool，也可以先以 0/1 整数表示再转换。它们不属于需要在 FP32/BF16 之间选择的浮点特征。不要为了混合精度把 token ID 转成半精度浮点数。

## 4. 沿模型前向选择计算精度

### 4.1 线性投影与特征加工

Q/K/V 投影、Attention 汇总、FFN 和 LM head 包含大量矩阵乘法，适合交给 autocast 与后端选择高效的低精度实现。张量 dtype 仍不能完整描述硬件内部的乘法、累加和归约过程；验证时还要记录设备、后端和运行设置。

也不能假定整条 hidden states 始终为 BF16。例如，FP32 残差与低精度分支相加，结果可能提升为 FP32，随后下一个 Linear 再由 autocast 选择低精度路径。混合精度允许计算链上的类型随运算变化。

### 4.2 RMSNorm 的统计与缩放

RMSNorm 对最后一维计算平方均值，再据此缩放整组特征：

```python
x_float = x.float()
scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
normalized = x_float * scale
output = (weight * normalized).to(dtype=x.dtype)
```

这里展示精度流向；严格对照时乘法的括号与顺序也应采用参考实现。平方与归约会把多个数的信息汇集成一个尺度，这个尺度随后影响整组输出，因此采用 FP32 计算，再恢复输入类型。

epsilon 是避免分母过小的数值常量，与 dtype 是不同概念。改变 epsilon 会改变公式的具体结果；把中间张量转为 FP32 则改变数值计算条件。Q/K Norm 的可学习权重依然是模型参数，内部统计升精度不要求永久更改它的存储类型。

### 4.3 RoPE 的位置与旋转

RoPE 用位置 m 和频率 omega 计算角度 m * omega，再作用于 Q/K 的特征对。若位置创建时就被舍入，后续三角函数会使用错误的角度。

本项目采用的处理是：位置、频率和角度使用 FP32；FP64 数值检查保留 FP64。用这些角度计算旋转，再把结果转回对应 Q/K 的类型。转换仍在计算图中，不使用 detach 或 no_grad。

这能减少位置及角度阶段的误差，不能消除输入 Q/K 本身的量化或舍入，也不代表改变了模型可使用的上下文范围。

### 4.4 Attention 分数与 Softmax

Attention 从 QK 点积得到分数，再经过缩放、mask 和 Softmax。Softmax 的数学表达涉及指数与归一化，实现通常也会采用减最大值等稳定处理；FP32 进一步保留了分数与概率的分辨能力。

```python
scores = q @ k.transpose(-2, -1)
if scores.dtype in (torch.float16, torch.bfloat16):
    scores = scores.float()
scores = scores / (q.shape[-1] ** 0.5)
scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(scores, dim=-1)
output = weights.to(dtype=v.dtype) @ v
```

allowed 在这里代表因果规则与 padding 规则合成后的 bool mask。代码假设每个查询至少存在一个可见 key；全屏蔽一行需要另行定义行为，不能靠升精度解决。

权重可以保留 FP32 供观察，参与汇总 V 时再转换。点积结果转为 FP32，只保护之后的缩放、mask 与 Softmax；如果点积本身已经舍入甚至溢出，转换不会追回信息。融合 SDPA 内核也可能在内部采用不同的精度管理方式，不能根据这段手写代码推断它的全部实现。

### 4.5 交叉熵与 loss 汇总

logits 可能来自低精度 LM head，交叉熵涉及对数概率和大量 token 的归约，适合使用 FP32。CUDA autocast 将 cross_entropy 列为 FP32 运算；是否处于正确的 autocast 区域，需要结合调用入口核验。[PyTorch 运算精度规则](https://docs.pytorch.org/docs/2.14/amp.html#cuda-op-specific-behavior)

在 autocast 之外单独处理低精度 logits，不能假定同一套自动转换规则仍生效。也不要先手工 Softmax 再把概率交给 cross_entropy；该函数接收 logits，并包含相应的稳定计算。

token 平均属于监督与归一化口径，不能被 dtype 选择替代。使用 sum/count 时，还要明确有效目标 token、padding 忽略和 causal shift；公式口径错误，改成 FP64 也不能修正。

## 5. 用 autocast 组织混合精度

### 5.1 自动转换与显式转换的分工

autocast 按运算规则选择类型，通常包围前向与 loss；backward 和 optimizer.step 放在外面。使用它时，无需先对整个模型调用 `.half()` 或 `.bfloat16()`。显式转换主要用于需要明确控制的局部计算，例如位置创建、自定义 Norm 和手写 Attention。[PyTorch autocast 用法](https://docs.pytorch.org/docs/2.14/amp.html#autocasting)

下面是 CUDA 的调用关系，model、input_ids、labels 和 optimizer 假设已经创建并放到合适设备：

```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```

这段只说明类型管理位置，没有包含梯度累积、裁剪、溢出处理、调度器与恢复，不能直接替代完整训练入口。

### 5.2 在小模型上观察不同对象

在 MiniMind implementation 环境中，可运行下面的 CPU 演示，不需要 GPU，也不代表已验证 CUDA 性能：

```python
import torch
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss

torch.manual_seed(42)
config = ModelConfig(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_layers=2, num_heads=4, kv_heads=2, qk_norm=True,
)
model = MinimindForCausalLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
ids = torch.tensor([[1, 3, 4, 2]])
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    logits = model(ids)
    loss = causal_lm_loss(logits, ids)
loss.backward()
optimizer.step()

parameter = model.backbone.blocks[0].attention.q_proj.weight
print("parameter:", parameter.dtype)
print("logits:", logits.dtype)
print("loss:", loss.dtype)
print("gradient:", parameter.grad.dtype)
print("m:", optimizer.state[parameter]["exp_avg"].dtype)
print("v:", optimizer.state[parameter]["exp_avg_sq"].dtype)
```

2026-10-24 在当前 PyTorch 2.14 CPU 环境实测：parameter、loss、gradient、m、v 均为 FP32，logits 为 BF16。AdamW 的 m/v 通常到首次 step 才创建，所以这里在 step 后检查状态。其他实现和设备不能直接套用这组实测输出。

## 6. 反向传播、梯度缩放与参数更新

前向的一部分使用低精度，会影响对应的中间反向计算。FP16 数值范围较窄，小梯度可能下溢为零。GradScaler 通过暂时放大 loss，让反向传播的梯度处于更合适的量级，再在更新前还原；它不改变希望优化的原始目标。

标准 FP16 训练通常使用 GradScaler，BF16 因数值范围更宽通常无需使用，但这不保证任意训练配置都稳定。梯度缩放不等于梯度裁剪，也不能代替修正无效输入、错误 loss 或过大的学习率。

同时使用梯度累积和缩放时，同一窗口要保持一致的 scale；全部 backward 完成后再 unscale，之后进行所需的 token 归一化和梯度裁剪，再由 scaler 执行更新与调整。若因为非有限梯度跳过参数更新，调度器推进也要与实际更新保持一致。[PyTorch 梯度缩放与累积示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation)

FP32 参数与优化器状态有助于保留更新信息。若将参数直接长期保存在 BF16，某些相对参数大小很小的更新可能被舍入掉；这与前向矩阵使用 BF16 是不同选择。具体使用主参数副本、状态量化或分片属于进一步的训练实现方案，需单独记录。

## 7. 把规则落实为可检查的策略

### 7.1 按用途制定规则

| 对象或计算 | 本文采用的基础策略 | 需要核验的内容 |
| --- | --- | --- |
| 模型 Parameter | FP32 存储 | 构造后与恢复后的实际 dtype |
| Linear 与大矩阵计算 | autocast 选择 BF16/FP16 路径 | 设备支持、实际输出类型、速度和显存 |
| Norm 内部统计 | FP32 | 归约维度、epsilon、输出恢复与梯度 |
| RoPE 位置与角度 | FP32，FP64 对照保留 FP64 | 较大位置是否准确、旋转与输出类型 |
| 手写 Attention 缩放和 Softmax | 点积结果后转 FP32 | mask、概率、汇总 V 的类型与参考误差 |
| loss 与归约 | FP32 | 有效 token、归一化口径、有限值 |
| 参数梯度与 AdamW m/v | FP32 参数方案中保留 FP32 | 实际状态、累积、unscale 和更新 |
| token ID、labels、可见性 mask | 整数或 bool | 索引、忽略标记、mask 的语义 |

这套规则针对本文的基础方案，不是所有模型和框架的统一标准。矩阵数量增加时通常重复应用同一规则；引入新的运算或优化器，再针对新对象扩展策略。

### 7.2 精度决定哪些成本

较低精度能减少某些权重副本、激活或中间张量的字节数，并可能加速矩阵运算；它不会自动把所有显存占用减半。FP32 Parameter、梯度、优化器状态、工作区和缓存仍可能占用大量空间。应分别观察存储对象、峰值显存和实际吞吐。

FP64 适合部分高精度对照，但存储和计算成本更高，且某些自定义模块仍会内部转 FP32。使用 `.double()` 之后，要检查实际实现，不能只看模型参数类型就宣布完成纯 FP64 计算。

## 8. 数值对照与运行验收

精度验证先固定权重、输入、mask、监督口径、dropout 和运行路径，再比较前向和梯度。模块 shape 正确、没有 NaN，只能说明一部分行为正常；还需要检查与参考结果的误差以及 mask、padding、因果性等约束。

torch.testing.assert_close 同时使用绝对与相对容差，其基本判据为：

```text
|actual - expected| <= atol + rtol * |expected|
```

expected 接近零时，绝对容差尤其重要。全局最大绝对差大于 atol，也不一定失败，因为非零参考值仍有相对容差。容差应根据 dtype、运算路径和验证目的预先确定，超出后先定位原因，不能只调大容差直到通过。

建议从 FP32 的明确参考开始，再检查低精度路径的输出类型、误差和梯度，最后在目标设备测量完整更新、长序列、吞吐与显存。组件 CPU 测试不能替代 CUDA autocast 的实际训练验证；一次小输入对照也不覆盖全部输入范围。

MiniMind 本次从 RoPE 位置舍入、Attention 精度，到 RMSNorm 运算写法的定位过程，见[第七周教程第 2 节](../../training_engineering/case_studies/minimind/practice_notes/week_07_pretrain_experiment.md#2-模型结构与数值对齐)与[官方数值对照归档](../../training_engineering/case_studies/minimind/experiments/minimind3_alignment_cpu_fp32/README.md)。它们保留具体配置、命令和证据；本文用于解释这些选择背后的通用关系。

## 9. 参考资料

- [PyTorch 2.14：Automatic Mixed Precision](https://docs.pytorch.org/docs/2.14/amp.html)，查阅 autocast 与算子规则。
- [PyTorch 2.14：AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)，查阅梯度缩放、累积和裁剪的位置。
- [PyTorch 2.14：Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)，理解数学等价、设备差异和浮点误差。

## 10. 学习问答

### 10.1 参数矩阵那么多，是否逐个决定 dtype？

先按用途与运算类型决定规则。各层相同模块重复使用策略，通过实际检查发现需要例外处理的对象。Parameter、输出、梯度与优化器状态分别记录，不能用一个“模型精度”概括全部。

### 10.2 为什么模块输出 BF16，内部还说使用 FP32？

例如 RMSNorm 先将输入转 FP32 做统计与归一化，结束后恢复输入类型。观察输出只能知道边界类型；内部运算需要阅读代码或采集对应中间结果。

### 10.3 转回低精度会不会让前面的高精度计算失去意义？

会产生最后一次舍入，但在此之前的统计、角度或概率已用较高精度计算。与每一步都在低精度中舍入不同。收益取决于误差传播，仍需参考比较；它不是无损转换。

### 10.4 `.to(dtype=...)` 会切断梯度吗？

浮点类型之间的普通转换可以参与自动求导。它与 detach、no_grad 的作用不同；整数类型不具备同样的可微性质，不能把浮点转换的结论推广到任意 dtype。

### 10.5 `sqrt` 改为 `rsqrt` 是提高了精度吗？

它没有把 FP32 变成 FP64。两种写法数学等价但可能产生不同舍入；本次采用 rsqrt 是为了对齐参考计算。判断稳定性需要更多条件，不能据此宣布一种表达式在所有情况下更精确。

### 10.6 BF16 测试通过后，是否可以直接长期训练？

还需要目标设备的完整更新检查，包括 loss、梯度、优化器、累积、保存与恢复，以及长序列和实际预算。测试结果只支持已经覆盖的条件。
