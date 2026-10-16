# 第 4 周：Attention 与完整 Decoder

> 更新：2026-10-11。第四周实践验收完成：真实数据单步训练、初始化、greedy 生成与 tiny 序列过拟合均已运行通过；生成测试为 16 passed，全项目为 221 passed。下一阶段是第五周单设备训练系统，本篇不再追加新功能。
> 本周目标是把第三周的基础组件接成能够利用上下文、产生词表预测的 tiny Transformer。

[实践路线](../PRACTICE_ROADMAP.md) · [Attention 理论专题](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/README.md)

## 位置

前三周已经走到：JSONL → tokenizer → Dataset / collator → batch，以及 Embedding、RMSNorm、SwiGLU、残差等基础组件。
这些组件能处理各位置的特征，现在需要加入跨位置的信息交流。

本周路线是：单头聚合 → causal mask → 可学习 Q/K/V → 多头与 GQA → RoPE → 完整 Decoder → LM head、loss 与简单生成。
其中顺序可以围绕理解需要做小范围衔接，验收仍逐步增加，不把读过理论记为实现通过。

Attention 第一篇已完成理论整理，实践已从单次聚合推进到带 RoPE 和 padding mask 的 GQA。当前进度如下；第四周的验收是完整 tiny Transformer，而非单独完成 Attention 文件。

| 环节 | 当前状态 | 作用与后续连接 |
| --- | --- | --- |
| 基础 Attention 与因果掩码 | 已完成并测试 | 在允许的位置之间匹配、归一化与汇总内容 |
| 可学习单头与普通多头 | 已完成基础验证 | 用投影学习读取方式，拆头计算后拼接回模型宽度 |
| GQA | 核心实现、配置门禁、梯度与逐头参考已通过 | 较少 K/V 头供多组 Q 头读取 |
| RoPE 与 mask 衔接 | 已接入并通过参考测试；不可读行的拒绝检查通过 | 将位置信息接入 Q/K，处理需要屏蔽的补齐位置 |
| Decoder block 与层堆叠 | 单个 Block 与完整主干的装配、前向及梯度测试通过 | 将 Attention 与已有 RMSNorm、SwiGLU、残差组合为完整模型主干 |
| LM head、CE 与简单生成 | 输出头、权重共享、loss、单步训练与 greedy 生成测试通过 | 主干特征映射到词表预测，计算目标并逐步生成 token |
| 整体模型验收 | 全项目 221 passed；tiny 序列过拟合与 EOS 生成对照通过 | 检查模型能记住极小目标，并用生成对照训练目标 |

## 单次聚合

### 任务

在 [inspect_attention.py](../implementation/scripts/inspect_attention.py) 中设置一个 query、三个 key 和三个 value，
依次计算匹配分数、softmax 权重和加权输出。先使用小张量，是为了让每个输出都能追溯到输入数值。

单 query 阶段使用：

```text
q = [[1, 0]]
k = [[1, 0], [0, 1], [1, 1]]
v = [[10, 0], [0, 20], [30, 30]]
```

在 implementation 目录运行：

```shell
uv run python scripts/inspect_attention.py
```

### 结果

2026-10-06 在 Windows 的项目环境运行，得到：

```text
scores  [1,3] = [[0.7071, 0.0000, 0.7071]]
weights [1,3] = [[0.4011, 0.1978, 0.4011]]
output  [1,2] = [[16.0445, 15.9889]]
```

这条链在验证三个对象：scores 是尚未归一化的匹配分数，weights 是候选位置的聚合比例，
output 是按这些比例汇总的内容特征。运行只打印形状还不够，必须同时看数值，才能对照手算理解矩阵乘法。

这一阶段使用 CPU 上的手工 Q/K/V，不含参数投影、mask 或优化器；后续在同一脚本中逐步增加约束和检查。

### 解释

保持 Q/K 不变，只增大第二个 value 的第二个特征，scores 与 weights 不变，输出第二个特征增大。
学习者已经正确说明这一依赖关系。注意最后计算是 weights @ V，直接取权重的行与 V 的列点积；
只有前面的 Q @ K.T 需要转置 K。

如果改的是原始模型输入 X，则投影得到的 Q/K/V 通常一起变化，不能再假设权重不变。
这一区分将在加入可学习投影后再次观察。

## 多个 Query 的聚合

### 任务

在同一个脚本中把 Q 扩展为三行，K、V 保持原值。三个 query 分别为 [1,0]、[0,1]、[1,1]，原有的点积、Softmax 和加权汇总可以直接处理整个矩阵。

### 结果

2026-10-06 运行学习者实现，脚本正常退出，得到：

```text
weights [3,3] = [[0.4011, 0.1978, 0.4011],
                 [0.1978, 0.4011, 0.4011],
                 [0.2483, 0.2483, 0.5035]]

output  [3,2] = [[16.0445, 15.9889],
                 [14.0111, 20.0556],
                 [17.5872, 20.0698]]
```

学习者写入的两个断言通过：`weights.sum(dim=-1)` 与 `torch.ones(3)` 接近，`output[0]` 与原先的 [16.0445,15.9889] 接近。

### 解释

权重矩阵的第 i 行属于第 i 个 query，第 j 列对应第 j 个 key/value 来源。Softmax 沿最后一维归一化，使每个 query 分别分配自己的聚合比例；输出第 i 行就是这些比例对所有 value 的加权汇总。

第一个 query、K、V 都没有改变，所以它的结果与单 query 阶段一致。这验证了增加其他 query 不会改变原有 query 的计算。三个 query 是三个读取位置，仍属于单个注意力头。

## 因果掩码

### 任务

保留未加掩码的计算作为对照，再构造布尔下三角矩阵，限制位置 i 只能读取 0 到 i。掩码作用于 Softmax 之前的分数：

```python
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
output = weights @ v
```

`allowed` 中的 True 表示允许读取，`~allowed` 指向禁止位置。把这些位置的分数设为负无穷后，Softmax 为它们分配零权重，并在剩余候选之间重新归一化。

### 结果

2026-10-06 运行学习者实现，脚本正常退出，得到：

```text
weights = [[1.0000, 0.0000, 0.0000],
           [0.3302, 0.6698, 0.0000],
           [0.2483, 0.2483, 0.5035]]

output  = [[10.0000,  0.0000],
           [ 3.3024, 13.3952],
           [17.5872, 20.0698]]
```

三个断言通过：每行权重之和接近 1，所有禁止位置的权重为 0，第一个位置的输出等于 `v[0]`。

### 解释

第一个位置只有自己这一个来源，所以权重为 [1,0,0]；第二个位置可以读取前两个来源，原先分给第三个位置的权重被排除，前两个位置重新分配比例；第三个位置可以读取全部来源，因此结果与未加掩码时一致。

这些检查验证了当前输入下的权重与聚合结果。接下来通过改变未来位置的内容，检查过去输出是否保持不变，将可见性规则落实为输入变化与输出变化之间的约束。

### 扰动实验的检查

学习者写入第三个位置的 K/V 副本修改与输出对照。首次检查时两个断言通过，但最终汇总仍使用原始 `v`，因此实际验证的是 K 扰动。学习者随后改为 `changed_weights @ v_changed`；2026-10-06 重新运行，前两行输出不变、最后一行改变的断言通过，K/V 同时扰动实验已完成。

这里也需要区分原始分数与掩码后分数：修改第三个 K 会改变各 query 与它的点积，因此原始 `scores` 的第三列可以变化。对前两个 query，这一列被掩码覆盖为负无穷，故它们的 `masked_scores`、`weights` 和输出保持不变。掩码在本例中限制信息参与汇总，原始矩阵乘法仍计算全部位置之间的分数。

这次检查说明，仅有“最终输出发生变化”的断言还不足以确认所有实验变量都已接入计算。需要同时检查数据依赖，确认修改后的 K 用于打分、修改后的 V 用于汇总。

## 注意力计算函数

### 实现

学习者已在 [attention.py](../implementation/src/minimind_lab/models/attention.py) 实现 `scaled_dot_product_attention(q, k, v, causal=True)`。当前约定为同长度、非空序列：Q/K 为 [B,T,d_k]，V 为 [B,T,d_v]，返回 output [B,T,d_v] 和 weights [B,T,T]。

函数按点积与缩放、可选因果掩码、Softmax、Value 汇总的顺序执行。K 使用 `transpose(-2,-1)` 交换位置轴和特征轴；[T,T] 掩码在 `q.device` 上构造，并广播到每条序列。

### 作用

提取函数使匹配与汇总成为独立的计算单元。后续投影层产生 Q/K/V 后可以直接调用它，而现有的已知数值、归一化和因果性检查可以持续验证这部分计算。

### 验证

2026-10-06 对学习者函数执行临时检查，CPU 与 CUDA 均通过：B=1 的因果和非因果输出与已有数值一致，每行权重之和为 1，因果模式中禁止位置的权重为 0；B=2 的两个不同样本一起计算时，结果与分别计算一致。

这些结果确认当前实现能够处理批量输入，掩码也能在 GPU 上使用。

2026-10-07，学习者在 [test_attention.py](../implementation/tests/test_attention.py) 写入 `test_causal_attention_matches_known_values`。测试使用 [1,3,2] 的 Q/K/V，检查 output [1,3,2]、weights [1,3,3]，并用 `atol=1e-4, rtol=0` 将输出与已有四位小数结果比较。运行 `uv run --locked pytest -q tests/test_attention.py`，结果为 1 passed。

随后学习者增加 `test_noncausal_attention_matches_known_values`，显式传入 `causal=False`，核对无掩码的已知输出，用 `ones_like` 检查每行权重之和，并确认 `weights[0,0,2] > 0`。2026-10-07 运行同一测试文件，结果为 2 passed。因果与非因果路径均已有已知数值回归测试。

学习者又增加 `test_attention_batch_matches_individual_samples`，用固定种子生成 [2,3,2] 的 Q/K/V。完整 batch 的各样本输出与权重，分别与该样本单独调用函数的结果比较，两个样本的对照都通过。2026-10-07 运行测试文件，结果为 3 passed。

首次逐样本调用使用 `q[i,:,:]` 等索引，输入成为 [T,d]；当前函数的末两维运算能够处理这种输入，因此对照成立。学习者随后统一为 `i:i+1` 切片，使逐样本路径保留 [1,T,d] 的 batch 维度，返回结果也用同样的切片进行比较，重新检查通过。

学习者增加 `test_causal_attention_ignores_future_changes`，使用 [1,3,2] 的 Q/K/V，保持 Q 不变，修改第三个位置的 K/V 副本后再次调用函数。断言确认前两个位置的输出和权重不变，第三个位置的输出改变。2026-10-07 运行测试文件，结果为 4 passed；修改位置的索引以及使用修改后 K/V 的调用均正确。

学习者增加 `test_attention_forward_and_gradients_match_torch`，准备独立的 float64 输入与参考副本，使用相同因果设置，前向结果在 `1e-10` 容差下匹配；两条路径也已分别执行平方和目标的 backward。2026-10-07 运行测试文件，结果为 5 passed。但末尾断言当前比较的是 `q` 与 `rq`、`k` 与 `rk` 的数值，尚未比较梯度，也未覆盖 V；需要改为三组 `.grad` 对照后才能验收反向一致性。

随后学习者将 Q/K 的断言改为 `.grad` 比较，检查通过后又补齐 `v.grad` 与 `rv.grad`。2026-10-07 重新运行仍为 5 passed，前向输出与 Q/K/V 三组梯度均在 `1e-10` 容差下与参考一致，本项验收完成。该对照使用固定种子、CPU float64、同长度因果输入与关闭 dropout 的设置。

### 参考接口与梯度检查

`torch.nn.functional.scaled_dot_product_attention` 是 PyTorch 提供的缩放点积注意力接口，接收已有 Q/K/V，执行缩放打分、可选掩码、Softmax、可选 dropout 与 Value 汇总。当前参考调用设置 `dropout_p=0.0`、`is_causal=True`，默认缩放为 `1/sqrt(d_k)`，返回输出张量；学习者函数额外返回注意力权重。Q/K/V 的可学习投影位于该计算接口之前。

`backward()` 把梯度写入输入叶子张量的 `.grad`，不会更新输入张量自身的数值。因此，比较复制后数值相同的 `q` 与 `rq` 无法检查反向计算是否一致。梯度对照必须比较 `q.grad` 与 `rq.grad`，K、V 同理。

## 可学习的单头自注意力

### 实现

学习者在现有 [attention.py](../implementation/src/minimind_lab/models/attention.py) 中实现 `SingleHeadSelfAttention(nn.Module)`。`q_proj`、`k_proj`、`v_proj` 分别将 hidden_size 映射到 head_dim，`o_proj` 将 head_dim 映射回 hidden_size，四个线性层都关闭 bias，并注册为模块属性。

`forward(x, causal=True)` 从同一个 X 生成 Q/K/V，将 `causal` 传给已验证的计算函数，再把内部聚合结果传入 `o_proj`，返回投影后的输出和注意力权重。hidden_size=4、head_dim=2 时，X [B,T,4] 转为 Q/K/V [B,T,2]，内部汇总为 [B,T,2]，最终输出为 [B,T,4]。

### 作用

这个模块把固定计算与可学习参数连在一起。三个输入投影的参数决定如何提取用于匹配和传递的特征；注意力函数在位置之间汇总信息；输出投影把汇总特征转换回模型主路径的宽度。线性层在所有位置和样本之间共享参数，但每个位置根据自己的输入产生不同向量。

参数的梯度通过输出投影、聚合与 Softmax 传回三个输入投影，并继续传到 X。后续组合归一化和残差时，已经有一条输入输出同宽的 Attention 分支可以使用。

### 验证

2026-10-07 临时检查使用固定种子、hidden_size=4、head_dim=2、X [2,3,4]，在 CPU/CUDA 上分别运行因果和非因果模式。输出 [2,3,4]、权重 [2,3,3]、逐行归一化均符合预期；平方均值目标 backward 后，X 和四个投影层的权重都有有限梯度，本例四个参数梯度范数均大于 0。原有 Attention 测试仍为 5 passed。

学习者随后在 `tests/test_attention.py` 增加 `test_single_head_attention_shapes_and_gradients`，固定种子生成 X [2,3,4]，构造 hidden_size=4、head_dim=2 的模块，检查输出和权重形状，并对 `output.square().sum()` 反向传播。X 和 `model.named_parameters()` 中四个投影参数的梯度存在且全部有限。2026-10-07 运行 Attention 测试文件，结果为 6 passed。

这个模块级测试已将形状与反向路径检查保存下来。平方和与平方均值都能在这里提供标量检查目标；当前测试采用平方和。

学习者增加 `test_single_head_attention_matches_explicit_projections`：实际路径调用 CPU float64 的模块，参考路径用同一组参数的转置显式生成 Q/K/V，调用 PyTorch SDPA，再显式应用输出投影。2026-10-07，输出在 `1e-10` 容差下匹配，Attention 测试文件结果为 7 passed，模块的投影与前向连接对照完成。

学习者增加 `test_single_head_attention_ignores_future_inputs`，保持模块参数不变，只将 X 第三个位置的特征加上 100，再次调用因果模式。前两个位置的输出和权重保持不变，第三个位置的输出改变。2026-10-07 运行 Attention 测试文件，结果为 8 passed，单头阶段已完成基础数值、batch 独立性、因果行为、参考梯度与投影连接验证。

### 线性层的参数布局

学习者在显式投影测试中进一步理解了 `nn.Linear` 的参数布局：weight 存储为 [输出维度,输入维度]，每一行对应一个输出特征使用的全部输入系数。因此 `nn.Linear(4,2)` 保存 [2,4] 的 weight，实际计算为 `x @ weight.T`；关闭 bias 时，无需再加偏置。输出投影 `nn.Linear(2,4)` 则保存 [4,2] 的 weight。

公式中的 `Q=XW_Q` 若把 W_Q 定义为 [hidden_size,head_dim]，对应的是 `q_proj.weight.T`。转置是在匹配行向量公式与参数存储方式，不会改变要学习的映射。详细说明见[线性层的批量计算](../../../../model_architecture_and_algorithms/01_model_architecture/foundations/02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md#批量计算)。

### 转置与未来信息的排除

修改 X 的第三个位置，会通过逐位置线性投影改变 Q/K/V 的第三行，前两行保持不变。Q 的第三行变化影响分数矩阵的第三行；K 的第三行变化在 K 转置后对应第三列，影响各 query 对这个来源位置的原始分数。

转置是交换行列索引：`K.T[j,i] = K[i,j]`。原矩阵的第三行按原有元素顺序成为转置矩阵的第三列，不能按顺时针或逆时针旋转来理解。原始分数中，前两个 query 对第三个 key 的分数可以改变；因果掩码将这些分数覆盖为负无穷，Softmax 后对应权重才是 0。

V 的第三行仍按位置保存内容，并不在最后的 `weights @ V` 中转置。前两个 query 对它的权重为 0，因此这行变化不会影响它们的汇总结果。输出投影逐位置作用于特征，不会再把第三个位置的信息传给前两个位置。

## 多头张量的拆分与换轴

### 实现与结果

学习者创建 [inspect_attention_heads.py](../implementation/scripts/inspect_attention_heads.py)，设置 B=1、T=3、H=2、d=2，生成连续值 0 到 11。先构造 Q [1,3,4]，reshape 为 [1,3,2,2]，再 transpose(1,2) 得到 [1,2,3,2]。

2026-10-07 运行脚本，三步形状均正确。第 0 个头为 [[0,1],[4,5],[8,9]]，第 1 个头为 [[2,3],[6,7],[10,11]]，符合按特征分组、按位置排列的约定。学习者随后补充 `q_flat`、`q_split` 与两个头的数据打印，并用 `torch.testing.assert_close` 比较 `q_flat.reshape(B,T,-1,d)` 与 `q_split`；重新运行，实际数值符合预期，断言通过。

### Reshape 的理解

学习者询问 reshape 是否打乱全部元素，以及 `-1` 是否保留某个维度。reshape 按张量当前的逻辑元素顺序重新分组，保留数值、总数和展开顺序。指定 [B,T,H,d] 时，B/T 在本例中仍沿用原值，每个位置的四个特征被分成两个头；若指定不同形状，原先的样本或位置分组不会自动得到保护。

`q_flat.reshape(1,3,-1,2)` 中的未知大小由 12/(1×3×2)=2 得到，结果与当前 `q_split` 相同；`-1` 表示推算维度大小，不表示保持某个原维度。reshape 与 transpose 的区别及线性层上下文见[张量维度基础](../../../../model_architecture_and_algorithms/01_model_architecture/foundations/02_VECTORS_MATRICES_AND_LINEAR_LAYERS.md#reshape-与维度推算)。

拆头把每个位置的特征分组，换轴则让每个头的 [T,d] 位于最后两维，为后续逐头矩阵乘法准备排列。这个阶段先使用投影后 Q 的数值示意，后续在同一脚本中接入逐头 Attention 观察；真实投影参数尚未接入。

### 逆向换轴与合并

学习者对 `q_heads` 执行 `transpose(1,2)` 恢复 [B,T,H,d]，再 reshape 为 [B,T,H*d]，断言结果与 `q_flat` 一致；另外直接对 `q_heads` reshape 到同一目标形状，断言其排列与 `q_flat` 不同。2026-10-07 运行脚本，两个断言都通过。

正确恢复的第一个位置为 [0,1,2,3]；直接 reshape 的第一个位置为 [0,1,4,5]，混入了另一个位置的特征。这个对照验证了合并时必须先把同一位置的各头排列到一起，不能只看最终形状是否正确。当前实验只操作排列，实际 Attention 汇总后的值会是新特征，但沿用同样的换轴与合并顺序。

### 并行计算与逐头计算

学习者沿用 Q [B,H,T,d]，设置 `k_heads=q_heads.clone()`、`v_heads=q_heads+10.0`，一次调用注意力函数，再逐头调用并对照 output 与 weights。2026-10-07 运行脚本，output [1,2,3,2]、weights [1,2,3,3] 的形状断言及两个头的数值对照均通过。

矩阵乘法在这里处理最后两维，每个固定的 batch、head 组合得到自己的位置匹配矩阵。每项原始分数是同一头中 query 与 key 的点积，沿 head_dim 求和；Softmax 再沿来源位置轴归一化，内容汇总沿来源位置轴求和。这里的操作是矩阵乘法与点积，不是三维向量叉积。

逐头与并行的结果一致，说明当前实现没有把不同头混在一起计算。它不表示多个头与只用一个头的完整模型能力相同；各头拥有各自的特征表示与聚合权重，完整模块还要拼接并通过输出投影组合这些结果。

### 汇总结果的合并

学习者将实际 `heads_output` [B,H,T,d] 先执行 `transpose(1,2)`，得到 [B,T,H,d]，再 reshape 为 [B,T,H*d]。参考路径按头取出 [B,T,d] 的输出，用 `torch.cat(..., dim=-1)` 沿特征维拼接。2026-10-07 运行脚本，输出 [1,3,4] 的形状断言与两条路径的数值对照均通过。

这个测试验证了同一个位置的不同头输出被按头顺序拼到一起。这里合并的是 Attention 已汇总出的新特征，不需要与原始 Q 相等。合并之后，每个位置再次拥有一个完整特征向量，接着交给 `o_proj` 学习组合各头的结果，最终返回模型主路径。

至此，投影后张量的拆分、逐头聚合与输出拼接都已有独立检查。下一步把可学习投影接到两端，组成一条完整的多头分支。

## 可学习的多头自注意力

### 实现与检查

学习者在 `attention.py` 中实现 `MultiHeadSelfAttention`：三个输入投影均产生 hidden_size 个特征，拆为 [B,T,H,d]，换轴为 [B,H,T,d] 后调用注意力函数。汇总结果按头取出，用 `torch.cat(..., dim=-1)` 拼接为 [B,T,hidden_size]，再经过输出投影。这里使用 cat 与此前验证的换轴加 reshape 具有相同的拼接含义，两种写法都可以。

2026-10-07 临时检查使用固定种子、hidden_size=4、num_heads=2、X [2,3,4]。默认因果模式的输出 [2,3,4]、权重 [2,2,3,3] 均符合预期，平方和目标反向后，输入和四个投影参数都有有限梯度。现有八项 Attention 测试通过，但这些测试尚未直接覆盖新类。

### 配置与行为边界

初版配置检查使用 `< 0`，没有拒绝零值。实际构造 `(hidden_size=4,num_heads=0)` 时，在取余处抛出 ZeroDivisionError；构造 `(0,2)` 则被接受。约定要求两个值都严格大于零，应先以 `<= 0` 拦截，再检查整除关系。Python 的 or 从左向右短路，提前识别零头数后，就不会执行取余。

初版 forward 接收 causal，但内部调用未传入它，计算函数因此始终使用默认的 True。临时检查中，即使调用 `model(x, causal=False)`，第一个位置对第三个位置的权重仍为 0。需要将该参数传入计算函数，才能让模块的公开接口真正控制因果行为。

这两个问题需要分别通过非法配置检查和非因果行为测试保护。形状与梯度检查证明默认路径能运行，不能替代对配置边界和模式切换的验证。

学习者将配置判断改为 `<= 0` 后，临时检查中的零隐藏宽度、零头数、负值及无法整除的五组配置均按预期抛出 ValueError。参数透传的第一次修正则在 forward 中额外加入 `is_causal=True` 并将其传入内部函数，原有 causal 仍未使用。复查时默认因果路径通过，`model(x, causal=False)` 的未来位置权重检查仍失败。当前接口只需保留 causal 并用 `causal=causal` 传递；PyTorch 官方 SDPA 的参数名 is_causal 不要求本项目也额外提供同名参数。

学习者随后移除多余的 is_causal，将 causal 作为第四个位置参数传给现有计算函数；这与使用 `causal=causal` 的效果相同。2026-10-07 复查五组非法配置均抛出 ValueError。在 CPU 与 CUDA 上分别检查 causal=True/False：输出及权重形状正确，权重各行和为 1，因果模式的未来权重为 0，非因果模式的第一个位置可以读取第三个位置。前向与 PyTorch SDPA 加输出投影的参考路径在 atol=1e-6、rtol=1e-5 下匹配，输入及四个投影参数的反向梯度存在且有限。现有测试仍为 8 passed，多头检查目前是临时检查，尚未进入测试文件。

### 测试目标的归约

学习者加入 causal=True/False 的参数化模块测试后，形状、权重行和及未来位置可见性的断言均通过，但两种模式都在 `output.sum(dim=-1).square().backward()` 处报错。2026-10-07 实际运行结果为 8 passed、2 failed，错误为 `grad can be implicitly created only for scalar outputs`。

output 为 [2,3,4]，沿最后一维 sum 只消去特征轴，得到 [2,3]；square 是逐元素操作，不继续消去维度。未提供外部梯度的 backward 需要单元素目标，当前应使用 `output.square().sum()`：先平方每个输出元素，再对所有元素求和，得到形状为 [] 的标量张量。这是检查反向路径用的测试目标，不是语言模型的正式训练 loss。

先求和再平方与先平方再求和还定义了不同目标，例如 [1,-1] 的和平方为 0，平方和为 2。即使在原写法后继续 sum 可以得到标量，也不等于本次约定的逐元素平方和目标。理解归约时应同时检查张量形状和计算含义。

学习者将目标修正为 `output.square().sum().backward()` 后，2026-10-07 再次运行 `tests/test_attention.py`，结果为 10 passed。新增参数化测试的两种模式均通过，包括输入及四个投影参数的有限梯度检查。模式切换与多头基础反向路径现在已有正式回归测试。

### 配置边界测试

学习者增加 `test_multihead_attention_rejects_invalid_config`，用 pytest 参数化分别测试 (0,2)、(4,0)、(-4,2)、(4,-2)、(5,2)，每组在 `pytest.raises(ValueError)` 中构造模块。2026-10-07 运行测试文件，五项边界测试全部通过，总结果为 15 passed。

该测试把配置拒绝行为保存为合同：非法配置必须抛出约定的 ValueError，静默接受或者抛出其他类型的异常都会导致失败。特别是零头数用例，可以防止初始化重新出现先取余、后报错的 ZeroDivisionError。

### 逐头参考路径的切片

学习者编写两种模式的参考前向测试，显式投影与输出比较已接入，但循环中虽然计算了 start/end，调用 PyTorch SDPA 时仍传入完整的 Q/K/V [2,3,4]。两个循环因此各自计算同一组完整特征，得到两个 [2,3,4] 输出，拼接后为 [2,3,8]，与输出投影 [4,4] 无法相乘。2026-10-07 运行测试文件为 15 passed、2 failed，报错为 `mat1 and mat2 shapes cannot be multiplied (6x8 and 4x4)`。

应将三个输入分别切为 `q[...,start:end]`、`k[...,start:end]`、`v[...,start:end]`，每头输入与输出都是 [2,3,2]，两个头拼接为 [2,3,4]。省略号保留所有前置维度，切片仅选择最后一维的头特征。这个修正同时恢复了每头点积宽度及其缩放因子，不能仅修改输出投影的形状来消除报错。

学习者将三个输入改为逐头切片后，2026-10-07 再次运行测试文件，结果为 17 passed。实际模块的多头并行路径与逐头 PyTorch SDPA 参考路径，在两种模式下均以 CPU float64、atol=1e-10、rtol=1e-10 数值匹配。普通多头模块当前已覆盖基础形状、权重行为、有限梯度、配置拒绝与参考前向；尚未包括 padding mask、位置编码或缓存。

## GQA 的 KV 分组

### 位置与头的区别

学习者提出：同一个输入中，各位置的查询向量已经共用一组 K/V，为什么还要分组共享？这个疑问需要先区分 token 位置与注意力头。一个头中的 Q/K/V 各自都是包含全部位置的矩阵，各位置 Q 确实读取这个头的同一组 K/V。普通多头又为不同头生成不同的 K/V 特征；GQA 则让同组的不同 Q 头也使用同一组 K/V 序列矩阵。

例如四个 Q 头、两个 KV 头时，Q0/Q1 使用 KV0，Q2/Q3 使用 KV1。每个 KV 头都包含全部位置，头编号不代表位置编号。共享 K/V 不要求共享 Q，同组的查询特征仍可不同，因此注意力权重与输出也可不同。

### 示意数值与观察

学习者创建 [inspect_kv_groups.py](../implementation/scripts/inspect_kv_groups.py)，以 `torch.arange(8)` 构造 KV [1,2,2,2]，轴含义为 batch、位置、KV 头、头特征。连续数字只是便于追踪的数据值，不是 token ID 或语义标签。忽略 batch 后，原始数据为：

| 位置 | KV 头 0 | KV 头 1 |
| --- | --- | --- |
| 0 | [0,1] | [2,3] |
| 1 | [4,5] | [6,7] |

沿头轴执行 `repeat_interleave(repeats=2,dim=2)`，得到 [1,2,4,2]，对应头顺序 KV0、KV0、KV1、KV1。2026-10-07 运行脚本，形状及四项头切片数值断言全部通过。`kv[:,:,0,:]` 保留全部样本与位置，只选择头 0，因此断言验证了共享完整序列，而非仅复制某一个位置。

这个实验只观察 KV 头扩展，不包含 Q 投影或 Attention 计算，也没有新增独立的投影参数。GQA 在投影时只生成较少的 K/V 头，计算时再建立其与各 Q 头的对应关系。相对同样 Q 头数和头宽度的 MHA，K/V 投影参数及未展开 KV cache 可减少；注意力权重矩阵仍按 Q 头数产生。这里使用 repeat_interleave 便于验证分组，不能据此宣称扩展后的临时张量没有内存开销。

### KV 扩展函数

学习者在 `attention.py` 中实现 `repeat_kv`，检查四维输入与正重复次数，n_rep=1 时直接返回原对象，其余情况沿 dim=2 调用 repeat_interleave。2026-10-07 临时检查在 CPU/CUDA 使用 X [2,3,2,4] 与 n_rep=1/2/3，按源头逐个堆叠构造独立参考：输出形状、头顺序、dtype/device 均符合约定，n_rep=1 的对象身份也保持不变。对输出总和反向后，输入各元素梯度等于重复次数，说明扩展仍连接原输入的计算图。三维输入、零重复次数及负重复次数均抛出 ValueError。现有正式测试文件仍为 17 passed，函数检查尚未保存为测试。

函数初版报错文本为 error，返回类型标注缺失。这两项不影响已检查的计算结果，但应补充返回类型 `torch.Tensor` 和包含输入约束的报错信息，便于后续接入 GQA 时定位调用问题。

学习者随后完善报错文本，并加入 n_rep=1/2/3 的 `test_repeat_kv_preserves_head_groups`，通过双层循环比较每个源头与其全部副本，检查形状和 dtype/device。2026-10-07 运行正式测试文件，结果为 20 passed。单次重复的 `actual is x` 断言尚未加入，返回类型标注也尚未补齐，不能因用例全部通过就认为所有约定均已覆盖。

学习者补充 n_rep=1 时的对象身份断言，并加入 `test_repeat_kv_accumulates_gradients`，直接创建四维浮点叶子输入，对扩展输出求和再 backward，将输入梯度与全部填为 n_rep 的张量对照。2026-10-07 运行 `tests/test_attention.py`，结果为 23 passed。三个重复次数的梯度用例均通过，KV 扩展的分组对应关系和重复路径梯度累加现在已有正式测试。

### 完整 GQA 的连续任务

下面保留完整 GQA 的任务地图，供分步带练时定位当前环节，不要求一次完成。学习者希望继续小步实现与验证，并及时串起已经学过的知识；每一步先说明在模型中的位置与职责，再追踪输入输出，随后写代码并检查。推进次序为初始化 → 投影与拆头 → KV 扩展与 Attention → 输出合并 → 必要测试。目标是接通 GQA 计算分支，复用已有注意力函数与 repeat_kv；不在这里同时加入 RoPE、Q/K 归一化、dropout 或缓存，也不宣称已完整复刻官方 Attention。

**接口与配置**

在 `attention.py` 增加 `GroupedQueryAttention(hidden_size,num_heads,num_kv_heads)`，forward 接口沿用 `(x,causal=True)`，返回输出与注意力权重。保留已有单头和普通多头类，便于参考对照。

先检查三个配置值均大于零，再检查 hidden_size 能被 num_heads 整除、num_heads 能被 num_kv_heads 整除，违反约定时抛出 ValueError。保存配置，计算 head_dim=hidden_size//num_heads 与 n_rep=num_heads//num_kv_heads。三个值为正的检查应在取余之前执行。

四个投影仍关闭 bias，但 K/V 输出宽度缩小：

| 投影 | 输入宽度 | 输出宽度 |
| --- | --- | --- |
| q_proj | hidden_size | num_heads × head_dim |
| k_proj | hidden_size | num_kv_heads × head_dim |
| v_proj | hidden_size | num_kv_heads × head_dim |
| o_proj | num_heads × head_dim | hidden_size |

共享来自 K/V 投影生成较少的独立头，不能先生成完整 MHA 的 K/V 再丢弃部分头。

2026-10-08，学习者以类名 `GroupQueryAttention`、KV 头配置名 `kv_heads` 写出初始化初版。正值与整除校验正确，临时检查七组非法配置均抛出 ValueError，head_dim 与 n_rep 计算正确；但四个 Linear 仍使用示意例子的固定数字 8、4。构造 (8,4,2) 时参数形状符合预期，换成 (12,6,3) 后仍得到 Q/O [8,8]、K/V [4,8]，而此配置应为 Q/O [12,12]、K/V [6,12]。当前仅需把投影宽度改为 hidden_size、num_heads×head_dim 与 kv_heads×head_dim，保留已有配置校验，不进入前向。这个检查说明接口接收配置，并不自动保证内部参数由配置决定。

学习者修正投影宽度后，同日临时复查 (8,4,2)、(12,6,3)、(8,4,4)、(8,4,1) 四组有效配置，四个投影的参数形状与无 bias 设置正确。X [2,3,hidden_size] 经过 Q/K/V 投影后，输出分别为 [2,3,num_heads×head_dim]、[2,3,kv_heads×head_dim]、[2,3,kv_heads×head_dim]，符合约定；七组非法配置仍被拒绝，正式 Attention 测试为 23 passed。当前 o_proj 使用 hidden_size→hidden_size，与 num_heads×head_dim→hidden_size 等价，因为构造合同保证两者相等。这个结果验证了初始化，不代表尚未编写的 forward 已通过。

**前向链路**

用 hidden_size=8、num_heads=4、num_kv_heads=2 理解本轮的主测试配置，head_dim=2、n_rep=2：

| 操作 | Q 形状 | K/V 各自的形状 |
| --- | --- | --- |
| 投影 | [B,T,8] | [B,T,4] |
| 分别拆头 | [B,T,4,2] | [B,T,2,2] |
| KV 扩展 | 不变 | [B,T,4,2] |
| transpose(1,2) | [B,4,T,2] | [B,4,T,2] |

repeat_kv 的头轴约定是 dim=2，因此在本链路中先扩展 KV 再换轴。将换轴后的 Q/K/V 交给现有计算函数，透传 causal；汇总输出为 [B,4,T,2]、权重为 [B,4,T,T]。恢复位置轴并拼接为 [B,T,8]，经过输出投影后返回 [B,T,8] 与权重。合并可沿用已验证的 cat，或使用 transpose 后 reshape。

**验证任务**

以下是模块最终需要的验证地图，随实现逐步加入，不是当前初始化步骤的一次性作业。固定种子，先用 CPU 测试；只有参考数值路径要求 float64。

1. `test_gqa_shapes_and_gradients`：参数化 causal=True/False，模型 (8,4,2)、输入 [2,3,8]。检查输出 [2,3,8]、权重 [2,4,3,3]、权重行和及因果可见性；输出平方和 backward 后检查输入与四个投影参数的有限梯度。可改写已有多头测试。
2. `test_gqa_matches_per_head_reference`：同样参数化两种模式，使用 double 模型和 float64 输入。显式生成 Q/K/V，遍历 Q 头 h，用 h//n_rep 得到对应 KV 头，按各自头编号切片，逐头调用 PyTorch SDPA，拼接并显式应用 o_proj。用 atol=1e-10、rtol=1e-10 比较输出。参考路径不能直接调用待验证的 repeat_kv，否则容易共同重复分组错误。
3. `test_gqa_matches_mha_when_head_counts_equal`：构造 MHA (8,4) 与 GQA (8,4,4)，加载同一组参数后使用同一输入比较输出和权重。通过 `gqa.load_state_dict(mha.state_dict())` 对齐参数，不能只用两个随机初始化模型比较。可参数化两种模式。这个测试检查 n_rep=1 时 GQA 能退化为普通 MHA。
4. 参数化非法配置测试：例如 (0,4,2)、(8,0,2)、(8,4,0)、(-8,4,2)、(7,4,2)、(8,4,3)、(8,4,8)，均应在构造时抛出 ValueError。给 repeat_kv 同时补上三维输入、零重复次数及负重复次数的拒绝测试。

逐头参考的核心是分别选 Q 头与 KV 头：Q 切片范围由 h×head_dim 决定，K/V 切片范围由 (h//n_rep)×head_dim 决定。以四个 Q 头、两个 KV 头为例，对应 KV 头编号为 0、0、1、1。输入与参数均不改变，两条计算组织不同但数学目标相同。

**运行与完成标志**

运行 `uv run --locked pytest -q tests/test_attention.py`。所有既有测试及新增测试通过后，再由助手审查实际代码，复查共享头的参数形状与参考路径。测试通过不自动意味着完整 MiniMind 模型已完成；这个里程碑只标记 GQA 分支已接通，然后进入位置编码。

2026-10-08，学习者写出 GroupQueryAttention 的前向初版，Q/K/V 按各自头数拆分，K/V 在换轴前调用 repeat_kv，并正确传递 causal。临时运行模型 (8,4,2)、输入 [2,3,8]，到输出合并时遇到 `TypeError: 'int' object is not iterable`，原因是列表推导写成 `for h in self.num_heads`。应以 `range(self.num_heads)` 生成 0 到 num_heads-1 的头编号，沿特征轴拼接每个 Q 头的输出。既有测试运行仍为 23 passed，但尚未直接调用 GQA，不能用该结果判断新模块成功。当前未完成参考数值与反向检查，先修合并循环。

学习者改为 range 后，同日临时复查 CPU/CUDA 上 (8,4,2)、(12,6,3)、(8,4,1)、(8,4,4) 四组配置的两种 causal 模式。输出及权重形状、权重行和、因果与非因果可见性均正确。逐头参考直接用 Q 头 h 与 KV 头 h//n_rep 的投影切片调用 PyTorch SDPA，拼接后显式应用输出投影；float64 输出在 atol=1e-10、rtol=1e-10 下匹配，平方和 backward 后输入及四个投影参数均有有限梯度。另将 MHA (8,4) 参数加载到 GQA (8,4,4)，两种模式的输出与权重均一致。正式测试仍为 23 passed，上述 GQA 检查尚未保存进测试文件。

学习者加入两种 causal 模式的 `test_gqa_shapes_and_gradients` 后，同日正式测试为 25 passed。输出 [2,3,8]、权重 [2,4,3,3]、行和、因果上三角以及输入与四个投影参数的有限梯度检查均通过。初版非因果断言只检查 weights[0,0,0,2]，随机种子也尚未固定；下一步在构造模型前设置 manual_seed(42)，并用 `(weights[...,0,2]>0).all()` 覆盖全部样本与头。明确记录覆盖范围，不把单个头的检查视为全部头的验证。

学习者固定随机种子并加入逐头参考测试后，同日运行结果为 27 passed。两种模式下，Q 头 h 与 KV 头 h//n_rep 的参考切片对应正确，拼接和输出投影与实际模块在 float64、atol=1e-10、rtol=1e-10 下匹配。核心 GQA 计算学习可以收尾，但不标记所有工程测试已完成。

复查发现非因果断言写成 `weights[...,0,2].all()>0`：先把数值张量归约为布尔值，验证的是全部元素非零，而非逐元素大于零。应改为 `(weights[...,0,2]>0).all()`，先形成逐元素比较结果，再检查全部成立。当前 Softmax 输出为正，错误顺序未导致用例失败，仍需修正断言语义。参考循环写成 range(4) 与本例四头配置一致，可改用 range(model.num_heads) 保持配置对应。

**补充测试待办**

- 将已临时验证的等头数 GQA/MHA 输出与权重对照保存为正式测试。
- 将 GQA 非法配置与 repeat_kv 非法输入的拒绝检查保存为正式测试。
- 扩展配置覆盖，包括一个 KV 头及另一组模型宽度；现有 GQA 正式数值测试固定为 (8,4,2)。

这些待办在模型级整合验收前补齐，当前不把临时检查算作正式回归覆盖。下一主题仍是位置编码，不再重复实现已有注意力核心。

## RoPE 的起点

### 已有理解与校准

学习者已能描述：在 Q/K 点积之前，按位置对特征做旋转，使匹配分数中出现相对位置关系。这与 [RoFormer 原论文](https://arxiv.org/html/2104.09864v5) 的构造一致。接下来不重新背定义，而是把旋转公式、特征配对、位置角度和现有 GQA 的形状连接起来。

需要先区分三层含义。配对的是同一 token、同一注意力头内的两个特征坐标，不是两个 token 位置。标准不带额外缩放的二维旋转保持向量的欧氏范数，也保持特征数量和张量形状；它没有通过旋转减少序列 token 数。每个特征对使用自己的频率，位置 m 对第 i 对特征的角度为 m×omega_i，因此同一位置的所有特征对不一定旋转同一角度。

旋转角度使用各自的绝对位置，但 Q/K 点积中的旋转关系取决于位置差。对一个二维特征对，若 q 位于 m、k 位于 n，则有 `(R(m*omega)q)^T(R(n*omega)k)=q^T R((n-m)*omega)k`。这意味着位置作用以相对距离进入匹配，不意味着点积只剩距离，内容向量 q/k 仍然参与计算。

RoPE 不应概括成自动消除上下文长度限制。有限的可学习绝对位置表有预设位置数量，而函数式绝对位置编码不都具有同样的硬表长约束；RoPE 可以按公式计算更远位置的旋转，但训练长度外的行为仍需验证。[Position Interpolation 原论文](https://arxiv.org/abs/2306.15595) 专门讨论了 RoPE 模型直接外推可能产生的问题。当前先实现基础 RoPE，不加入长上下文扩展方法。

### 与 MiniMind 的连接

本机官方实现先投影、拆头并处理 Q/K，再应用 apply_rotary_pos_emb，随后扩展 KV 头并计算 Attention。其 rotate_half 对前后两半特征配对：四维 [a,b,c,d] 对应 (a,c)、(b,d)，不是默认相邻配对 (a,b)、(c,d)。后续实现必须保持配对方式和 cos/sin 排列一致，才能与该参考对照；本轮二维观察只有一对坐标，暂不涉及两种排列差异。标准基础 RoPE 在这里作用于 Q/K，V 保留内容路径。

### 第一项观察任务

由学习者创建 `scripts/inspect_rope.py`，只实现单个二维向量的 `rotate_2d(x,angle)`，angle 使用弧度。计算两个新坐标：x0*cos(angle)-x1*sin(angle)，x0*sin(angle)+x1*cos(angle)，再堆叠为形状 [2] 的输出，保持浮点 dtype。

使用 float64 的 X=[3,4]，分别观察角度 0 和 pi/2：预期结果为 [3,4] 与 [-4,3]。打印结果，检查形状仍为 [2]，以 assert_close 验证已知值以及输出与输入的平方和相同。两者的平方和均为 25，表示向量长度仍为 5。对已知值比较允许浮点误差，例如 atol=1e-10、rtol=0。

本步只验证二维旋转的计算与范数保持，不宣称已实现完整位置编码或验证相对位置性质。观察通过后，再将角度接到 token 位置，并对照同时平移位置时的 Q/K 点积。

2026-10-09，学习者写出脚本后，实际运行首先在已知值断言处失败：输出 dtype 为 float64，而 `torch.tensor([3,4])` 自动推断为 int64。assert_close 默认也检查 dtype，设置数值容差并不会忽略类型差异；预期张量应显式使用与输入一致的浮点类型。

代码复查还发现第二坐标写成 `x[0]*cos_angle+x[1]*sin_angle`，应为 `x[0]*sin_angle+x[1]*cos_angle`。初版在零角度产生 [3,3]，不能还原 [3,4]，在 pi/2 时产生 [-4,4]，也不符合预期。零角度对照不只是数值例子，它能直接发现没有保持恒等映射的公式错误。平方和应分别与原输入对照，而不只比较两次旋转的结果，避免两个错误结果恰好一致却被误判为范数保持。当前由学习者修正公式、预期 dtype 和范数断言，尚未重新验证成功。

学习者完成修正后，2026-10-09 再次运行 `uv run --locked python scripts/inspect_rope.py`，退出码为 0。零角度与 pi/2 的已知值对照、两种结果的形状 [2]、分别对原输入的平方和对照均通过。当前脚本只有断言，没有打印输出；成功证据是这些断言通过，不是已有完整 RoPE 模块。

### 位置角度与相对点积

下一步沿用 rotate_2d，固定 float64 的 q=[1,2]、k=[3,4] 和标量频率 omega=0.5，按位置构造 q_rot=R(m×omega)q、k_rot=R(n×omega)k，再求两者点积。torch.dot 作用于两个一维向量，逐元素相乘后求和，返回标量；这里对应当前注意力计算中一个 query 与一个 key 的原始分数，暂不加入缩放、mask 或 Softmax。

比较 (m,n)=(2,5) 与 (9,12)：两次保持内容向量及频率不变，绝对位置共同增加 7，位置差均为 3，点积应在浮点容差下相同。再观察 (2,6)，本例位置差改为 4，分数会改变。固定内容后只改变位置的对照，帮助区分位置作用与内容作用；不能把点积分数理解为距离本身，也不能据此声称不同距离必然得到不同分数。

由学习者在已有脚本后追加这个实验，打印三组分数，使用 assert_close 比较前两组，并确认本例第三组与第一组不接近。所有张量保持 float64，前两组比较可使用 atol=1e-10、rtol=0。角度由位置乘频率产生，旋转 Q/K 时使用同一频率；当前只验证一个二维特征对，尚未实现多频率表、批量特征配对或位置编码接入。

学习者完成实验后，同日运行脚本退出码为 0，两项点积分数断言通过。脚本未添加打印，助手另执行脚本并读取返回变量：score_a=2.7730991915528422、score_b=2.7730991915528413、score_c=-2.759020348367203。前两组误差为浮点舍入量级，第三组明显不同，符合固定内容下的相对位置构造。该记录是单个二维特征对的数值观察，不是完整 RoPE 的批量验收。

### 从二维旋转到头特征配对

一个 token 在一个注意力头中只有一个位置编号，却拥有多个特征。RoPE 将偶数宽度的头特征拆成多个二维平面，每个平面使用自己的频率。位置 `m` 是标量，但它与频率向量逐元素相乘后得到角度向量 `angles_i=m*omega_i`；因此“一个位置”并不意味着所有特征对只使用同一个角度。高、低频特征对共同提供不同尺度的位置变化。

本轮对齐 MiniMind 的前后半配对：四维 `X=[a,b,c,d]` 中，二维对为 `(a,c)`、`(b,d)`。为避免把辅助写法误认为额外的旋转步骤，学习者先直接使用二维旋转公式：沿最后一维拆出 `first=[a,b]` 与 `second=[c,d]`，再逐元素计算 `new_first=first*cos-second*sin` 和 `new_second=first*sin+second*cos`，最后沿最后一维拼回原布局。

学习者在 `src/minimind_lab/models/rope.py` 中实现 `rotate_features(x, angles)`。观察样本 `X=[1,2,3,4]` 使用角度 `[0,pi/2]`：第一组 `(1,3)` 保持不变，第二组 `(2,4)` 旋转 90 度，按前后半布局得到 `[1,-4,3,2]`。这里的两个角度是用于检查特征配对的人工测试输入，并非标准 RoPE 频率表。

同日运行 `scripts/inspect_rope.py` 退出码为 0，人工样本的已知值断言通过。助手另用形状 `[2,3,4]` 的 float64 输入检查批量前导维度、输出形状、逐 token 范数与反向传播，结果均通过。正式边界检查和独立测试尚未加入，当前证据只说明直接公式的核心计算正确。

二维公式也可改写为 `rotated=x*cos(angle)+rotate_half(x)*sin(angle)`。其中 `rotate_half` 只是把每个二维坐标的 `[-y,x]` 项批量排列出来，不是先额外旋转 90 度、再进行位置旋转。它与当前直接公式数学等价，等直接实现与频率生成连通后再作为参考实现对照。

### 从位置表到 GQA 的 Q/K

学习者随后实现 `build_rope_angles(seq_len,head_dim,rope_base,dtype,device)`。函数以 `arange(0,head_dim,2)` 为每组二维特征生成频率索引，再将位置列向量 `[T,1]` 与频率行向量 `[1,d/2]` 相乘，得到角度表 `[T,d/2]`。`head_dim=4`、`rope_base=1e6` 时，两组频率为 `[1,0.001]`；位置 0、1、2 对应角度行为 `[0,0]`、`[1,0.001]`、`[2,0.002]`。初版误用会包含终点且已弃用的 `torch.range`，额外产生索引 4 和第三个频率；改用不含终点的 `torch.arange` 后，手工路径与函数路径数值一致。

角度表先作用于序列特征 `[T,d]`，再广播到真实 Attention 布局 `[B,H,T,d]`。`[T,d/2]` 从右侧对齐旋转公式中的 `[B,H,T,d/2]`，因而同一套位置与频率规则可复用于所有 batch 和所有头。观察脚本验证了输出形状、位置 0 恒等、逐 token 范数保持和梯度回传；位置 0 检查与范数检查不可互相替代，因为错误的非零旋转仍可能保持范数。

最后实现 `apply_rope(q,k,angles)`，允许 Q、K 头数不同，对应 GQA 中的 `[B,Hq,T,d]` 与 `[B,Hkv,T,d]`。函数冻结相同 batch、序列长度和 head width，明确不要求 `Hq==Hkv`，并要求角度表严格为 `[T,d/2]`。不同头数下 Q/K 的形状、位置 0、范数及两条梯度路径均通过；错误宽度的角度表被入口合同拒绝。当前仍是直接二维公式版本，尚未完成与 MiniMind `rotate_half` 写法的数值对照和正式 pytest 覆盖。

随后学习者实现 MiniMind 风格的 `rotate_half` 表达。初版外层误写为减法，但 `rotate_half` 的前半部分已经包含二维公式中的负号，导致符号重复翻转；修正为 `x*cos_full+rotate_half(x)*sin_full` 后，它与直接二维公式在真实 GQA 形状上的前向输出一致。使用相同输入副本和相同上游梯度进行反向对照，两条路径的输入梯度也一致。

为控制重复劳动，学习者负责核心公式、形状推导、观察实验与错误修正，助手将已确认的行为整理为 `tests/test_rope.py`。正式测试只冻结高价值合同：已知角度表、非法频率配置、`rotate_half` 布局、两种表达的前向与梯度等价、不同 Q/K 头数以及 `apply_rope` 非法输入。2026-10-10，RoPE 测试为 17 passed，完整项目测试为 145 passed；观察脚本继续作为学习过程记录，不被 pytest 文件逐行复制。

### 接入 GroupQueryAttention

学习者在 `GroupQueryAttention` 中加入 `rope_base` 配置，在 Q/K/V 投影、拆头、KV 扩展和换轴之后生成 `[T,d/2]` 角度表，再于 Attention 点积前旋转 `[B,H,T,d]` 的 Q/K，V 保持原内容路径。初版把默认基数误写为 `1e-6`，使第二组频率从预期的 `0.001` 变为 `1000`；同时构造门禁使用 `<0` 而未拒绝零。修正为 `1e6` 与 `<=0` 后，核心配置符合当前 MiniMind 对照。

接线后原有逐头参考测试出现两项失败：模型实际路径已包含 RoPE，而测试参考路径仍用未旋转的 Q/K。助手将参考路径更新为“投影、拆出原始 Q/KV 头、旋转 Q/K、按 Q 头选择共享 KV 头、逐头调用 PyTorch SDPA、合并并输出投影”。该参考路径先旋转较少的 K 头，模型路径先重复 K 再旋转；两者结果一致，同时证明旋转与相同副本扩展可交换。补充 GQA 构造参数门禁后，Attention 测试为 34 passed，完整项目测试为 152 passed。

## Padding 与可见性

第二周 collator 交付的 `attention_mask` 现在进入模型计算：形状为 `[B,K]`，1 表示实际位置，0 表示补齐位置。causal mask 限制 query 不能读取未来 key，padding mask 限制 query 不能读取补齐的 key；labels 中的 `-100` 则独立控制预测目标是否进入 loss。当前屏蔽 key 列，因此 padded query 仍可能产生非零输出。

### 广播与可选输入

底层 Attention 创建与 scores 同形状的布尔 `allowed`，再与 causal、padding 可见性取逻辑与。scores 为 `[B,H,Q,K]` 时，`[Q,K]` 的 causal mask 可以自然从右侧对齐；`[B,K]` 的 padding mask 却会将 B 对齐到 Q，因此需要显式插入中间的单例维度，得到 `[B,1,1,K]`。广播只按维度长度从右侧匹配，不识别 batch、head 或位置的含义；维度碰巧相等时，错误对齐也可能静默通过。

初版将 `attention_mask.to(torch.bool)`、补维和合并语句放在可选分支外，导致未传 mask 的旧测试对 `None` 调用 `.to()`。学习者修正缩进后，Attention 原有 34 项测试恢复通过；这只能证明默认无 padding mask 的行为保持，新增 mask 的效果还需要独立观察。

### 单独验证 Padding

学习者编写 `scripts/inspect_padding_mask.py`：Q/K 均为零，V 为 `[10,20,1000]`，mask 为 `[1,1,0]`。关闭 causal，以便单独观察 padding 的作用。无 mask 时 scores 为 `[0,0,0]`，Softmax 得到 `[1/3,1/3,1/3]`，每行输出约 343.33；有 mask 时 scores 变为 `[0,0,-inf]`，Softmax 得到 `[0.5,0.5,0]`，每行输出为 15。零分数对应的指数为 1，因此并不意味着零权重。

2026-10-11，脚本实际运行退出码为 0，输出与手算一致；`masked_output==15` 以及第三列权重为零的断言通过。选择 PAD 的 V 为 1000，使错误读取 PAD 的影响容易辨认。接下来沿用该数据开启 causal，预期输出为 `[10,15,15]`，确认最终可见性是两种 mask 的交集。

学习者随后启用 causal 并验证完整权重与输出：第 0 行只能读取位置 0，输出为 10；第 1 行读取位置 0、1，输出为 15；第 2 行排除 PAD 列后仍读取位置 0、1，输出为 15。实际观察脚本运行通过。特征宽度不决定 scores 的列数：Q/K 点积的特征轴被求和，保留下来的是 query 与 key 的位置轴；Softmax 沿 key 轴生成读取权重，随后与 V 相乘恢复内容特征宽度。

### GQA 接线与回归

学习者在 `GroupQueryAttention.forward` 增加可选的 `[B,T]` mask，并将它原样传给底层 Attention。底层统一完成补维和可见性组合，避免模块重复处理。助手整理 `tests/test_attention_padding.py`，以两种 causal 模式覆盖已知值、三维与四维 Attention 对照 PyTorch 的前向及梯度、不同有效长度下 GQA 与去除 PAD 的逐样本输出对照、PAD 内容扰动以及形状门禁。使用 B=2、T=4 避免 batch 与 query 长度相等掩盖广播错误；GQA 采用 d=4，以覆盖多个 RoPE 特征频率。

新增测试 10 passed，全项目 162 passed。有效 query 的目标反向传播后，PAD 输入梯度为零，所有投影参数梯度有限。这里验证的是当前右侧 padding 且每条样本至少包含一个实际 token 的情况。另行诊断全零 mask 时，权重与输出均出现非有限值；该场景尚未被入口拒绝，下一步补齐“组合可见性中每个 query 至少有一个可读 key”的合同。

### 无可读 Key 的拒绝检查

学习者在两种 mask 合并后、填充负无穷与 Softmax 之前，加入 `allowed.any(dim=-1)`。最后一维是 key 轴，因此结果逐行说明每个 query 是否至少能读取一个 key；随后 `.all()` 检查所有样本、所有头与所有 query。任意一行没有可读 key，立即抛出 ValueError，避免整行负无穷进入当前手写 Softmax 后产生 NaN。

检查不能只看原始 padding mask 是否存在 1。例如 `[0,1,1]` 虽然有两个有效 key，但开启 causal 时，第 0 个 query 不能读取未来的 key，第 0 个 key 又被 padding mask 排除，最终仍是全屏蔽行。关闭 causal 后，同一输入可以读取位置 1、2，计算应正常通过。

助手补充 6 项回归：三维与四维输入、两种 causal 模式下拒绝全零样本；以同一 padding mask 对比 causal 拒绝与 noncausal 正常计算；检查 GQA 是否传递底层错误。2026-10-11 实际运行，padding 测试为 16 passed，完整项目为 168 passed。

当前输入合同是非空、右侧 padding 的序列；检查覆盖所有 query，包括补齐位置。因果计算中带有全屏蔽早期 query 的左侧 padding 输入会被拒绝，不能据此宣称已经支持左侧 padding 或所有框架的全屏蔽行处理方式。

## 理论衔接

实践中的疑问集中在以下几组关系，详细推导已进入理论正文，不在这里重复整篇：

| 疑问 | 理解要点 | 对应位置 |
| --- | --- | --- |
| Q/K/V 宽度为何这样设置 | 模型隐藏宽度、头宽度与 FFN 中间宽度是不同层次；Q/K 点积宽度必须匹配 | [Q、K、V 的生成](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/01_ATTENTION_MECHANISM.md#4-qkv-的生成) |
| 内部 AV 为什么不是 hidden_size | AV 是内部聚合，完整模块还会拼接与输出投影 | [输出投影与模型连接](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/01_ATTENTION_MECHANISM.md#7-输出投影与模型连接) |
| 换轴为何必要 | 把 token 与头放在正确的矩阵/批量轴上，决定 matmul 在哪些对象之间发生 | [输出投影与模型连接](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/01_ATTENTION_MECHANISM.md#7-输出投影与模型连接)，后续多头篇详解 |
| 同宽度 o_proj 有何作用 | 维度相同仍能学习混合特征，不是恒等映射 | [输出投影与模型连接](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/01_ATTENTION_MECHANISM.md#7-输出投影与模型连接) |
| Attention 如何学会读取 | 参数由 loss 梯度更新，注意力权重在每次前向由输入重新计算 | [Attention 的学习过程](../../../../model_architecture_and_algorithms/01_model_architecture/components/attention/01_ATTENTION_MECHANISM.md#8-attention-的学习过程) |

## Decoder Block 装配

组装完整 Decoder Block，让跨位置读取与逐位置特征加工构成一次完整的状态更新：

```text
x [B,T,D]
-> RMSNorm -> GQA（含 RoPE、causal 与 padding mask）
-> 与原 x 相加，得到 h [B,T,D]
-> ResidualFeedForward(h)
   -> 独立的 RMSNorm -> SwiGLU -> 与 h 相加
-> 输出 [B,T,D]
```

Attention 把可读位置的内容汇入当前 token；SwiGLU 再加工该位置已经汇入的信息。两个分支各有自己的归一化参数，残差保留各分支输入并叠加新增特征。输入与输出维度保持一致，之后才能连续堆叠多个 Block。

学习者在 `models/block.py` 中注册 Attention 前的 RMSNorm、GroupQueryAttention，以及已有 ResidualFeedForward。初版将最后一个模块写成单独的 SwiGLU；它不包含第二条分支的归一化与残差，随后改成 ResidualFeedForward，复用第三周已实现的组合。

forward 先计算 `x_normed=attention_norm(x)`，将 causal 与 attention_mask 传给 Attention，并解包特征输出与权重。第一条残差使用原始输入：`h=x+attention_output`。随后调用 `feed_forward(h)`，由该模块内部完成第二次归一化、SwiGLU 与残差相加，外部不重复加 h。最后返回 output 与 Attention 权重。

RMSNorm 在这里控制分支入口的数值尺度，而不是给 loss 加惩罚。它沿每个 token 的最后一维计算均方根，除以共同尺度后再乘可学习系数，不改变 `[B,T,D]` 形状，也不要求特征均值为零。两个分支的归一化参数相互独立，残差旁路保留各自的未归一化输入。

2026-10-11，助手用 `DecoderBlock(16,32,4,2)` 做 CPU 小输入检查。输入为 `[2,4,16]`，两条样本的有效长度为 4 与 2；输出为 `[2,4,16]`，权重为 `[2,4,4,4]`。仅对有效 query 的输出计算平方均值并 backward，输入与全部参数均获得有限梯度。本次是临时接线检查，不计入正式 pytest 数量，也不代表完整模型验收。

### 残差观察与正式测试

学习者完成 `scripts/inspect_decoder_block.py`：正常调用打印 `[2,4,16]` 的输出与 `[2,4,4,4]` 的权重，随后在 no_grad 中将 Attention 的 o_proj 和 SwiGLU 的 down_proj 权重置零，再次计算并断言输出等于 x。2026-10-11 实际运行退出码为 0，形状与恒等输出均符合预期。两个输出投影都没有 bias，置零使分支新增量消失，但保留了输入残差旁路；这不是把整个 Block 的参数全部清零。

助手整理 `tests/test_decoder_block.py`，共 8 项测试：两种 causal 模式的形状、有限梯度与独立归一化参数；分别关闭 Attention、前馈或两条分支的残差对照；改变未来输入时过去输出不变；两种 causal 模式下带 PAD 的 batch 与去除 PAD 的逐样本输出对照。最后一组同时检查 PAD 扰动不影响有效输出，仅对有效 query 计算目标时 PAD 输入梯度为零。两条分支均关闭时还验证 `output.sum()` 对 x 的梯度为全 1，说明恒等旁路不仅保留数值，也能传递梯度。

本次 Block 测试为 8 passed，完整项目为 170 passed。复查发现此前记录的 6 项无可读 key 边界测试已不在当前 `test_attention_padding.py` 中，该文件当前为 10 项；Attention 中的核心拒绝检查仍在。因此历史 168 passed 不能与本次新增项直接相加。未擅自恢复缺失测试，当前无可读行的正式回归覆盖需要在模型级验收前补齐。

### 两层串联观察

沿用观察脚本，新建两个独立的 DecoderBlock，放进 `nn.ModuleList`。不要重复使用已经清零的 block，也不要通过 `[block]*2` 重复引用同一个实例。以新的 x 作为 hidden，逐层调用并把上一层输出交给下一层，观察每一层的输出仍为 `[B,T,D]`。

ModuleList 负责注册子模块，使后续父模型能收集它们的参数、保存状态与迁移设备；它不自动执行这些层，因此仍由 forward 或观察脚本中的循环定义数据流。每层是同结构、独立参数的模块，允许逐层加工特征。完成串联观察后再装配模型主干，而不是立即增加新的训练或生成逻辑。

学习者用列表推导式在每次迭代新建 `DecoderBlock(16,32,4,2)`，放入 ModuleList，再通过 `hidden,_=block(hidden)` 将上一层输出交给下一层。2026-10-11，观察脚本运行通过，两层输出均为 `[2,4,16]`。助手另检查两层参数对象集合没有交集，并且 ModuleList 收集的参数恰好包含两层的参数。这验证了独立实例与模块注册，未增加新的正式 pytest，也未验证层堆叠的完整训练或语言模型输出。

### 模型配置

学习者在 ModelConfig 中增加 num_layers、num_heads、kv_heads 与 rope_base，并逐项拒绝非正值。初版默认值为两层、两个 Q 头、两个 KV 头与基数 1e6；Q/KV 头数相同是合法的普通多头特例，不再具有跨 Q 头共享 KV 的压缩效果。随后学习者将 num_heads 改为 4，恢复此前观察的四个 Q 头、两个 KV 头设置。

助手扩展现有配置参数化测试，覆盖新增字段的零值和负值，以及自定义配置能否保存传入值。2026-10-11，`tests/test_config.py` 为 17 passed，完整项目为 178 passed。配置目前做正数检查，头数整除与 RoPE 头宽度约束仍由构造 GroupQueryAttention 时检查。

### 主干构造与配置传递

学习者在 `models/model.py` 实现 MinimindBackbone 的构造方法，保存 config 并注册 TokenEmbedding、ModuleList 和末端 RMSNorm。初版创建 DecoderBlock 时只传前四个参数，遗漏 eps 与 rope_base。默认配置下无法暴露这个问题；自定义 eps=1e-6、rope_base=10000 时，末端归一化使用配置值，Block 内部却仍使用默认 1e-5 与 1e6，造成配置与实际计算不一致。

补齐两个关键字参数后，助手用三层非默认配置检查全部 Block 的两处归一化、RoPE 基数及参数独立性。该行为保存为 `tests/test_model.py` 的构造测试，实际运行 1 passed，并检查父模型能够收集全部层参数。此测试只验证装配，不调用尚未实现的主干 forward。

### 主干 Forward 与回归

学习者完成 MinimindBackbone.forward，形成 `input_ids -> TokenEmbedding -> 多层 DecoderBlock -> 末端 RMSNorm -> hidden_states`。先查表得到 hidden，再循环调用各个 Block，将上一层输出交给下一层，并原样传递 causal 与 attention_mask。解包 Block 输出时暂不收集注意力权重，只保留 hidden；最后调用一次 final_norm 并返回 `[B,T,D]` 的最终特征。末端归一化位于所有 Block 之后，不替代各个 Block 分支入口的归一化。

助手增加 6 项主干回归，连同已有构造测试共 7 项：两种 causal 模式下对照逐组件路径、输出形状与全部参数的有限梯度；两种 causal 模式下带 padding 的 batch 与去掉 PAD 的逐样本结果对照；未来 token 扰动不影响过去输出；全屏蔽样本向上层传递 ValueError。PAD 内容变更不影响有效位置，且仅对有效输出计算目标时，未在有效位置出现的 PAD token 对应 embedding 行梯度为零。

反向测试使用随机上游梯度，而非仅依赖最终输出平方和。归一化可能使平方范数目标接近常数，单独用它不适合证明特征路径产生了有意义的梯度。逐组件对照验证装配顺序，不被当作底层 Attention 算法的独立参考；后者已有对应的 PyTorch 对照。2026-10-11 实际运行 `tests/test_model.py` 为 7 passed，完整项目为 185 passed。

### 词表输出与 Linear 的数据流

学习者新增 MinimindForCausalLM，forward 将主干输出送入 lm_head，返回 `[B,T,V]` 的 logits。初版 Linear 没有指定 bias，默认创建了偏置参数；随后补上 bias=False，与当前 MiniMind 输出层设置对齐。

无 bias 的 Linear 对每个位置的 D 维特征 h 与每个候选 token 的权重行 W[j] 做点积，得到一个标量分数 `logits[b,t,j]=sum_i(hidden[b,t,i]*W[j,i])`。nn.Linear(D,V) 的权重为 `[V,D]`，整个词表的计算合并为 `hidden @ W.T`：`[B,T,D] @ [D,V] -> [B,T,V]`。D 是被求和消去的特征轴，B/T 保留，V 是新产生的输出特征轴；同一份权重应用于所有样本和位置，不在此处混合它们。

因此 Linear 的每个输出分量是一次点积；带 bias 时还要加该分量的偏置，整个模块不是只返回一个点积标量。Embedding 则使用已知 ID 选择参数表中的一行，输出层使用上下文特征对所有候选行打分，不是反向查表。这里的分数尚未经过 Softmax，也没有完成 token 选择或生成。

2026-10-11，助手用 V=20、D=16、输入 `[2,3]` 运行模型，得到特征 `[2,3,16]`、输出权重 `[20,16]` 与 logits `[2,3,20]`。模型 logits 与 `hidden @ weight.T` 一致，单独取 b=0、t=1、j=7 的点积也与对应 logit 一致。该检查是临时数值对照，尚未加入输出层正式回归。

### 权重共享与赋值语义

学习者在输出层构造后加入 `self.lm_head.weight=self.backbone.embedding.weight`。两个模块由此引用同一个 nn.Parameter，不是两份数值相同的独立权重。助手补充输出形状、矩阵与逐 token 点积对照、参数对象身份及 model.parameters 中去重检查。

为验证两条路径的梯度汇总，测试复制一个数值相同的参考模型，并将其输出权重替换为独立的 Parameter。两个模型使用相同输入、相同上游梯度，前向 logits 一致；共享模型的词表参数梯度等于独立参考的输入 embedding 梯度与输出投影梯度之和。参考中未被输入查表使用的 token 19 行，其输入梯度为零，但输出路径仍贡献梯度；因此不能将早先纯 Embedding 的“未查行梯度为零”直接套到完整共享语言模型。

2026-10-11，模型相关测试为 9 passed，完整项目为 187 passed。测试使用人工上游梯度验证链路，尚未构成语言模型交叉熵训练验收。

本次疑问是“何时直接赋值、何时原地修改、何时 detach/clone”。理解重点是区分对象引用、数据存储与求导连接：共享参数用同一个 Parameter 对象；设置已有参数数值用 no_grad 中的 copy_ 或初始化操作；保留可导关系的数据副本用 clone；独立数值快照用 detach().clone()。这些写法的区别及优化器引用风险已补入[自动求导与参数更新](../../../../model_architecture_and_algorithms/01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md#引用数值与梯度连接)。

### 语言模型 Loss 与有效目标

学习者在 `models/loss.py` 实现 causal_lm_loss，接收 `[B,T,V]` 的原始 logits 与 `[B,T]` 的 labels。当前 collator 保持实际位置的 labels 与 input_ids 同位置，因此 loss 函数使用 `logits[..., :-1, :]` 与 `labels[..., 1:]` 对齐下一 token 目标。第一项标签不参与比较，最后一个位置的 logits 不参与本批目标；EOS 仍是有效目标，PAD 标签为 -100，被交叉熵忽略。causal mask 管可见性，causal shift 管预测与标签对齐，不能互相替代。

切片后通过 reshape 将分数整理为 `[B*(T-1),V]`、标签整理为 `[B*(T-1)]`，传给 F.cross_entropy，使用 ignore_index 与 reduction="mean"。函数不先做 Softmax，返回的是所有有效目标 token 的平均负对数似然，而不是先求每条样本均值再对样本平均。当前输入约定是形状匹配、合法目标 ID 的批量张量；函数显式拒绝 T<2，以及 shift 后整个 batch 没有有效目标的情况，尚未覆盖全部错误 dtype、rank 或 shape 的入口诊断。

实现过程中先遇到 functional 大小写与中文逗号问题，随后遇到全忽略目标的判断错误。直接对标签数值调用 all() 并不表示所有标签等于 ignore_index，需要先构造布尔比较。学习者最终使用 `ignored=shift_labels==ignore_index`，再以 `ignored.all()` 检查整个 batch。中间版本的 `all(dim=-1)` 返回每条样本各一个布尔值，B>1 时无法直接放入 Python if；即使通过某种额外转换，也必须区分逐样本拒绝与整个 batch 拒绝。一个样本没有有效目标而其他样本仍有目标时，当前 loss 合同允许计算。

助手整理 `tests/test_loss.py`，12 项测试覆盖以下行为：

- 全零 logits 对应均匀概率，平均 loss 为 log(V)，用于手算对照。
- 默认与自定义 ignore_index 下，loss 及 logits 梯度与逐 token 的负 log_softmax 目标相同；两条样本含 3 与 1 个目标，验证全局 token 平均。
- 改变被丢弃的首标签、最后位置 logits 和忽略目标对应的 logits，不改变 loss；最后位置及被忽略目标位置的 logits 梯度为零。
- batch 中可以包含无有效目标的单条样本；但整个 batch 在 shift 后没有有效目标时拒绝，即使 shift 前首标签有效也仍拒绝。
- 序列长度 0 或 1 被拒绝。
- 用已有 PretrainCollator 构造两条人工 token 样本，经共享权重模型产生 logits，计算 5 个有效目标的 loss，并对全部参数回传有限梯度。

2026-10-11，loss 测试为 12 passed，完整项目为 199 passed。最后一项是合成 token 样本的集成测试，不是 JSONL/tokenizer 全链路验收，也不是完成参数更新；下一步在真实数据脚本中亲自观察这些连接。

### 真实数据链路验收

学习者沿用第二周的 `scripts/inspect_pretrain_batch.py`，在已有 batch 检查之后追加小模型、forward、causal_lm_loss 与 backward。词表大小使用 len(tokenizer)，隐藏宽度 16、中间宽度 32，层数和头数使用当前配置默认值。模型接收 input_ids 与 attention_mask，loss 独立接收 logits 与 labels。

换到 Windows 后，来源清单存在，但被 Git 忽略的样本与 accepted.jsonl 未迁移。助手按原 revision 恢复两份 20 条来源样本并核对 SHA256，再重新执行审计；两份均为 20 条通过、0 条拒绝。恢复入口见[样本恢复](../README.md#样本恢复)。没有改用新数据，也没有代写学习者的模型或 loss。

2026-10-11，助手运行学习者的观察脚本，input_ids、attention_mask 与 labels 均为 `[2,512]`；两条样本分别有 355、137 个真实位置，shift 后有效目标分别为 354、136，共 490 个。模型输出为 `[2,512,6400]`，本次 loss 为 9.3970479965，共享词表参数梯度范数为 0.3849409819，均为有限值。数值来自未固定随机种子的初始化模型，不作为下一次运行必须相等的验收标准。

这次观察验证了真实数据到反向传播的连接，不代表参数已经更新或模型已经学会生成。脚本没有执行 optimizer.step，也尚未显式接入第三周的教学初始化规则。

### 初始化接入与观察

学习者在创建模型前调用 `torch.manual_seed(42)`，创建模型后、前向之前调用
`model.backbone.apply(initialize_weights)`。apply 遍历主干的子模块，已有初始化函数按模块类型设置参数；
当前输出头与 embedding 共享同一 Parameter，且无 bias，因此初始化主干也覆盖了输出头的全部参数。
这避免对共享权重重复随机抽样，不是任意独立输出头都可省略初始化的通用规则。

2026-10-11，助手运行修改后的真实 batch 脚本，embedding 权重标准差为 0.0199819282，
共享参数身份检查为 True，末端 RMSNorm 的 16 个缩放参数均为 1。loss 为 8.7685794830，
共享参数梯度范数为 0.2825376391，均为有限值。当前打印在 backward 之后仍可观察初始权重，
因为 backward 只累积梯度，没有改变参数值。本次没有执行 optimizer.step，也没有新增 pytest。

### 一次参数更新与梯度尺度

学习者创建 AdamW，设置 lr=1e-3、weight_decay=0.0，保存共享词表参数的 detach().clone() 快照，
随后调用 optimizer.step 并比较最大绝对差值。当前优化器创建在 backward 之后也能读取已有 .grad，
快照仍在 step 之前，因此这次单步观察有效；完整循环中会在循环外创建优化器，反复复用其状态。

2026-10-11，助手运行学习者脚本：最大参数变化为 0.0010000002，更新后共享关系仍为 True。
额外临时打印得到这份参数的最大绝对梯度为 0.0366102718，平均绝对梯度为 0.0001492790。
临时观测未写入学习者脚本，也没有新增 pytest。这次证明参数实际发生更新，不证明训练能力已提升。

学习者问“变化为 0.001 是否说明梯度接近 1”。对于无动量、无衰减的普通 SGD，变化等于
学习率乘梯度；但 AdamW 使用一阶、二阶矩估计。当前第一次更新的状态从零开始，偏差校正后
一阶量为 g、二阶量为 g²，因此每个元素的更新为 `-lr*g/(abs(g)+eps)`。在绝对梯度远大于 eps 时，
更新幅度接近 lr，不能反推原梯度接近 1；零梯度则没有这项更新。该近似不能直接推广到已有状态的后续步。

### 更新后的重新前向

学习者在 optimizer.step 之后，使用同一个 input_ids 与 attention_mask，在 torch.no_grad 中重新调用模型，
再使用相同 labels 计算 after_loss。2026-10-11，助手运行脚本：更新前 loss 为 8.7685794830，
更新后为 8.7521066666，下降约 0.01647。当前模型没有 dropout，因此这次比较不涉及训练态与评估态的
dropout 差异；无梯度前向用于观察，不会再次 backward 或更新参数。

更新前保存的 logits 和 loss 不会随参数变化自动刷新，必须重新前向才得到新参数的预测。
本次下降说明同一 batch 的平均目标 NLL 在这次更新后改善，不证明验证集泛化、自然语言生成质量或
稳定收敛。正式训练、验证和完整优化器学习仍在后续周次。本次没有新增 pytest。

### 单 Token 生成与维度保持

学习者取第一条真实样本前 8 个 token，构造全 1 attention_mask，前向后使用
`prefix_logits[:, -1:].argmax(dim=-1)` 选择最后位置的最高分 token，再与前缀沿序列轴拼接。
2026-10-11，助手运行脚本：prefix_logits 为 `[1,8,6400]`，next_token 为 `[1,1]`，
extended_ids 为 `[1,9]`，维度与追加行为正确。本次未新增 pytest，也不评价未充分训练模型的文本质量。

`-1:` 是保留序列维的切片，结果为 `[1,1,6400]`，随后 argmax 消去词表维，得到 `[1,1]`；
它与先用 `[:, -1, :]` 得到 `[1,6400]`、再 argmax 并 keepdim=True 的结果等价。
初版生成前向漏掉 no_grad，临时检查显示 prefix_logits.requires_grad 为 True。
需要补上生成前向的无梯度保护，避免为不反向的预测保留不必要的计算图。
学习者随后将前缀构造与模型前向放入 torch.no_grad。2026-10-11，助手重新运行脚本，
prefix_logits.requires_grad 为 False，extended_ids 仍为 `[1,9]`，无梯度保护复核通过。

### 固定次数的自回归循环

学习者从 prefix_ids.clone() 重新开始，设置最多新增 5 个 token，并在 model.eval 与 torch.no_grad 下循环。
每轮使用当前 generated_ids 和 torch.ones_like(generated_ids) 前向，取最后位置 argmax，再将拼接结果
赋回 generated_ids。2026-10-11，助手运行脚本：前缀为 `[1,8]`，结果为 `[1,13]`，实际新增 5 个 token；
最后一次前向输出为 `[1,12,6400]`，requires_grad 为 False。这与第 5 轮先读长度 12、再追加到 13 的顺序一致。

本次循环正确反馈了模型自己的输出，并保持 mask 与增长后的序列等长。它尚无 EOS 停止条件，
固定次数执行结束不等于模型主动结束。当前 tokenizer.eos_token_id 实际为 2；下一步在追加 token 后判断该 ID。
本次未新增 pytest，也不使用未充分训练模型的重复输出评价算法正确性或训练效果。

### EOS 判断接入

学习者在每次拼接之后比较 next_token.item() 与 tokenizer.eos_token_id，相等时 break，
保留生成的 EOS，同时仍由最大新增次数兜底。2026-10-11，助手运行脚本，新增数为 5，
每轮输出长度依次为 9、10、11、12、13。模型在本次前缀后连续选择“天”，没有选中 EOS，
所以本次仅验证正常循环及上限路径；提前停止分支的行为将在函数提取后通过可控模型测试。
不能把代码审查通过或没有异常等同于 EOS 分支已经触发验证。本次未新增 pytest。

### 生成函数提取与行为测试

学习者将循环提取为 models/generation.py::greedy_generate，输入模型、前缀、最大新增次数和可选 EOS ID，
输出包含原前缀的完整 token 序列。函数内部使用 no_grad，调用方管理 eval 模式；每轮仍读取完整增长后的序列。

助手编写 tests/test_generation.py，共 16 项测试。可控 ScriptedLM 按预先指定的 token 顺序输出 logits，
并让非最后位置偏好其他 token，避免随机模型碰巧触发停止，也能检查是否取错预测位置。
测试覆盖完整前缀反馈、全 1 mask 更新、无梯度前向、第一步和第二步 EOS 停止、未配置 EOS 时继续生成、
零新增次数的独立副本、六种非法前缀、负数预算诊断、训练/评估模式与已有参数梯度保持、函数无打印，
以及真实 tiny 模型与手写三步 greedy 路径的对照。

2026-10-11，助手执行 `uv run --locked pytest tests/test_generation.py -q`，结果为 13 passed、3 failed。
EOS 两条分支、零新增次数、真实模型参考和状态保持均通过。失败项分别是：

- 标量输入的 shape 为空，当前代码先访问 shape[0]，在 ndim 检查之前抛出 IndexError；应先检查 ndim。
- 负数预算已被拒绝，但报错仍称至少为 1，与允许 0 的约定不符；文字应为 non-negative 或 at least 0。
- 函数内部仍打印 step，违反由观察脚本负责打印的接口约定；应移除该打印。

这些核心实现修正留给学习者。测试由助手整理，尚未运行全项目回归，不能把新增项直接计入全项目通过数。

### 生成回归通过与已有边界覆盖补齐

学习者将 ndim 检查放在 shape 索引之前，修正负数预算提示并删除函数打印。
2026-10-11，助手复跑生成测试，结果为 16 passed；全项目当时为 215 passed。
随后助手在 test_attention_padding.py 中补上此前已记录的 6 项覆盖缺口：单头/多头分别检查
全屏蔽样本在因果/非因果模式下被拒绝，以及左 padding 导致首个 query 没有可读 key 时被拒绝。
这些测试保护已有拒绝约定，没有修改 Attention 核心实现，也没有增加左 padding 支持。
最终执行 `uv run --locked pytest -q`，结果为 221 passed in 7.28s。

生成函数的提取及已有接口测试至此通过。生成器仍只支持单条无 padding 前缀，不含采样、KV cache
或批量停止处理。参数更新与生成接通不等于模型已学会一项任务，因此本周最后用 tiny 过拟合检查可学习性。

### Tiny 过拟合初次运行与索引类型错误

学习者编写 inspect_tiny_overfit.py，使用人工序列 `[1,3,4,2]`，初始化模型后以 AdamW 重复更新
100 次，每轮清空梯度并重新前向。2026-10-11，助手复现运行：首轮更新前 loss 为 2.9990031719，
100 次更新后的重新前向 loss 为 0.0038795925，说明这一条序列的训练目标已经得到很好拟合。
生成对照尚未通过：脚本传入 `torch.Tensor([[1]])`，构造的是浮点张量，随后在 embedding 查表索引处
抛出 IndexError，退出码为 1。本次失败不是梯度或优化循环失败，不能据此把整个实验记为通过。

需要由学习者将生成前缀改为 `torch.tensor([[1]], dtype=torch.long)`。小写 tensor 按数据推断类型，
整数列表通常得到 int64；大写 Tensor 在这里构造浮点张量。显式 long 表达的是词表整数索引，
不是 token 特征向量；查表之后得到的可训练特征才是浮点数。修正后仍需重跑生成，检查目标序列与 EOS。

### Tiny 过拟合与生成对照通过

学习者将生成前缀改为小写 torch.tensor，整数列表得到 int64，同时将生成上限设为 9。
2026-10-11，助手重新执行 `uv run --locked python scripts/inspect_tiny_overfit.py`，退出码为 0。
实际设置为 CPU、随机种子 42、词表大小 20、hidden_size=16、intermediate_size=32，
其余层数与头数沿用配置默认值；使用已有初始化规则和 AdamW，lr=1e-2、weight_decay=0.0，
对 `[1,3,4,2]` 重复更新 100 次，每轮清空梯度并重新前向。

关键输出为：

```text
首轮更新前 loss: 2.9990031719207764
100 次更新后的 loss: 0.0038795925211161375
generated_ids: tensor([[1, 3, 4, 2]])
```

生成只给前缀 `[1]`，后面的 3、4、2 由模型逐次选择并反馈为下一轮输入。
最多允许新增 9 个 token，实际新增 3 个，并将 EOS=2 保留在结果中后停止。
所以这次同时验证目标记忆、从短前缀自回归生成与真实模型的 EOS 提前停止，不是达到次数上限才结束。
这些数字是人工 token ID，不承诺对应真实 tokenizer 的文本语义。

之后助手再次执行 `uv run --locked pytest -q`，全项目为 221 passed in 13.30s。
tiny 过拟合是独立观察脚本的运行证据，不被算作额外 pytest，也不增加通过数量。

## 第四周验收

第四周至此完成约定的教学规模模型验收：

| 内容 | 证据 | 结论边界 |
| --- | --- | --- |
| Attention、RoPE、GQA、mask、Decoder 与主干 | 对应数值、梯度、配置与行为测试通过 | 不宣称复现了官方全部运行或性能行为 |
| LM head、权重共享、causal loss | 输出映射、共享梯度与有效目标测试通过 | loss 计算正确不等于已获得领域能力 |
| 真实 JSONL 到一次参数更新 | 真实 batch 脚本运行通过，更新后 loss 降低 | 不代表真实数据完成了正式预训练 |
| 最小 greedy 生成 | 16 项测试通过，EOS、上限、零新增次数均有检查 | 仅支持单条无 padding 前缀，无采样或 KV cache |
| Tiny 目标可学习性 | loss 约 2.99900 到 0.00388，生成 `[1,3,4,2]` 并停在 EOS | 证明极小样本记忆，不证明泛化或聊天质量 |

学习者编写核心算法、模型装配、观察脚本与短循环；助手负责讲解、复核运行、重复性测试及文档整理。
本周不继续增加生成功能。第五周将把短循环发展为单设备训练系统，深入 AdamW、参数组、学习率调度、
有效 token 平均、梯度累积与裁剪；checkpoint 与恢复仍按路线留到第六周。

已核对本机官方 `minimind/model/model_minimind.py`：官方默认共享 embedding 与 lm_head 权重，学习版已接入。后续还需结合已有 initialize_weights 做初始化对照，并处理共享参数的初始化路径，当前不宣称已完成官方全部模型行为复现。

## 环境

本机已建立 Python 3.12.13、PyTorch 2.14.0+cu130 的 uv 环境，RTX 4060 Ti CUDA 可用。
基础组件的 GPU 前向、反向、AdamW 更新已做环境 smoke；现有测试在 Windows 上为 101 passed。
安装与跨平台说明见[案例 README](../README.md#环境)。这些是环境可用性的记录，不是第四周 Attention 的验收。
