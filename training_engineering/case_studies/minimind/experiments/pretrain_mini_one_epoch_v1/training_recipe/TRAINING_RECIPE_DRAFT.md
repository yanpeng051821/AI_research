# MiniMind mini 单轮预训练配方草稿

> 状态：历史设计记录。2026-10-25 的最终采用版本与冻结产物见上级 [README](../README.md) 和本目录 [训练合同](TRAINING_CONTRACT.md)；本文保留讨论过程，不作为启动依据。

## 1. 本轮实验回答什么

本轮从随机初始化的 MiniMind-3 Dense 模型出发，使用已经冻结的 mini 预训练数据，在自己的单设备训练工程中完成一次可恢复、可验证、可分析的预训练实验。

核心问题是：

> 在固定数据、模型和一个 epoch 的训练预算下，训练链路能否稳定完成，并让模型对未参与参数更新的同分布验证文本产生更低的 token 加权 NLL？

本轮不以获得实用聊天模型为目标。预训练只直接优化 next-token prediction；对话格式、指令遵循和工具调用能力属于后续 SFT 或其他后训练阶段。

## 2. 已确认的实验输入

### 2.1 数据合同

正式训练只能引用冻结数据合同 `pretrain_mini_data_v1`：

| 项目 | 固定身份 |
| --- | --- |
| 数据合同 | `../data_recipe/DATA_CONTRACT.md` |
| 合同 SHA256 | `6f8c1c6201f230c4c093f7b51860d99c73c3e49789634aa63ccea9633e7d158a` |
| 冻结清单 | `../data_recipe/data_contract_freeze.json` |
| 训练记录 | 1,265,235 |
| 验证记录 | 5,000 |
| 序列最大长度 | 768 |
| 训练有效 targets | 301,810,690 |
| 验证有效 targets | 1,206,758 |

训练入口必须在加载模型前核对合同、数据文件、编码报告和 tokenizer 身份。路径可以因机器而变化，文件内容身份不能变化。

### 2.2 起始模型

本轮使用已经完成 CPU FP32 官方对照的 MiniMind-3 Dense 结构：

| 项目 | 当前选择 |
| --- | --- |
| 初始化 | 从随机参数开始，不加载预训练 checkpoint |
| 词表大小 | 6,400 |
| hidden size | 768 |
| FFN intermediate size | 2,432 |
| Decoder 层数 | 8 |
| Query heads / KV heads | 8 / 4 |
| head dim | 96 |
| Q/K Norm | 开启 |
| embedding 与 LM head | 权重共享 |
| 独立参数量 | 63,912,192 |

模型初始化 seed、参数 dtype 和初始化状态 SHA256 仍要在训练合同中冻结。这里确认的是模型结构与“从随机初始化开始”，不是完整初始化身份。

## 3. 一个 epoch 的精确定义

本轮正式预算确定为一个 epoch。这里的一个 epoch 不是“大约训练一轮”，而是：

1. 从 `committed_position=0` 开始；
2. sampler 对冻结训练集的全部 1,265,235 个索引生成一次固定排列；
3. 每个索引恰好提交一次，不重复进入第二轮；
4. 全部提交后，`committed_position=1,265,235`；
5. 累计参与 loss 的有效 target 总数应为 301,810,690。

micro-batch 大小和梯度累积会改变 optimizer update 数量，但不改变一个 epoch 应消费的记录数和有效 target 总量。最后不足一个完整累积窗口的数据仍要执行一次参数更新，不能静默丢弃。

暂停和恢复不产生新的 epoch。恢复后继续消费原来固定排列中尚未提交的部分，直到完整预算结束。

## 4. 随机状态与复现边界

### 4.1 固定选择

本轮把数据划分、模型初始化和训练顺序作为不同的随机过程管理：

| 项目 | 固定值 | 作用 |
| --- | ---: | --- |
| `split_seed` | 43 | 已在数据合同中冻结，只负责训练/验证划分 |
| `model_seed` | 42 | 在模型构造前设置，固定随机初始化参数 |
| `sampler_seed` | 42 | 传给独立 `torch.Generator`，生成单 epoch 的索引排列 |
| `num_workers` | 0 | 避免多 worker 预取与随机状态增加恢复复杂度 |
| deterministic algorithms | 开启 | 在 PyTorch 支持范围内拒绝非确定性算法 |
| CUDA matmul / cuDNN TF32 | 关闭 | 避免 TF32 改变本轮数值路径 |

`model_seed` 与 `sampler_seed` 数值相同，但语义独立。sampler 使用自己的 Generator，不消费模型初始化所用的全局 RNG；以后可以只改变其中一个做对照实验。

### 4.2 新实验与恢复

新实验从 seed 创建初始状态：

```text
设置 model_seed
→ 构造随机初始化模型
→ 记录初始化 state_dict SHA256

设置 sampler_seed
→ 对 1,265,235 个索引生成一次完整排列
→ committed_position = 0
```

断点恢复不能重新设置 seed 或重新执行 `randperm`。checkpoint 必须恢复：

- sampler 的完整 `indices` 和 `committed_position`；
- PyTorch CPU RNG state；
- 全部可见 CUDA 设备的 RNG state；
- 与数据位置共同推进的 model、optimizer、scheduler 和 global step。

当前正式训练路径不使用 Python `random` 或 NumPy RNG 做训练期随机决策，因此暂不把它们写入 checkpoint。后续如果加入随机数据处理或其他相关逻辑，必须先扩展保存与恢复合同。

### 4.3 复现要求

本轮分三层定义可复现：

1. **配置与语义一致**：数据、tokenizer、模型、初始化身份、sampler 排列、训练配置和预算一致，这是所有运行的必要条件。
2. **同机精确恢复**：连续训练与“暂停、退出、恢复”路径的模型和完整训练状态必须一致；正式运行前先用短跑验证。
3. **跨硬件可比较**：不同 GPU、CUDA、驱动或 PyTorch 环境不要求 checkpoint 浮点值逐位相同，但必须使用相同合同并报告环境与指标差异。

当前模型没有 Dropout，数据路径没有随机增强，因此模型初始化和样本排列是主要随机来源。仍然保存 CPU/CUDA RNG，是为了保证恢复语义完整，并避免后续实现增加随机行为后静默破坏合同。

## 5. 设备与数值精度

### 5.1 目标设备

本轮以本机单张 RTX 4060 Ti 8GB 为目标设备：

```text
device = cuda:0
distributed = false
num_workers = 0
```

模型规模不要求分布式训练。单卡可以避免通信、分片 checkpoint 和多 rank sampler 引入额外变量，使样本顺序、显存、暂停和恢复更容易核验。

运行 manifest 必须记录实际 GPU、驱动、CUDA、PyTorch 和依赖版本。若正式训练前改用其他 GPU，需要建立新的运行环境身份并重新做受控短跑，不能无痕替换目标设备。

### 5.2 混合精度边界

本轮采用 FP32 主状态与 BF16 autocast 计算：

| 对象 | 固定精度 |
| --- | --- |
| 模型参数 | FP32 |
| 参数梯度 | FP32 |
| AdamW 一阶、二阶状态 | FP32 |
| 主要前向矩阵计算 | BF16 autocast |
| RoPE 角度 | FP32 |
| RMSNorm 统计 | FP32 |
| Attention Softmax | FP32 |
| loss 与 token 归一化 | FP32 |
| GradScaler | 不使用 |

模型只执行 `model.to("cuda")`，不执行 `model.to(torch.bfloat16)`。参数保持 FP32，autocast 只让适合的算子在前向和反向计算中使用 BF16 路径；FP32 参数对应的 `.grad` 保持 FP32，AdamW 状态也由 FP32 参数创建为 FP32。

BF16 与 FP32 具有相同的指数位宽，数值范围比 FP16 更适合训练。当前 RTX 4060 Ti 和锁定 PyTorch 环境已经报告支持 BF16，因此不引入 FP16 loss scaling 路径。未使用 GradScaler 不代表训练不会出现异常，入口仍要拒绝非有限 loss 和梯度。

### 5.3 数值稳定与确定性

本轮继续使用：

```text
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CUBLAS_WORKSPACE_CONFIG = :4096:8
```

关闭 TF32 是本轮的数值边界，不代表 TF32 不可用于其他实验。若以后评估 TF32 的吞吐收益，应建立独立对照，不能在本轮训练中途切换。

当前实现已经把 RoPE 角度、RMSNorm 统计、Attention Softmax 和 loss 放在明确的 FP32 边界内。正式短跑还要观察 loss、梯度范数和显存，防止仅凭 dtype 配置推断训练一定稳定。

### 5.4 显存依据

63,912,192 个独立参数对应的 FP32 参数、梯度及 AdamW `m/v` 状态合计约 976 MiB。除此之外，实际显存还包括激活、Attention 中间量、logits、优化器临时张量、CUDA workspace 和 PyTorch reserved cache。

既有 8-step 工程短跑在长度 768、`micro_batch=2` 和 BF16 autocast 下测得 PyTorch 峰值 allocated 约 2.47 GiB。随后完成的受控基准固定每次更新消费 256 条相同记录，只改变物理 batch 与梯度累积。正式依据见 `../../pretrain_mini_microbatch_benchmark_v2/README.md`；第一版因未显式锁定 optimizer 后端而被取代。

### 5.5 Batch 组织

正式训练冻结为：

```text
micro_batch_size = 4
gradient_accumulation_steps = 64
effective_batch_records = 256
```

显式锁定 AdamW `foreach=false`、`fused=false` 后，`4 × 64` 达到 22.59 records/s 和 5,314.62 valid targets/s，中位更新耗时 11.22 秒，峰值 allocated / reserved 约为 3.73 / 3.79 GiB，最低设备空闲比例约为 38.31%。它比 `2 × 128` 略快，同时明显比 `8 × 32` 保留更多显存余量。`8 × 32` 只剩约 5.26% 设备空闲量且吞吐更低；`16 × 16` 触发 Windows 共享内存换页，被判定为不可用。

在 1,265,235 条训练记录上，该选择对应 4,942 个完整更新和一个包含 83 条记录的 tail update，总计 4,943 次 optimizer update。tail update 仍按自身全部有效 target 做全局 token 平均，不补齐、不重复样本，也不丢弃。

按正式候选配方短跑外推，数据准备及参数更新窗口合计约需 15.64 小时；加上训练运行中的验证与 checkpoint 后，当前端到端墙钟规划值约为 16.3 小时，实际速度仍受样本长度与系统负载影响。

## 6. 优化器

### 6.1 固定配置

本轮参考 MiniMind 官方预训练入口的 `AdamW(model.parameters(), lr=5e-4)`，把其依赖的 PyTorch 默认值展开并显式冻结：

| 项目 | 固定值 |
| --- | --- |
| optimizer | AdamW |
| 参数组 | 全部唯一可训练参数放在一个组中 |
| learning rate | `5e-4` |
| betas | `(0.9, 0.999)` |
| epsilon | `1e-8` |
| weight decay | `0.01` |
| foreach | `false` |
| fused | `false` |

官方入口没有建立 Norm、embedding 或 bias 的特殊无衰减组。本轮为了减少复现变量，也不额外引入参数分组规则。因此 weight decay 会应用于该组中的所有唯一可训练参数，包括 RMSNorm 权重和共享的 embedding / LM head 权重。

embedding 与 LM head 引用同一个 `Parameter`。`model.parameters()` 默认按参数对象去重，因此共享权重只会进入 optimizer 一次，不会被重复更新或维护两份 AdamW 状态。

### 6.2 实现边界

`foreach=false`、`fused=false` 不是 MiniMind 官方显式给出的超参数，而是本工程为当前单卡复现实验锁定的执行后端。这样可以避免 PyTorch 根据设备自动切换实现，并与已经通过的暂停恢复路径保持一致。将来研究 fused optimizer 的速度或显存收益时，应建立新的对照实验，不能在本轮中途切换。

本节只确定 optimizer 及其初始学习率。学习率是否 warmup、何时衰减以及最终降到多少属于 scheduler 配方，在下一节单独决定。

## 7. 学习率与梯度裁剪

### 7.1 固定配置

本轮继续参考 MiniMind 官方预训练配方，采用无 warmup 的余弦衰减和梯度范数裁剪：

| 项目 | 固定值 |
| --- | --- |
| 调度单位 | optimizer update |
| 总更新数 | 4,943 |
| warmup updates | 0 |
| 初始学习率 | `5e-4` |
| 调度方式 | cosine decay |
| 最终学习率 | `5e-5`，即初始值的 10% |
| 梯度裁剪 | global L2 norm，`max_grad_norm=1.0` |

官方实现按 micro-batch 位置计算余弦值，但只在梯度累积完成时更新参数。本工程把调度器直接绑定到 optimizer update：物理 batch 或梯度累积的拆分发生变化时，只要逻辑更新预算不变，学习率轨迹就不变。

### 7.2 余弦曲线

设本轮共有 `B=4,943` 次更新，第 `k` 次更新从 1 开始计数，则：

```text
progress = (k - 1) / (B - 1)
lr(k) = lr_min + 0.5 × (lr_max - lr_min) × [1 + cos(π × progress)]
```

三个锚点是：

| optimizer update | progress | 实际用于更新参数的学习率 |
| ---: | ---: | ---: |
| 1 | 0 | `5e-4` |
| 2,472 | 0.5 | `2.75e-4` |
| 4,943 | 1 | `5e-5` |

代码在 `optimizer.step()` 前记录 `lr_used`，随后执行 `scheduler.step()` 计算下一次更新使用的 `next_lr`。因此日志中的 `lr_used` 才是刚刚那次参数更新实际使用的学习率。

### 7.3 无 warmup 的边界

warmup 常用于降低训练初期参数、梯度和优化器状态尚未稳定时的大步更新风险。本轮没有断言 warmup 普遍无用，而是基于以下边界选择 0：

1. 复现主线优先参考官方无 warmup 配方；
2. 本轮有效 batch 同样为 256 条记录，初始学习率同为 `5e-4`；
3. loss 按完整逻辑 batch 的有效 target 总数归一化；
4. 每次更新执行 global norm 裁剪，并拒绝非有限 loss 和梯度；
5. 正式训练前仍要用相同配方短跑，观察早期 loss、梯度范数和裁剪频率。

如果短跑显示连续强裁剪、loss 非有限或明显不稳定，应停止并建立带 warmup 的新候选配方，而不是在已经启动的正式运行中临时改曲线。

### 7.4 裁剪口径

梯度累积完成并除以该逻辑 batch 的全部有效 target 数后，对所有有梯度的参数计算一个全局 L2 norm。`clip_grad_norm_` 返回裁剪前的 norm；若它大于 1.0，梯度会按统一比例缩小，再执行 AdamW 更新。日志必须保留这个裁剪前数值，才能统计训练是否频繁依赖裁剪。

## 8. 验证、保存与运行预算

### 8.1 完整验证

验证使用冻结的全部 5,000 条记录和 1,206,758 个有效 target，固定 `batch_size=4`、BF16 autocast 与 token 加权 NLL。受控测量中，随机初始化完整验证耗时 76.86 秒，峰值 allocated / reserved 约为 0.75 / 1.02 GiB。

验证时间点冻结为：

```text
step 0
step 500, 1000, ..., 4500
step 4943
```

总计 11 次，预计约增加 14.1 分钟。初始和最终验证不可省略；周期验证负责给出训练进度上的同分布 NLL 曲线，但不代替逐步 loss、梯度和显存健康检查。

### 8.2 Checkpoint

每 100 次 optimizer update 保存一次，并在最终 step 4,943 额外保存。单个完整训练状态约 0.72 GiB，一个 epoch 预计产生 50 个、占用约 36 GiB。它们全部保留在：

```text
E:\AI_research_checkpoints\minimind\pretrain_mini_one_epoch_v1
```

代码、冻结配置、run manifest 和 metrics 保留在 NVMe 的运行目录。checkpoint 使用同盘临时文件、flush、`fsync`、原子替换、SHA256 与 receipt；日志记录完整保存耗时。2026-10-25 正式候选配方短跑实测端到端保存为 24.62～24.89 秒，因此每次按约 24.7 秒预算，替代此前分项相加的 23.81 秒估计。

100-step 间隔把最坏重算量限制在约 18.9 分钟。机械盘剩余约 1.03 TiB，本轮不删除中间 checkpoint，避免在首次正式预训练中加入滚动清理风险。

### 8.3 时间预算

按目前测量：

| 组成 | 预计时间 |
| --- | ---: |
| 数据准备与 optimizer 参数更新窗口 | 约 15.64 小时 |
| 11 次完整验证 | 约 15.9 分钟 |
| 50 次 checkpoint 持久化、SHA256 与 receipt | 约 20.6 分钟 |
| 端到端训练运行墙钟预算 | 约 16.3 小时 |

以上由正式候选配方短跑校正，按一次不中断运行估计；额外恢复、运行波动和启动收尾仍会增加时间，因此 16.3 小时是规划值，不是完成承诺。

完整测量证据见 `../evidence/runtime_budget_v1/README.md`。

## 9. 观测与验收

### 9.1 主要结果

主要质量指标是完整冻结验证集上的 token 加权 NLL：

```text
validation_nll = 全部有效 target 的 loss_sum / 1,206,758
```

至少在随机初始化后、训练开始前测量一次，在完整一个 epoch 结束后再测量一次。方向性成功条件是最终 validation NLL 低于初始基线；是否增加数值阈值，要等训练配方和测量波动明确后再冻结。

### 9.2 过程证据

训练过程中至少记录：

- 已提交记录数和有效 target 数；
- optimizer update、epoch 进度和当前学习率；
- token 加权训练 loss；
- 梯度裁剪前范数以及是否发生裁剪；
- CUDA allocated、reserved、peak allocated 和设备空闲显存；
- 每次更新耗时、有效 targets/s 和 records/s；
- checkpoint、验证和主动暂停事件。

### 9.3 工程验收

正式运行还必须满足：

1. 无 NaN、Inf、OOM 和持续性显存泄漏；
2. 数据、模型初始化和训练配置身份与冻结合同一致；
3. checkpoint 包含模型、优化器、调度器、sampler、CPU/CUDA RNG 和进度元数据；
4. 暂停恢复后不重读已提交样本，也不跳过尚未提交样本；
5. 最终记录数和有效 target 数与一个 epoch 的预算一致；
6. 完整验证集使用固定数据和 token 加权口径。

### 9.4 停止与恢复合同

停止分为三类，不能只看 manifest 中最后出现的 step：

| 情形 | 持久状态 | 恢复依据 |
| --- | --- | --- |
| 主动暂停 | 在完整 optimizer update 和 sampler commit 后立即保存；checkpoint 与 receipt 成功后记为 `paused` | 本次暂停产生的 checkpoint |
| 可捕获异常 | manifest 记为 `failed` 并保存异常信息；异常前未落盘的更新不视为可恢复状态 | 最近一个通过检查的 checkpoint |
| 强制退出或断电 | 进程来不及收尾，manifest 可能仍为 `running`，也可能残留 `.tmp` | 结合进程、日志更新时间判断运行已停止，再选最近有效 checkpoint |

manifest 的 `global_step` 是运行观测，不自动等于持久恢复点。假设内存已经运行到 step 137，而最近完整 checkpoint 是 step 100，恢复必须从 step 100 开始；step 101～137 的状态已经随进程消失，需要重新计算。

恢复点必须同时满足：checkpoint 和 receipt 均存在；文件名、receipt 和 checkpoint 内部 step 一致；receipt 中的文件 SHA256 与实际字节一致；外部合同、receipt 和 checkpoint metadata 的合同身份一致；serialized state 包含模型、optimizer、scheduler、CPU/CUDA RNG、sampler 和进度元数据；sampler 提交位置与 receipt 一致。只按文件名选择最大 step 不满足该门禁。

每次启动使用新的 NVMe 运行目录，保留本次 manifest 和日志；恢复时继续引用原机械盘 checkpoint 根目录，并显式传入已经检查通过的 checkpoint。若更高 step 的同名产物已经存在，入口拒绝覆盖；应先检查它是否是有效恢复点，而不是删除后重写。

只读检查命令为：

```powershell
uv run python scripts/inspect_pretrain_recovery.py `
  --checkpoint-dir E:\AI_research_checkpoints\minimind\pretrain_mini_one_epoch_v1 `
  --contract <本次运行目录>\contract.json `
  --output <本次运行目录>\recovery-inspection.json
```

检查器报告缺少 receipt、合同不一致、哈希不一致、无法反序列化、孤立临时文件和无 checkpoint 的 receipt，并只从 `valid` 项中选择最新恢复点。检查器不删除、移动或修改任何训练产物。真正启动恢复时，训练入口还会再次检查 receipt、哈希和 metadata，并把状态加载到当前模型、optimizer、scheduler 与 sampler。

## 10. 结论边界

即使本轮顺利完成且 validation NLL 下降，也只能支持以下结论：

- 自己实现的预训练链路能够在冻结数据和预算下稳定工作；
- 模型比随机初始化时更适应这份 mini 数据的同分布文本预测；
- 训练和恢复证据足以分析本轮运行。

它不能单独证明模型事实知识可靠、具备对话能力、达到官方发布权重质量，或在外部分布上泛化良好。

## 11. 尚待逐项确定

2026-10-25，全部候选正式参数的暂停恢复短跑已经通过：连续至 step 4 与 step 2 暂停恢复至 step 4 的模型、optimizer、scheduler、sampler、RNG、关键日志及最终 checkpoint SHA256 全部一致。延长至 step 20 后，完整验证 NLL 为 7.0976711146，step 8～20 不再裁剪，数值及显存正常。完整证据见[短跑验收](../evidence/resume_gate_v1/README.md)。

当前只剩最终训练合同冻结：复核候选配置、代码身份和初始化身份，确定正式运行目录并生成不可混淆的冻结清单。正式一个 epoch 尚未启动。

## 12. 与既有工程短跑的关系

已有 `pretrain_mini_probe_v1` 只证明当前机器上的 BF16 完整更新、周期保存和暂停恢复链路可以工作。它使用 8 次 optimizer update 和固定 64 条验证子集，不是本轮正式配方。

短跑中的 `micro_batch=2`、`gradient_accumulation=4`、`lr=5e-4`、无 warmup 和每 4 步保存都只是当次工程检查设置。正式配方可以参考它的显存和耗时证据，但任何参数都必须重新解释和冻结，不能把短跑配置直接放大为一个 epoch。
