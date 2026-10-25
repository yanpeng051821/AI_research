# MiniMind mini 单轮预训练：冻结产物

> 本目录是正式实验的训练配方归档。数据配方已单独冻结，本目录引用它，并固定模型、更新规则、评测、保存恢复和验收。当前仅完成冻结与启动检查，完整一个 epoch 尚未启动。

## 1. 先看关系

“配方”是本次实验的一套规则，可以由多份各司其职的文件表达。把 JSON、合同和报告全部叫作“配置”，会掩盖它们的作用。

```text
数据配方：哪些文本 → 如何划分 → 如何编码 → 哪些 token 参与监督
    │
    └── 已冻结的数据合同、manifest、编码审计、tokenizer 身份
            │ 引用
            ▼
训练配方：什么模型与起点 → 如何更新 → 训练多久 → 如何验证与恢复
    ├── training_config.json      给程序执行的参数
    ├── TRAINING_CONTRACT.md      给人理解的实验约定与验收
    ├── contract.json             程序解析出的配置、代码及依赖身份
    └── training_freeze.json       把上述版本和数据、初始化、短跑证据绑定
            │ 启动核验通过后执行
            ▼
运行产物：manifest、metrics、result、checkpoint、receipt
```

短跑报告是采用候选配方的证据，本身不定义正式训练的全部规则。冻结配置之后，也不可能提前冻结未来 checkpoint 的内容；运行产物在训练中产生，再通过 hash 与合同关联。

## 2. 文件分工

建议先读本页，再读合同，最后对照 JSON 和冻结清单。

| 文件 | 回答的问题 | 本次内容 |
| --- | --- | --- |
| [TRAINING_CONTRACT.md](TRAINING_CONTRACT.md) | 这次实验做什么，怎样判定完成？ | 目标、数据引用、模型起点、更新规则、预算、评测、恢复和结论边界 |
| [training_config.json](training_config.json) | 程序实际接收哪些参数？ | 数据/tokenizer 路径与身份，模型结构，seed，batch，optimizer，scheduler，频率 |
| [contract.json](contract.json) | 参数展开后，实际采用什么训练实现？ | 默认值展开后的配置，4,943 步预算，核心 Python 文件 SHA256 和关键依赖版本 |
| [training_freeze.json](training_freeze.json) | 怎么确认启动时仍是这些版本？ | 上述文件哈希、数据合同与证据引用、初始化身份、Python/GPU/驱动环境和存储位置 |
| [preflight.json](preflight.json) | 当前文件实际检查过了吗？ | 冻结启动器的只读检查结果，没有执行训练 |
| [TRAINING_RECIPE_DRAFT.md](TRAINING_RECIPE_DRAFT.md) | 这些选择是怎样讨论出来的？ | 历史设计过程；正式执行以冻结产物为准 |

`training_config.json` 和 `contract.json` 不是两套要手工同步的配置：前者是输入，后者由程序读取、校验、展开后生成。`training_freeze.json` 是最后生成的清单，记录前面各文件的实际字节 SHA256，避免“内容变了但文件名没变”被忽略。

## 3. 数据引用

本轮数据配方在[数据合同目录](../pretrain_mini_preparation/split_seed43_v1/DATA_CONTRACT.md)，已经定义：

1. 固定来源和人工决定，按正文组隔离训练/验证，训练 1,265,235 条、验证 5,000 条。
2. 固定 tokenizer 文件与词表、BOS/EOS/PAD ID。
3. 长度 768，正文最多 766 token，右侧截断与 padding，causal shift 后忽略 PAD 监督。
4. 文件哈希、编码审计及已接受的数据质量边界。

训练 JSON 中仍能看到 tokenizer 和长度，因为程序需要这些输入。其依据来自被引用的数据合同，不代表训练配方又独立决定了第二套 tokenizer 规则。

## 4. 本次冻结内容

| 项目 | 采用版本 |
| --- | --- |
| 模型与起点 | MiniMind-3 Dense，63.91M 参数，seed=42 随机初始化，固定实际权重哈希 |
| 计算 | RTX 4060 Ti，FP32 状态 + BF16 autocast，确定性算法，TF32 关闭 |
| 更新 | micro-batch 4，累积 64，token 加权 loss，AdamW lr=5e-4，全局裁剪 1.0 |
| 计划 | 全数据一个 epoch，4,943 次更新，最后 83 条，无 warmup 余弦至 5e-5 |
| 验证 | 全 5,000 条，batch=4，step 0/每 500/最终，token 加权 NLL |
| 保存 | 每 100 步及最终保存，全量状态写 E 盘；日志写 D 盘 |
| 判定 | 完成原数据预算与可恢复交付；最终验证 NLL 低于本次初始 NLL |

核心运行合同继续使用短跑验证过的 `b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58`。新增冻结启动器单独由外层清单固定，它负责核对后委托原训练入口，没有改变参数更新算法。

## 5. 启动检查

工作目录为 `implementation/`。先执行只读核验：

```powershell
uv run python scripts/train_frozen_pretrain.py --freeze ../experiments/pretrain_mini_one_epoch_v1/training_freeze.json --check-only
```

检查依次覆盖冻结文件哈希、当前真实数据与 tokenizer、当前运行代码和依赖、设备环境、CPU 重建的初始权重。检查会构造模型和计算哈希，但不加载训练 batch、不执行 forward/backward 或 optimizer 更新；通过状态为 `frozen_recipe_verified_not_trained`。

冻结不会把文件变成操作系统级只读，也不是防恶意篡改的数字签名。它约束正常实验中的版本变更，Git 和文件哈希提供追溯。

## 6. 正式执行与运行产物

本次尚未执行以下长训练命令，列出供核对：

```powershell
uv run python scripts/train_frozen_pretrain.py --freeze ../experiments/pretrain_mini_one_epoch_v1/training_freeze.json --run --report runs/pretrain_mini_one_epoch_v1.frozen-launch.json
```

通过门禁后，新训练从随机初始化开始，写入：

```text
implementation/runs/pretrain_mini_one_epoch_v1/
    contract.json           本次启动的核心合同快照
    resolved_config.json    本次实际配置
    run_manifest.json       运行状态与已提交进度
    metrics.jsonl           更新、验证与保存事件
    result.json             正常结束后的汇总

E:\AI_research_checkpoints\minimind\pretrain_mini_one_epoch_v1\
    checkpoint-100.pt
    checkpoint-100.pt.receipt.json
    ...
    checkpoint-4943.pt
    checkpoint-4943.pt.receipt.json
```

外层 `.frozen-launch.json` 记录采用的冻结清单 SHA256；运行目录内 `contract.json` 记录原训练入口的核心合同。二者共同把“采用哪份人可读约定”和“实际执行什么代码”关联起来。

恢复使用相同冻结清单，显式传入已经检查通过的 checkpoint，并用 `--output-dir` 指定新的日志目录。只有输出日志位置的变化不改变更新轨迹；学习率、数据、预算或运行代码发生变化则会被门禁拒绝，应建立新实验版本。

## 7. 已有证据

恢复对照及早期稳定性见[正式候选短跑报告](../pretrain_mini_resume_gate_v1/README.md)。连续 4 步与 2+2 恢复完全一致；延长至 step 20 后验证 NLL 为 7.09767，显存与数值正常。端到端预算约 16.3 小时，实际波动仍需记录。

本次产物整理、冻结工具及门禁验证由助手按委托完成。正式长训练尚未启动；短跑通过不替代完整训练后的验收与结果分析。
