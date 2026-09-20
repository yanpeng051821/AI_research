# SFT 工程代码主题回顾

## 定位与当前状态

本目录用于把已经完成的真实 S1 工程转化为学习者能够解释、修改和验证的能力。它是案例文档的代码阅读伴册；上一级 01–11 负责说明实验过程、实际结果与经验。

七个主题的学习文档已准备完成；主题 01 至 04 已通过讲解、代码追踪、独立回答与补测验收。
阅读文档、测试已存在和本人独立掌握是三件不同的事。后续仍按一次一个主题推进，根据
练习表现调整深度，不规定必须在几次课内结束。

已完成主题同时保留简短“快速问答”，用于间隔复习和发现遗忘点。它们只覆盖关键合同，
不能替代重新沿代码调用链定位实现和运行测试。

## 阅读顺序

| 主题 | 文档 | 完成时应能做到 | 学习状态 |
| --- | --- | --- | --- |
| 01 | [入口、配置与实验身份](01_ENTRY_CONFIG_AND_IDENTITY.md) | 从命令追踪到组件装配，区分两条训练入口 | 已完成（2026-09-17） |
| 02 | [真实样本与数据交付](02_SAMPLE_AND_DATA_FLOW.md) | 跟踪样本、顺序、padding 和监督位置 | 已完成（2026-09-18） |
| 03 | [SFT 计算与参数更新](03_LOSS_AND_OPTIMIZER_STEP.md) | 解释并验证 shift、归一化和更新时机 | 已完成（2026-09-19） |
| 04 | [TRL 接入与框架边界](04_TRL_AND_FRAMEWORK_BOUNDARIES.md) | 解释自定义 Trainer 的每一项必要修改 | 已完成（2026-09-21） |
| 05 | [恢复、失败与显存](05_RESUME_FAILURE_AND_MEMORY.md) | 判断能否恢复及如何定位资源问题 | 未开始 |
| 06 | [评测与证据闭环](06_EVALUATION_AND_EVIDENCE.md) | 复核指标口径、样本配对和结论边界 | 未开始 |
| 07 | [测试策略与 pytest](07_TESTING_STRATEGY_AND_PYTEST.md) | 能独立设计、运行和诊断本项目的分层测试 | 未开始 |

## 首先分清两条实际路径

正式 S1：

```text
scripts/train_sft_trl.py::main
-> ExperimentConfig / 冻结 artifact / 固定顺序 / run identity
-> build_trl_sft_args
-> model + TRL collator + FrozenOrderSFTTrainer
-> trainer.train
-> 原生 Trainer checkpoint / final_model / run_manifest
```

独立实现：

```text
sft-train 或 scripts/train_sft.py
-> cli.main
-> runner.run_experiment
-> runner._build_dataloaders
-> engine.train / run_optimizer_step
-> 自定义 checkpoint / sampler commit / metrics
```

正式入口没有调用 runner.run_experiment 或 engine.train。两条路径共享部分配置和数据工具，
但 sampler、恢复状态和训练循环不同。之前对话把它们说成一条串行调用链，这里以实际代码纠正。

正式 TRL 封装针对单卡、单 epoch、冻结数据顺序、FP32 参数与可选 BF16 autocast。
扩展到多卡、多 epoch、packing 或其他算法，需要重新检查恢复、步数、loss 和测试合同。

## 每个主题怎么学

1. 先讲一个实际工程问题和本次为什么需要处理它。
2. 从入口沿一条具体样本或状态走完调用链。
3. 学习者说明输入、输出、关键不变量与失败位置。
4. 阅读一个现有测试，预测结果后再运行。
5. 学习者完成一个小修改或一个新的边界测试，由助手审查。
6. 在本篇底部记录结果，确认理解后再进入下一主题。

查 API 和框架源码允许。核心计算要求能借助文档独立实现；编排代码要求能定位和修改；
第三方框架内部先掌握当前接口与相关行为。所有练习使用临时目录、小数据或 tiny model，
不覆盖已冻结训练产物，不默认下载模型或启动付费 GPU。

## 代码与验证范围

代码根目录：
[D:/pythonlearning/small_model_post_training/independent_implementation](D:/pythonlearning/small_model_post_training/independent_implementation)

文档按当前本地代码编写。各篇测试命令均在该根目录运行，属于后续教学操作，本次编写文档
不代表已经执行这些测试。框架行为以项目锁定版本和对齐测试为准。方法与模块边界可以复用，
具体可复用比例和学习耗时不做保证。
