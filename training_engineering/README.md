# Training Engineering

这个目录沉淀“如何把一次模型训练做成可信实验”的工程方法。从 2026-09-27 起，
新训练工程的文档、代码、测试和配置统一放在对应 case_studies 案例下。
模型权重、大数据集和大日志放忽略目录或外部存储；既有工程保留原代码位置。

## 分类方式

```text
training_engineering/
├── README.md
├── playbooks/       # 跨模型、跨算法可复用的方法，后续从案例中提炼
└── case_studies/    # 一次具体训练工程的完整过程和证据边界
```

模型、数据和训练阶段已经冻结时，案例目录使用以下命名方式：

```text
<model>-<data-or-domain>-<training-stage>/
```

这个名字同时说明模型、数据/领域和训练阶段，避免把一个具体案例误认为普遍 recipe。
例如：

```text
qwen3-0.6b-openr1-math-sft/
gemma-4b-domain-sft/
qwen-small-code-midtrain/
```

若案例仍处于 Gate 0、模型尚未冻结，可以暂时使用稳定的任务名；模型与实验合同冻结后，
由案例 README 记录具体版本，不为了目录整齐提前假定技术选择。

## 当前案例

- [`minimind`](case_studies/minimind/README.md)：当前实践主线。以原理论笔记和 MiniMind
  源码为参照，由学习者独立实现算法、数据与训练工程，并亲自完成各阶段实验。
- [`qwen3-0.6b-openr1-math-sft`](case_studies/qwen3-0.6b-openr1-math-sft/README.md)：
  使用 Qwen3-0.6B-Base 和 OpenR1-Math 数据建立数学 SFT 的数据、训练、恢复、
  评测与证据闭环。正式 S1、配对评测与结果分析已有记录，完整 MATH-500 延期。
- [`minglan-roleplay-sft`](case_studies/minglan-roleplay-sft/README.md)：
  第二个训练工程，研究剧版盛明兰的中文日常对话与多轮一致性。公开剧版语料未通过
  Gate 0 数据可行性门禁，当前暂停，未进入基座评测或训练。
- [`small-model-tool-use-post-training`](case_studies/small-model-tool-use-post-training/README.md)：
  工具调用候选工程，已有 Gate 0 审计材料，尚未冻结模型或训练合同。当前实践优先级
  转向 MiniMind，该案例的新增训练暂缓。

## 文档与实验仓库的边界

- 本目录回答“为什么这样做、完整流程是什么、下次如何独立复用、失败如何判断”。
- 新案例在自己的 `implementation/` 内保存代码、测试、配置和运行入口，逐步建立。
- 既有 `small_model_post_training` 实验代码与证据继续保存在原仓库，不随本次规划迁移。
- 文档中的数值结论必须能追溯到实验仓库中的文件；无法追溯的历史数字只可作为
  线索，不作为 baseline。
- 案例完成后，再把稳定的共性提炼到 `playbooks/`。不要在第一个案例中急着把
  所有做法写成普遍规律。
