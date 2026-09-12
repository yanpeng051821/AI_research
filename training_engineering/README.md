# Training Engineering

这个目录沉淀“如何把一次模型训练做成可信实验”的工程方法。它不保存模型权重、
大数据集和原始运行日志；这些可执行产物仍由对应实验仓库保存。

## 分类方式

```text
training_engineering/
├── README.md
├── playbooks/       # 跨模型、跨算法可复用的方法，后续从案例中提炼
└── case_studies/    # 一次具体训练工程的完整过程和证据边界
```

案例目录使用以下命名方式：

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

## 当前案例

- [`qwen3-0.6b-openr1-math-sft`](case_studies/qwen3-0.6b-openr1-math-sft/README.md)：
  使用 Qwen3-0.6B-Base 和 OpenR1-Math 数据建立数学 SFT 的数据、训练、恢复、
  评测与证据闭环。当前正式 S1 尚未启动。

## 文档与实验仓库的边界

- 本目录回答“为什么这样做、完整流程是什么、下次如何独立复用、失败如何判断”。
- `D:/pythonlearning/small_model_post_training` 保存真实代码、测试、配置、manifest
  和运行证据。
- 文档中的数值结论必须能追溯到实验仓库中的文件；无法追溯的历史数字只可作为
  线索，不作为 baseline。
- 案例完成后，再把稳定的共性提炼到 `playbooks/`。不要在第一个案例中急着把
  所有做法写成普遍规律。
