# AI Research

这个目录是模型训练与研究路线的长期资料库，与可运行的实验仓库
`D:/pythonlearning/small_model_post_training` 并行维护。

## 目录职责

```text
AI_research/
├── README.md
├── model_architecture_and_algorithms/  # 模型结构、训练目标、算法和论文映射
├── training_engineering/              # 训练系统、框架、显存、数据和评测工程
└── frontier_observation/              # 按案例沉淀前沿观察和路线决策
```

## 与实验仓库的边界

模型原理从[模型架构与算法](model_architecture_and_algorithms/README.md)进入；
其中[模型基础系列](model_architecture_and_algorithms/01_model_architecture/foundations/README.md)
连接预测任务、基础组件、整体信息流和反向传播，与各案例的实践教程互补。

- `AI_research` 保存研究问题、来源、证据、对比、假设和路线决定；从 2026-09-27 起，
  新训练工程的代码、测试与配置也统一在 `training_engineering/case_studies/<case>/` 下维护。
- `small_model_post_training` 继续保存既有实验代码与产物。本次目录约定不迁移历史工程；
  新案例的大数据、权重和大日志使用忽略目录或外部存储。
- 研究文档不自动变成实验任务。只有完成问题定义、可控变量、可信指标和成本评估后，才把一个案例升级为实验计划。
- 实验中的结果和故障，可以反向链接回这里，作为后续研究判断的证据。

## 前沿案例观察约定

每周从主线中划出约 20% 的时间，通常为 2 至 3 小时：

1. 选择一个主要案例，最多附带两个快速扫描案例。
2. 只使用论文、官方技术报告、官方模型卡、官方仓库或可追溯的实验材料作为关键判断依据。
3. 记录它解决的问题、修改的层次、baseline、证据强度、没有公开的部分，以及对当前路线的影响。
4. 最终给出 `忽略`、`记录`、`深读` 或 `候选实验` 标签。
5. 默认只记录，不因为新模型采用了不同架构或 recipe 就改变当前主线。

每个案例建立一个独立目录，目录名使用 `YYYY-MM-案例名`。`README.md` 负责状态、
问题和文档索引；具体沉淀按 `01_...md`、`02_...md` 逐步增加。同一周研究多个案例时，
在各自 README 中记录周次，不再把不同项目的架构、数据、证据边界和决策混入同一篇文档。

案例索引见 [`frontier_observation/README.md`](frontier_observation/README.md)。

## 当前主线

学习目标、阶段状态与下一步只在 [career_transition/README.md](../career_transition/README.md) 维护。
本研究库是资料与案例库，不复制目标路线；案例内的未来计划属于当时实验，不自动成为当前任务。

2026-09-27：本机尚未定位到上述 career_transition 目录。本轮明确开始
[MiniMind 独立实践](training_engineering/case_studies/minimind/README.md)，具体周安排与状态
统一维护在其 PRACTICE_ROADMAP.md；其他案例计划暂不作为本轮执行任务。

训练工程的体系化入口：

- [`training_engineering/README.md`](training_engineering/README.md)
- [`MiniMind 实践工程`](training_engineering/case_studies/minimind/README.md)
- [`Qwen3-0.6B OpenR1-Math SFT 工程案例`](training_engineering/case_studies/qwen3-0.6b-openr1-math-sft/README.md)
- [`小模型工具调用后训练案例`](training_engineering/case_studies/small-model-tool-use-post-training/README.md)
