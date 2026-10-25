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
其中[模型基础系列](model_architecture_and_algorithms/01_model_architecture/README.md)
连接预测任务、基础组件、整体信息流和反向传播，与各案例的实践教程互补。

- `AI_research` 保存研究问题、来源、证据、对比、假设和路线决定；从 2026-09-27 起，
  新训练工程的代码、测试与配置也统一在 `training_engineering/case_studies/<case>/` 下维护。
- `small_model_post_training` 继续保存既有实验代码与产物。本次目录约定不迁移历史工程；
  新案例的大数据、权重和大日志使用忽略目录或外部存储。
- 研究文档不自动变成实验任务。只有完成问题定义、可控变量、可信指标和成本评估后，才把一个案例升级为实验计划。
- 实验中的结果和故障，可以反向链接回这里，作为后续研究判断的证据。

## 模型发布约定

2026-10-25 起，每个完成训练并通过工程验收的正式阶段，都将模型及完整阶段说明、配方和结果发布到我们的 Hugging Face。该约定适用于 Pretrain、SFT、LoRA、蒸馏、DPO、RL 等后续实验；smoke、恢复短跑和中间 checkpoint 作为证据，不默认单独建立模型仓库。

### 发布内容

| 内容 | 必须说明或提供的项目 |
| --- | --- |
| 模型产物 | 全量权重或 adapter、模型配置、适用的 tokenizer/模板及版本、实际可运行的加载与推理方式 |
| 阶段履历 | 当前阶段、输入模型与精确 revision/hash、此前经历的训练阶段、本次输出身份；区分从头训练、新阶段和断点恢复 |
| 数据配方 | 来源与 revision、许可、筛选及切分方法、数据量、文件 hash、tokenizer、模板、截断、padding 和监督规则；链接准备代码 |
| 训练配方 | 最终配置、合同与必要冻结清单、精度、参数更新范围、batch/累积、优化器、学习率、预算、seed、启动和恢复方法 |
| 代码与环境 | 可公开的代码入口、固定 Git commit、依赖锁及必要运行环境，注明与上游参考实现的差异 |
| 结果与成本 | 固定评测口径、基线与训练后结果、训练曲线或摘要、硬件与实际耗时、显存、主要故障与恢复记录 |
| 模型卡 | 目标、使用方式、已验证范围、未完成评测、已知问题、许可和用途限制；区分学习者与助手的实际贡献 |

正式发布不要求实验指标一定提升。若目标未达成或出现能力回退，标明 research checkpoint、实际结果和限制，不把工程完成包装成能力提升。完整配方表示公开本次实际采用的规则与身份，不暗示已经逐位复现上游模型。

### 发布流程

1. **验收**：检查训练预算、最终 checkpoint、评测与合同一致，整理已完成和未完成项。
2. **整理**：导出模型产物，汇总阶段履历、配方与证据；HF 模型卡链接固定版本的 GitHub 实现，GitHub 案例反向记录 HF 地址与 revision。
3. **检查**：核对权重、数据、代码的许可与再分发条件，移除凭据、私人信息、内部地址和不必要的本机路径。不能公开的材料只给出允许披露的来源、身份或重建说明，并明确可复现性限制。
4. **上传**：用独立仓库或明确版本区分各阶段，保留历史版本及父模型身份，不静默覆盖原有发布。
5. **验证**：从 HF 指定 revision 下载，核对发布文件 hash，并按模型卡执行最小加载/推理检查；成功后记录仓库地址、revision、文件清单和检查结果。未完成上传或验证时保持待发布状态。

原始数据不因配方公开而自动重新上传；完整 optimizer、scheduler、RNG 等恢复 checkpoint 默认保留在本地或外部备份，不作为 HF 推理模型的必传附件。发布前确认两类产物各自的位置与保存情况，不能把 HF 推理权重当成完整恢复备份。

原冻结合同和原始证据保持不变。需要去除路径等信息时，另生成公开版本，记录与原始产物的对应关系及原 hash，不修改已冻结文件来迁就发布。上传模型卡前还要检查外链是否实际可访问；私有仓库链接不能冒充公开训练代码。

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
