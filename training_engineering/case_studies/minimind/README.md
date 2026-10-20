# MiniMind 实践工程

以 MiniMind 的开源代码和第一轮理论笔记为参照，由学习者从 0 到 1 实现并运行小模型训练工程。
核心算法、各阶段数据、训练与恢复、生成、评测和工具交互共同纳入实践。

## 路线

每周安排、当前状态、数据合同、设备预算与验收统一见 [实践路线](PRACTICE_ROADMAP.md)。
先完成自己的 Pretrain/SFT 工程，再复用它进入 LoRA、蒸馏、DPO、在线 RL 和 Agent 实验。

## 目录

本案例的文档与实现统一在这里归档。每周实践教程的交付标准是：读者不依赖原对话，也能按顺序实现、运行、检查和排错；当前任务统一维护在实践路线中。

完成步骤必需的理解性说明放在正文：先解释模块解决什么问题、怎样与上下游配合，
再连接公式、实现与验证，并注明直觉的适用边界；对话中的补充问答统一归入文末 QA，不只保留代码和报错结论。
实验与测试也说明验证目的、条件选择、可发现的错误和结论边界，形成“作用 -> 计算 -> 验证 -> 解释”的完整关系。

### 教程编写约定

从 2026-10-19 起，第一至第六周实践教程采用同一套组织规则。前五周已按第六周样板重组，保留推导、手算、实现与验证理由，学习 QA 统一放在每篇末尾。历史测试数仅作阶段记录，当前复练使用锁定环境与现有接口。

1. **开篇给全貌**：本周目标、上周输入、本周产物、适用环境、完成标准，以及操作顺序表。不以测试数量代替学习目标。
2. **正文按依赖推进**：每一步说明为什么现在做、在哪个文件做、输入输出、具体操作或关键代码、运行命令、预期结果和排错方向。步骤之间说明前一项产物怎样进入下一项。
3. **任务与说明分开**：必做操作写清楚；历史排错、可选扩展和未覆盖边界另列。用于完成本步的公式、形状和调用关系留在正文，不能全部挪到附录。
4. **提供可执行闭环**：命令注明工作目录与 shell；代码片段注明新增还是替换、插入位置和导入。没有数据时提供小输入，已有脚本演进到后续功能时明确提示。完整源码是核对答案，不代替操作说明。
5. **统一收尾**：正文最后给出验收、实际运行证据、限制与下一步；学习 QA 全部放在文档末尾，按问题组织，可从正文链接过去。QA 后不再追加新任务。
6. **区分教程与记录**：正文面向读者讲解，不堆叠“随后完成、当时通过、下一步待实现”。历史数值保留日期；未执行的复练不得写成已验证结果，协助范围如实记录。

数字标题采用 1、1.1、1.2 的层级，标题概括内容而不是口语问句。QA 的小标题可以直接使用问题。长篇按阶段拆成步骤，不为形式统一机械复制空小节。

```text
minimind/
  README.md
  PRACTICE_ROADMAP.md         # 唯一实践路线与状态
  theory_notes/               # 第一轮理论笔记与历史实验记录
  practice_notes/             # 本轮逐周实践教程与运行记录
  implementation/            # 代码、测试、配置与可执行入口
  experiments/               # 计划：小型报告、数据卡、结果摘要
```

原理论笔记已经迁入 [第一轮理论笔记](theory_notes/README.md)，其中也保留了当时的历史实验记录与
Agent 实践待办。历史环境与完成标记不直接当成本轮运行证据。模型权重、完整数据和大日志存放在忽略目录
或外部存储，Git 保存脚本、配置、数据版本、产物 hash 和必要的小型证据。

## 环境

进入 `implementation/` 后，使用项目独立的 Python 3.12 环境：

```shell
uv sync --locked --python 3.12
uv run python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run pytest -q
```

`pyproject.toml` 为 Windows 指定官方 PyTorch CUDA 13.0 软件源，macOS 使用 PyPI 的原生版本；
两边固定 PyTorch 2.14.0，其他依赖由 `uv.lock` 管理。Linux 当前仍使用 PyPI 来源，服务器的 CUDA
版本需要在部署时根据驱动检查。虚拟环境在各台机器上分别创建，不能复制 Mac 的 `.venv` 到 Windows。

安装 CUDA 版 PyTorch 后，模型和张量仍默认创建在 CPU。实际 GPU 计算需要把模型和输入放在同一设备，
例如 `model.to("cuda")` 和 `input_ids.to("cuda")`。目前基础组件测试仍以 CPU 为主，便于跨平台复练；
训练阶段再统一配置设备。`nvidia-smi` 显示的 CUDA 版本代表驱动支持范围，`torch.version.cuda`
显示 PyTorch 构建使用的 CUDA 版本，两者不要求数字完全相同。

Windows 如果遇到 Scoop 的 `uv` 启动器找不到程序，可执行 `scoop reset uv` 修复链接；
如果 uv 缓存和项目位于不同盘而出现 hardlink 警告，可用 `uv sync --locked --link-mode copy`。

## 样本恢复

`data/source_samples/*.jsonl` 和 `data/audit/` 不进入 Git，因此换机器后只有来源清单，没有样本和
`accepted.jsonl` 是可能出现的情况。Windows 在 `implementation/` 中执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_source_samples.ps1 -DatasetType pretrain
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_source_samples.ps1 -DatasetType sft
uv run --locked minimind-audit data/source_samples/pretrain_sample.jsonl --dataset-type pretrain --output-dir data/audit/pretrain
uv run --locked minimind-audit data/source_samples/sft_sample.jsonl --dataset-type sft --output-dir data/audit/sft
uv run --locked python scripts/inspect_pretrain_batch.py
```

恢复脚本按 `manifest.json` 中的固定 revision 下载有限字节，提取前 20 条完整记录，并在写入前校验
原始字节的 SHA256；已有文件不一致时拒绝覆盖。恢复原样本不等于重新选择训练数据。
如果需要代理，可在当前终端设置 `HTTP_PROXY` 和 `HTTPS_PROXY` 为本机实际代理地址，再执行下载。

2026-10-11 在 Windows 上完成恢复：两份样本 hash 与原清单一致，Pretrain/SFT 均为 20 条通过、
0 条拒绝。真实 Pretrain batch 为 `[2, 512]`，shift 后监督 token 共 490 个，logits 为
`[2, 512, 6400]`，loss 有限且共享权重获得梯度。这次验证没有执行 optimizer 更新，也不是正式训练。

## 协作

实践教程：

- [第 1 周：数据入口与审计](practice_notes/week_01_data_audit.md)。
- [第 2 周：Tokenizer 与训练数据](practice_notes/week_02_tokenizer_dataset.md)，已完成 BPE、官方 tokenizer、Pretrain Dataset、collator 和真实 batch 验证。
- [第 3 周：模型基础组件](practice_notes/week_03_model_components.md)，代码验收完成，包含组件、配置、初始化、参数统计与组合 smoke 的实践过程，全项目101项测试通过。
- [第 4 周：Attention 与完整 Decoder](practice_notes/week_04_attention_and_decoder.md)，验收完成：模型、单步训练、greedy 生成与 tiny 序列过拟合通过，全项目 221 项测试通过。
- [第 5 周：单设备训练系统](practice_notes/week_05_training_system.md)，验收完成：整批与累积更新对齐，整轮循环和尾部更新通过；训练测试 8 项，全项目 229 项通过。下一步进入第 6 周验证与保存恢复。
- [第 6 周：验证与训练恢复](practice_notes/week_06_validation_and_checkpoint.md)，2026-10-18 教学工程验收完成：验证、随机状态、样本位置、更新边界恢复和最小入口接通，新增回归 19 项、全项目 249 项通过。下一步进入第七周真实语料子集与实验合同准备；周期保存和数据身份门禁尚未接入，不能视为生产级训练入口。

学习者亲手完成环境、工程结构、算法、数据处理、测试、运行、分析与归档。
助手负责讲解、提出验收问题、审查实现和协助定位错误；默认不代写核心实现或代跑实验。
允许查阅 API 和原笔记，遇到具体问题再逐步给提示。
