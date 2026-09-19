# 01 入口、配置与实验身份

## 本次要解决的问题

当一条训练命令启动后，谁读取配置、谁检查数据、谁加载模型、谁写运行记录？
如果更换数据文件却沿用同一个输出目录，程序如何防止把两个实验混成一个？

对应案例文档：上一级 02、04、09。先完成调用地图，再深入训练计算。

## 代码阅读顺序

| 顺序 | 文件与入口 | 阅读目的 |
| --- | --- | --- |
| 1 | [正式脚本](D:/pythonlearning/small_model_post_training/independent_implementation/scripts/train_sft_trl.py)，main | 从 argparse 到 manifest 再到 trainer.train |
| 2 | [配置](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/config.py)，ExperimentConfig | YAML 解析、校验、路径和 semantic_hash |
| 3 | [TRL 参数](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/trl_training.py)，build_trl_sft_args | 项目参数如何转为 SFTConfig |
| 4 | [源码身份](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/provenance.py) | 文件 hash 与 runtime tree |
| 5 | [运行工具](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/experiment.py) | 原子 JSON 写入、环境和时间 |
| 6 | [独立入口](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/cli.py) 与 [装配层](D:/pythonlearning/small_model_post_training/independent_implementation/src/post_training_core/runner.py) | 对比独立实现，避免混淆 |

## 沿一次启动走完整链路

正式脚本先解析 config、output-dir、resume-from-checkpoint、stop-after-steps 和 dry-run，
再读取 ExperimentConfig。接着扫描 tokenized artifact，固定索引顺序，构造 TRL 参数。

这些信息组合为 identity：配置 hash、源码树 hash、实验合同 hash、训练和验证文件 hash、
样本顺序 hash、记录数、总更新数和 warmup。恢复时比较旧 identity，并检查 checkpoint
是否属于本 run；新运行要求输出目录为空。

校验通过后写 sample_order.jsonl、run_manifest.json 和本次 attempt 文件。dry-run 在此返回；
正常路径继续加载 tokenizer/model、collator 和 Trainer，执行训练，再写 completed、
paused 或 failed。

注意 dry-run 仍构造 SFTConfig，当前脚本也会检查 world_size/n_gpu。不能仅凭名称推断它
是完全无设备依赖的纯配置解析。模型加载发生在 dry-run 返回之后。

独立入口则通过 runner 装配 DataLoader 和 engine，并使用自己的运行初始化及状态更新。
两者异常保护范围也不同：正式脚本部分配置和 identity 检查位于 try 之前，此类早期失败
不保证产生 failed manifest。阅读时应实际标出 try/finally 的边界。

## 用具体变化理解身份

想象已有一次训练，保持文件名不变但换了一条训练样本。train_sha256 会改变，即使记录数
不变也应拒绝恢复。另一个例子是只把暂停点改小：暂停边界与完整训练计划应区分，否则
scheduler 的总步数会随着暂停行为改变。

本案例 61,224 条、每批 1 条、累积 128 次，一轮共 479 次更新，warmup 为 15。
这些数值用于理解历史合同；新实验必须按新 artifact 重算。

## 测试与动手

先读 [test_config.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_config.py) 的路径解析、未知字段与 semantic_hash 测试，
再读 [test_trl_training.py](D:/pythonlearning/small_model_post_training/independent_implementation/tests/test_trl_training.py) 中
test_formal_cli_runs_pauses_and_resumes_offline，查看它如何在离线 tiny 模型上走入口。

```powershell
uv run python -m pytest tests/test_config.py -q
```

### 本次发现的真实缺陷

正式入口原先存在两个输出目录来源：YAML 中的 `config.output_dir` 参与
`semantic_hash`，CLI 的 `--output-dir` 则决定 manifest、checkpoint 和模型的实际
写入位置。两者不一致时，程序仍会启动，从而出现“合同声明目录 A、产物写入目录 B”。

本次先在 `test_trl_training.py` 增加失败测试，要求目录不一致时：

1. CLI 返回非零退出码；
2. stderr 包含两个冲突目录；
3. 实际输出目录中不产生 `run_manifest.json`。

初次执行时测试在 `assert result.returncode != 0` 处失败，证明缺陷真实存在。随后在
`ExperimentConfig.from_yaml` 之后、扫描数据和创建运行记录之前比较两个目录的
`resolve()` 结果，不一致立即抛出 `ValueError`。原有暂停/恢复成功路径也同步为同一个
输出目录，防止新门禁误伤合法运行。

这个练习形成了完整的 RED-GREEN-REGRESSION 链路，而不是先改代码再补一个必然通过
的测试。最终定向测试、整个 `test_trl_training.py`、Ruff 和 `git diff --check` 均通过。

### 运行身份的边界

- `config_hash`：证明关键训练语义一致，不等同于 YAML 文件字节 hash；
- 训练/验证文件 hash：证明实际数据文件内容一致；
- `sample_order_sha256`：证明样本交付顺序一致，不能替代数据内容 hash；
- runtime tree hash：证明运行时源码身份；
- `resume_from_checkpoint`：描述执行位置，独立记录和校验，不纳入训练语义 hash；
- 内部 manifest：记录程序理解的运行语义及 `running/paused/completed/failed` 状态；
- 外层 launcher：补充保存命令、PID、stdout、stderr 和 exit code，并覆盖 manifest 创建前
  失败、SIGKILL 或宿主机中断等内部异常处理无法覆盖的边界。

当前配置中的 `model_name_or_path` 和固定 `model_revision` 会进入 `semantic_hash`，因此能
约束“应加载哪个模型版本”。但正式入口没有另外计算模型权重文件和 tokenizer 文件的
内容 hash。固定到不可变 commit revision 时已经具备较强的可追溯性；若使用可移动分支、
本地目录或需要字节级审计，则还应记录 resolved revision 或实际文件清单及 hash。不能把
“模型引用进入配置 hash”表述成“模型和 tokenizer 实体已经做了内容 hash”。

## 快速问答

**Q：数据审计和 `train_sft_trl.py --dry-run` 是同一件事吗？** 不是。审计脚本从原始数据
构造并划分冻结 artifact；dry-run 消费已经生成的 artifact，检查配置、身份、顺序和正式
入口能否形成合法训练计划。

**Q：`config_hash` 证明了什么？** 它证明参与 `semantic_hash` 的训练语义字段一致，不证明
YAML 字节完全相同，也不单独证明远端模型文件的字节内容一致。

**Q：为什么 `seed` 属于训练语义，而 `resume_from_checkpoint` 不属于？** `seed` 会影响冻结
样本顺序和优化轨迹；checkpoint 路径描述的是同一训练计划从哪个执行位置继续。

**Q：为什么程序内已经有 `try/except/finally`，仍需要 launcher？** 因为部分启动门禁发生在
`try` 之前，进程还可能遭遇 SIGKILL、宿主机终止等无法由 Python 异常处理落盘的情况。

**Q：YAML 与 CLI 的输出目录为什么必须一致？** 一个参与合同身份，另一个决定真实写入位置；
不一致会造成“合同指向 A、产物写到 B”，使恢复和审计失去可信依据。

## 验收与学习记录

- [x] 能画出正式与独立实现两条入口，指出它们没有串行调用。
- [x] 能解释配置身份、数据身份、源码身份分别防什么问题。
- [x] 能说明 dry-run 做到哪里、不能证明什么。
- [x] 能找到一次早期失败和一次训练期失败各自的落盘边界。
- [x] 完成一次测试驱动的小练习，并对原成功路径执行回归测试。

学习日期：2026-09-17。

独立完成部分：阅读入口与配置实现；回答 `seed`、`resume_from_checkpoint`、配置 hash、
样本顺序以及异常状态问题；编写输出目录冲突的集成测试；根据失败信息修复测试代码并
完成生产入口门禁。

查询或提示：在调用地图、配置语义边界、测试结构和错误定位上接受了引导；具体测试与
门禁代码由学习者完成。测试过程中经历了断言失败和测试收集期 `SyntaxError`，并能够
区分“生产行为不符合合同”与“测试文件无法被 Python 导入”。

测试结果：目录冲突测试通过；暂停/恢复成功路径通过；`test_trl_training.py` 共 7 项
通过；Ruff 与补丁格式检查通过。

后续改进项：正式实验可增加统一外层 launcher，以便完整记录 manifest 创建前的失败和
操作系统级非正常退出；共享配置及历史 runbook 中的输出目录写法应在下一次正式运行前
按新合同统一。
