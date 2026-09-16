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

练习：从现有配置测试选一个尚未覆盖的边界，先写预期行为和理由，再写最小测试。
如果已被现有参数化测试覆盖，就解释该测试，不重复添加。此次不直接改正式配置或 hash 规则。

## 验收与学习记录

- [ ] 能画出正式与独立实现两条入口，指出它们没有串行调用。
- [ ] 能解释配置身份、数据身份、源码身份分别防什么问题。
- [ ] 能说明 dry-run 做到哪里、不能证明什么。
- [ ] 能找到一次早期失败和一次训练期失败各自的落盘边界。
- [ ] 完成一次测试阅读或小练习。

学习日期、独立完成部分、查询或提示、测试结果、仍不理解的问题：待本主题结束后填写。

