# S1固定FP32评测

## 1. 执行结果

2026-11-01北京时间05:03–05:05完成正式S1评测。使用最终49,087步模型，不挑中间checkpoint。36题全部生成、全部2,048条验证计算完成；结果和原始输出保持独立，不覆盖B0。

| 项目 | 结果 |
| --- | --- |
| NLL | 1.7210819774489265，FP32，833,052个有效目标 |
| 核心/工具 | 32/4条生成齐全，任务初评见配对分析 |
| 停止 | EOS 17条，长度上限19条 |
| 生成 | 共5,587个新token；预算仍为256，不因更早结束补采样 |
| 模型 | 原生FP32、63,912,192参数、共享embedding/lm_head |
| 只读性 | 评测前后完整参数摘要不变，输入及代码身份复查通过 |
| 输入对照 | 36题实际prompt文本与input_ids逐条和B0相同 |

没有主动展开think、修改system、增加重试或更换解码参数。工具题只看文本/给定结果，不执行真实服务。不对教学小模型的这些输出作医学、法律或现实工具执行承诺。

## 2. 入口与来源绑定

原[evaluate_sft.py](../../../../implementation/scripts/evaluate_sft.py)的加载函数实际可读取相同原生格式，不存在必须改写的B0权重硬编码；此前文档对此判断过严。新增[evaluate_sft_pair_member.py](../../../../implementation/scripts/evaluate_sft_pair_member.py)只作为受约束的S1适配层：

1. 核对原评测冻结的所有文件及依赖版本，不改原冻结标记。
2. 确认已批准V2补丁与训练冻结一致；确认S1来自已核验的最终正式checkpoint。
3. 要求S1配置与B0逐项一致，检查模型包及独立导出验证身份。
4. 直接调用原evaluate_sft.run，复用严格加载、generate_case、greedy_generate、SFTDataset/collator和evaluate_nll。
5. 执行前后核对适配层与来源身份，写evaluation_member.json。

原引擎输出字段名pretrain_source在S1报告里保留，但字段实际stage为supervised_fine_tuning；[evaluation_member.json](evaluation_member.json)补充明确阶段，未修改冻结引擎。该引擎的training_started=false表示评测不训练，不能读成S1没训练过。

## 3. 文件地图

| 文件 | 用途 |
| --- | --- |
| [generations.jsonl](generations.jsonl) | 每题完整prompt、输入/输出token、原始/展示文本、耗时及停止原因 |
| [validation.json](validation.json) | 全验证FP32 NLL与目标数 |
| [result.json](result.json)、[run_manifest.json](run_manifest.json) | 运行结果与输入/代码身份 |
| [evaluation_member.json](evaluation_member.json) | S1来源、冻结评测及评分V2补丁的绑定 |
| [verification.json](verification.json) | 完整覆盖、B0/S1实际输入一致与结果文件摘要 |
| [judgments.assistant.v1.json](judgments.assistant.v1.json) | 助手阅读36份完整输出后给出的逐检查点意见与引句 |
| [配对分析](../paired_analysis_v1/README.md) | 来源绑定后的完整评分、分类与翻转，不是独立人工复核 |

原始generation行里的pending_manual_review保留生成时状态，不回写原始输出；后续评分独立存放。verification证明文件和覆盖范围，不自动证明语义评分正确。

## 4. 实际命令

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/evaluate_sft_pair_member.py `
  --experiment-dir ../experiments/sft_mini_v1 `
  --bundle exports/sft_mini_one_epoch_v1_fp32_v1 `
  --audit-dir ../experiments/sft_mini_v1/results/final_run_v1 `
  --output-dir ../experiments/sft_mini_v1/results/s1_v1 --run
```

输出已存在，禁止覆盖；不要为评分意见变化重新生成回答。若修订评分，保留本版并新增判定版本，保持B0/S1同一检查点。
