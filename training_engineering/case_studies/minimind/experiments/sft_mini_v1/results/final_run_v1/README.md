# 正式SFT训练验收与本地导出

## 1. 验收结论

2026-11-01北京时间05:03核验通过。训练于10月31日17:43:07启动，11月1日04:26:24退出，监督进程退出码0，stderr为空。完成完整一轮，不是预算暂停。本文只判断工程执行与交付一致性，能力结果见[配对分析](../paired_analysis_v1/README.md)。

| 项目 | 真实结果 |
| --- | --- |
| optimizer更新 | 49,087次，日志从1连续覆盖到49,087 |
| 训练记录 | 785,377条，提交位置与sampler一致 |
| micro-batch | 196,345个；最后一次更新只有1个micro-batch、1条记录、413个有效目标 |
| 有效助手目标 | 326,249,045，与数据及训练合同一致 |
| 参数/状态 | 最终FP32模型和AdamW状态有限；90个optimizer状态条目均在49,087步 |
| 恢复状态 | scheduler、完整样本排列、已提交位置与CPU/CUDA RNG已核对；未重新执行最终checkpoint恢复训练 |
| 保存 | 50份checkpoint及对应receipt齐全；最终文件完整计算SHA并加载，未逐份加载历史50份 |
| 最终验证 | 全2,048条BF16 NLL=1.7212176707061748，833,052个目标 |

最终checkpoint：`E:/AI_research_checkpoints/minimind/sft_mini_v1/formal/checkpoint-49087.pt`，770,864,235字节，SHA256：

```text
181ce5bbc64c0c6ad0f3b79682f71df33206665c8212182422ce1363fec1e03a
```

## 2. 验收实现

新增[finalize_sft_run.py](../../../../implementation/scripts/finalize_sft_run.py)，不改冻结训练代码。它按以下顺序检查：

1. 调用原训练冻结检查，重新核对数据、模型起点、配置、代码、依赖和接受证据。
2. 交叉核对训练manifest、监督退出与stderr，再逐条检查更新步数、消费位置、尾批、目标累计和数值有限性。
3. 核对验证/保存调度与50份文件/receipt清单；对最终文件做字节校验、反序列化和内部状态检查。
4. 按合同架构严格加载最终模型，确认原生FP32及embedding/lm_head共享，导出不含优化器的推理包。
5. 在新进程中只使用包内代码、配置和tokenizer，比较固定输入、完整参数摘要、logits与短生成。

完整一轮计数、状态和时间统计见[training_audit.json](training_audit.json)。报告明确保留限制：历史文件清单齐全不等于已逐份验证所有tensor状态，观测显存稳定也不是所有泄漏情形的证明。

## 3. 实际成本

| 组成 | 实际时间 |
| --- | ---: |
| 墙钟总时间 | 10小时43分17秒 |
| 更新窗口累计 | 10小时12分24秒 |
| 50次checkpoint保存、刷盘与receipt累计 | 22分46秒 |
| 11次全验证累计 | 6分53秒 |
| 其余启动/检查/调度等 | 约1分13秒 |

更新窗口均值0.74856秒，中位0.73964秒。墙钟开销与短跑估计基本同量级，未用满14小时活动预算；保存/验证不是免费开销，已经计入总耗时。50份checkpoint合计38,543,211,750字节，约35.90GiB，均保留在E盘，未删除。

更新后allocated全程为846,709,760字节，约0.789GiB；峰值allocated约3.773GiB，reserved范围约3.365–3.885GiB。记录的是程序对应时点的allocator状态，不等于整机nvidia-smi占用。

裁剪前grad norm中位1.1799，49,029次高于阈值1.0。最大4.6601出现在只含1条样本的最后一次更新，已按原裁剪流程处理且数值有限；不能仅凭它断定发散，也不将该单样本窗口与16条窗口直接等同。首/末1,000次更新按目标加权loss约2.0326/1.7168，不是等权平均各batch loss。

## 4. 导出与独立加载

本地模型包：`implementation/exports/sft_mini_one_epoch_v1_fp32_v1/`。它保持原生FP32，不恢复训练状态，不是Transformers AutoModel适配包。命令行原始prompt入口仅作续写/加载检查；助手评测仍走固定chat template。

[export_verification.json](export_verification.json)确认：63,912,192个参数，离线新进程使用包内runtime；两条短前缀的输入、mask、logits和短生成逐值相同，最大logit差0，权重共享保持。这个探针不等于全部验证集或任意长输入数值等价，完整FP32评测另行执行。

S1参数状态摘要：`733c18c9775eadd7f67330fabb0568ab61103f9d5fba995e6d967405c904cc62`。

导出包清单副本见[export_manifest.json](export_manifest.json)。本地包未上传HF，checkpoint未移动或删除。

## 5. 命令与回归

在implementation目录执行如下关系；本次输出已经存在，复练需新目录，不能覆盖：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/finalize_sft_run.py `
  --experiment-dir ../experiments/sft_mini_v1 `
  --run-dir runs/sft_mini_one_epoch_v1 `
  --supervisor-dir runs/sft_mini_one_epoch_v1_supervisor `
  --report-dir ../experiments/sft_mini_v1/results/final_run_v1 `
  --bundle-dir exports/sft_mini_one_epoch_v1_fp32_v1
```

收尾、导出、评测与评分相关定向回归31项通过，见[测试XML](closeout_tests.xml)，不是全项目测试。覆盖尾部计数、非有限值拒绝、S1来源/架构约束、实际提示一致、停止原因以及既有导出/评分行为。没有把测试通过数当成模型效果证明。
