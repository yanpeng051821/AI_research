# SFT真实GPU短跑：20步完成

## 1. 结果与范围

2026-10-31，用户关闭暂时不用的应用后，按原候选配置完成20次optimizer更新并保存checkpoint，外层进程退出0，状态paused。**短跑成功，不是正式完整一轮，也没有自动继续。** 数据、模型、精度、batch、学习率及训练入口与前三次失败尝试保持一致。

| 项目 | 实测 |
| --- | ---: |
| 更新 / 已提交记录 | 20 / 320 |
| 有效助手目标 | 128,859 |
| 前64条验证NLL，BF16 | 2.8312329279 → 2.5991891775 |
| 验证有效目标数 | 27,337，两次相同 |
| 单次更新中位耗时，去首步 | 0.733秒 |
| 含数据取批的窗口中位耗时，去首步 | 0.748秒 |
| 峰值allocated | 3.773GiB |
| 末次reserved / 更新后allocated | 3.885 / 0.789GiB |
| checkpoint大小 | 770,864,235字节，约735MiB |
| 保存与receipt耗时 | 约1.544秒 |

所有更新的loss、裁剪前梯度范数与学习率有限，4个micro-batch累积后每次提交16条，最终320条。首步后更新后allocated保持846,709,760字节，reserved从第2步起稳定；这20步内未见逐步增长，不能据此保证整轮无泄漏或无其他异常。grad_norm是裁剪前值，大于1不等于没有裁剪。

验证是工程前缀64条、训练用BF16精度，不是最终全2,048条FP32配对评测。下降只能说明本次短跑在该前缀上的NLL下降，不足以宣布助手任务能力提高，也不能直接对比B0全量NLL=2.395456。

## 2. 启动与产物

从implementation目录执行的实际命令：

下面保留实际运行时的旧配置路径；运行后唯一候选文件已迁移至[training_recipe](../../training_recipe/sft_mini_candidate.json)，以后运行将config参数改为../experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json，内容与本次快照相同。

```powershell
uv run --locked python -X utf8 -u scripts/train_sft.py `
  --config configs/sft_mini_candidate.json `
  --output-dir ../experiments/sft_mini_v1/results/probe20_v4 `
  --probe-updates 20 --validation-records 64
```

| 文件 | 用途 |
| --- | --- |
| [candidate_contract.json](candidate_contract.json) | 本次候选配置、数据/B0来源、样本顺序、代码及依赖身份 |
| [metrics.jsonl](metrics.jsonl) | 每步有效目标、loss、梯度、学习率、耗时、显存和验证/保存事件 |
| [result.json](result.json) | paused、step20、已提交320条及checkpoint路径 |
| [run_manifest.json](run_manifest.json) | 本次运行终态 |
| [checkpoint receipt](checkpoint-20.pt.receipt.json) | checkpoint SHA256、合同、步数与提交位置 |
| [verification.json](verification.json) | 保存证据与checkpoint内部状态核对、原始输入hash及估时 |

checkpoint-20.pt位于本次输出目录，不推入Git；正式checkpoint保存位置尚未决定。摘要为e28317613ee01128d67ebe8ea3c1dfbac17a336b71e722933f81ea2224964b73。候选训练身份为e403c68bc3433e8071a6439179d24b6a2f14cdd9af5966ab9f5367bfaa32d914，与前次准备及失败尝试相同。

## 3. 核对与诊断解释

[verify_sft_probe.py](../../../../implementation/scripts/verify_sft_probe.py)已执行：读取已保存文件，核对20条更新顺序、有效目标求和、checkpoint字节SHA、模型来源元数据、scheduler进度、Adam各参数step、sampler排列及已提交位置，检查指标有限并再次确认读取期间文件未变。它只检查保存证据与内部状态，**没有执行实际恢复训练**。

```powershell
uv run --locked python -X utf8 scripts/verify_sft_probe.py `
  --run-dir ../experiments/sft_mini_v1/results/probe20_v4 `
  --output ../experiments/sft_mini_v1/results/probe20_v4/verification.rerun.json
```

重查时用新输出名，既有文件拒绝覆盖。详细故障历史见[分配排查](../PROBE20_STATUS.md)。本次启动前可用物理内存约7.24GiB，系统虚拟/提交相关可用约15.22GiB，GPU桌面占用1,041MiB；同配方随后运行成功，结合独立分配诊断，强烈支持先前系统内存/提交压力是影响因素，而不支持“8GB显存必然无法容纳该配置”。关闭多项应用改变了多个环境因素，不将某个应用或提交上限写成唯一已严格隔离的根因。

## 4. 估时与下一步

按0.748秒窗口中位数外推49,087次更新，**纯训练约10.2小时**，不含启动、验证、保存和长期吞吐波动。这只基于20步，不能当成完整墙钟承诺。前64条验证和D盘一次保存不足以直接确定全验证频率、E盘长期保存开销或最终磁盘预算。

下一步做真实GPU连续/恢复对照，再确认整轮时间可接受、验证/保存频率、存储与保留策略，最终冻结配方并单独确认正式启动。当前不把短跑checkpoint当作已批准正式SFT的续训起点。
