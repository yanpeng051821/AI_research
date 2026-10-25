# 正式预训练启动记录

> 2026-10-25 10:43（北京时间）按用户要求后台启动。此页记录启动方式与查看入口，实时进度以运行目录中的 manifest 和 metrics 为准。

## 1. 本次运行

| 项目 | 实际值 |
| --- | --- |
| run ID | pretrain_mini_one_epoch_v1 |
| 起点 | seed=42 随机初始化，resume_from=null |
| 预算 | 1,265,235 条、301,810,690 个有效 target、4,943 次更新 |
| 设备 | 本机 RTX 4060 Ti，FP32 状态 + BF16 autocast |
| Python PID | 1640，仅代表本次启动，不作为后续恢复进程的固定 ID |
| 监督进程 PID | 10352 |
| 核心合同 | b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58 |
| 外层冻结清单 | b496d4fb257bae495d6a3a86552d948206227bd5a8a78a08fbccfd425cc35828 |

启动前单独检查通过；后台 Python 进程也再次执行完整冻结核验，随后调用原 run_experiment。当前运行不是从 step 20 短跑继续。

北京时间 10:44:59 核对：完整初始验证 NLL 为 8.912246169216113，与短跑基线一致，耗时 88.59 秒；第 1 次更新完成，loss=8.9079907121，累计提交 256 条、63,875 个有效 target，参数更新耗时 11.49 秒。监督状态 running，错误日志为空。这是启动时验收记录，不代表之后的实时进度。

## 2. 后台方式

使用 [run_pretrain_background.ps1](../../../implementation/scripts/run_pretrain_background.ps1) 作为隐藏的 Windows PowerShell 监督进程，启动项目 `.venv\Scripts\python.exe -u` 执行冻结入口，并把标准输出、标准错误重定向到文件。

监督进程记录 Python PID，等待进程结束并记录退出码。训练期间通过线程电源请求阻止空闲自动休眠，结束时释放请求，不改全局电源计划；该请求不阻止用户手动休眠、关机或系统重启。

## 3. 实时记录

工作目录为 `implementation/`：

```powershell
Get-Content runs/pretrain_mini_one_epoch_v1/run_manifest.json
Get-Content runs/pretrain_mini_one_epoch_v1/metrics.jsonl -Tail 5
Get-Content runs/pretrain_mini_one_epoch_v1.process/process.json
Get-Content runs/pretrain_mini_one_epoch_v1.process/stderr.log -Tail 20
```

需要跟随完整控制台输出时：

```powershell
Get-Content runs/pretrain_mini_one_epoch_v1.process/stdout.log -Tail 10 -Wait
```

Ctrl+C 结束这条查看命令，不会停止后台训练。manifest 的 running 只是最后写入的状态，判断存活还应结合 PID、日志更新时间和 GPU 活动。

日志和实际合同快照在 `implementation/runs/pretrain_mini_one_epoch_v1/`。checkpoint 每 100 步及最终保存到 `E:\AI_research_checkpoints\minimind\pretrain_mini_one_epoch_v1`，首次周期 checkpoint 对应 step 100。

## 4. 本目录的启动证据

- [frozen_launch.json](frozen_launch.json)：后台进程的冻结检查结果，mode=run。
- [background_launch.json](background_launch.json)：启动时进程信息快照，不随实时进度更新。
- [preflight.json](preflight.json)：目录迁移后、正式启动前的只读核验。

长训练按短跑外推约 16.3 小时，起点为北京时间 10 月 25 日 10:43，粗估到 10 月 26 日凌晨 03:00 左右。后续用真实吞吐及验证、保存耗时更新估计，不将这个时间当作完成保证。
