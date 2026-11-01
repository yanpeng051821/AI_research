# 正式SFT启动记录

2026-10-31北京时间17:43，用户明确要求启动后，按冻结配方从B0、step0后台启动正式SFT，目标49,087次更新。监督进程PID20340，训练Python PID35096；此处PID为启动快照，不是永久存活证明。

启动核验时已完成7次更新、提交112条记录，manifest为running/formal_sft；初始全验证与首批更新已正常执行，stderr为空。每次更新窗口约0.735秒，末次reserved约3.88GiB、峰值allocated约3.77GiB。未更改冻结参数。

外层监督状态和stdout/stderr：implementation/runs/sft_mini_one_epoch_v1_supervisor/。内部运行状态与指标：implementation/runs/sft_mini_one_epoch_v1/。大checkpoint按合同写E:/AI_research_checkpoints/minimind/sft_mini_v1/formal，每1000步保存；每5000步全验证，累计活动预算14小时。

后台监督设置prevent_idle_sleep=true，结束后释放该请求，并记录Python退出码。它不阻止用户手动休眠、关机、系统重启或硬件断连。保持E盘连接，避免重新打开大量占内存应用；关闭本聊天不等于停止独立后台进程。

实际命令由run_sft_background.ps1启动：.venv/Scripts/python.exe -X utf8 -u scripts/train_frozen_sft.py --freeze ../experiments/sft_mini_v1/training_recipe/training_freeze.json --run --output-dir runs/sft_mini_one_epoch_v1。冻结合同身份3ac716dee78ce33d5d418e7f1f5e7e6b176dcdad6d06c52ac608322172c1abe9。
