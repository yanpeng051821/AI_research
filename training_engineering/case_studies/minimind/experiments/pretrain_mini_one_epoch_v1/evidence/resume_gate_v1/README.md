# 正式候选配方的暂停恢复验收

> 日期：2026-10-25。连续 4 步与“2 步暂停、退出、恢复至 4 步”逐状态比较通过，两个最终 checkpoint 的文件 SHA256 完全相同。本次是正式训练前的门禁实验，完整一个 epoch 尚未启动。

## 1. 实验目的

第六周与早期 8-step probe 已经验证过基本恢复机制。本次要确认：把模型尺寸、编码长度、有效 batch、optimizer 后端、完整验证集和机械盘存储全部替换为正式候选设置后，恢复仍能复现连续训练。

输入配置见 [candidate_config.json](candidate_config.json)，运行时解析出的完整合同见 [contract.json](evidence/pretrain_mini_resume_gate_continuous_v1/contract.json)。合同 SHA256 为 `b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58`。

| 项目 | 配置 |
| --- | --- |
| 模型 | MiniMind-3 Dense，63,912,192 个独立参数 |
| 训练记录 | 1,265,235 |
| 编码长度 | 768 |
| micro-batch / accumulation | 4 / 64 |
| 完整更新预算 | 4,943 |
| 精度 | FP32 参数与状态，CUDA BF16 autocast |
| optimizer | AdamW，lr=5e-4，weight_decay=0.01，foreach=false，fused=false |
| scheduler | 无 warmup，完整 4,943-update 余弦计划 |
| 验证 | 全部 5,000 条，1,206,758 个有效 target，batch=4 |
| checkpoint | E 盘 USB HDD；每次主动暂停额外保存 |

`max_updates=null` 表示从一个 epoch 的数据量解析训练预算。短跑使用启动参数 `--stop-after-updates` 提前停止，保留完整学习率计划。将配置直接改为 `max_updates=4` 会改变 scheduler，不能回答本次问题。

## 2. 对照设计

```text
A：初始化 → step 1 → 2 → 3 → 4 → 保存并退出
B1：相同初始化 → step 1 → 2 → 保存并退出
B2：加载 B1 checkpoint-2 → step 3 → 4 → 保存并退出
```

新启动各用独立的 D 盘日志目录。A 的 checkpoint 写到 `E:\AI_research_checkpoints\minimind\pretrain_mini_resume_gate_v1\continuous`；B1/B2 共用其并列的 `resumed` 目录。正式一个 epoch 的 checkpoint 目录不参与本次短跑。

B1 退出后先运行只读检查器，确认 step 2 checkpoint 的 receipt、合同、字节哈希和内部进度符合要求，再显式把该路径交给 B2。恢复后先运行完整验证，比较保存前后的 step 2 NLL，然后继续参数更新。

## 3. 等价性证据

完整程序化结果见 [comparison.json](comparison.json)。

| 检查项 | 结果 |
| --- | --- |
| 最终 step | A/B2 均为 4 |
| 已提交样本 | A/B2 均为 1,024 |
| 累计有效 target | A/B2 均为 245,140 |
| tensor leaf | 363 个逐元素完全一致 |
| 非 tensor 状态 | optimizer 参数组、scheduler、sampler、metadata 等完全一致 |
| CPU/CUDA RNG | 完全一致 |
| step 1～4 关键日志 | loss、有效 target、梯度范数、学习率及提交位置完全一致 |
| 完整验证 NLL | A/B2 均为 7.851118193106291 |
| 最终 checkpoint SHA256 | A/B2 均为 `5eba8ae2ddff5e67eb72a66b206df37f533a046a921eb5ac44fdd2d33f30a973` |

step 2 暂停前与恢复后的完整验证 NLL 也都为 `8.196152648730045`。时间、吞吐、显存缓存池占用不要求逐值一致：它们受到系统调度和进程初始化影响，不属于恢复后必须相同的参数更新轨迹。

比较脚本同时修复了两个历史假设：checkpoint 路径改为读取 result 中的 `checkpoint_dir`；报告不再把步数固定写成 eight-update 或把完整验证写成 subset validation。脚本在本次真实机械盘 checkpoint 上实际运行通过。

## 4. 稳定性观察

恢复等价性通过后，从 B2 的 step 4 继续到 step 20，保持候选合同不变。延长观察的原因是前四步裁剪前梯度范数均超过 1，需要辨别这是初始化初期现象，还是持续的梯度异常。

step 1～7 发生裁剪；step 8～20 均未触发，裁剪前范数范围为 `0.4036～0.8805`。step 1 loss 为 `8.90799`，step 20 为 `7.12169`，中间允许不同 batch 引起的正常波动。前 20 步共提交 5,120 条、1,223,955 个有效 target。

完整验证 NLL 在 step 0/2/4/20 分别为 `8.9122461692 / 8.1961526487 / 7.8511181931 / 7.0976711146`。step 20 已正常保存并以 `paused` 收尾；未发现 NaN、Inf、OOM 或持续的活跃显存增长。step 20 checkpoint SHA256 为 `c80de6910e6ed3052f3702c1e462e74f672186a15ef5b1be2c05fa0ced5ce141`。

当前证据支持保留无 warmup 候选配方。它没有比较带 warmup 的版本，因此不能得出 warmup 没有收益的结论；20 步的正常运行也不能保证整个 epoch 都正常。

## 5. 成本与文件

本次三次对照保存的端到端耗时为 `24.64 / 24.65 / 24.62` 秒，后续 step 20 保存为 `24.89` 秒。完整验证实际约 87 秒。前 20 步每个数据准备及更新窗口平均约 11.39 秒，峰值 allocated 约 3.77 GiB，最小设备空闲约 2.97 GiB。

按上述实测外推，完整 4,943 步、50 次保存及 11 次验证约需 16.3 小时；启动、恢复次数和系统波动仍会影响实际耗时。每次额外暂停恢复会增加保存和诊断验证，原 11 次验证预算假定不中断运行。

`evidence/` 保存各分支的合同、解析配置、manifest、逐步日志和结果；checkpoint 与 receipt 真实文件保留在 E 盘。小型证据不能代替大文件备份。

原入口的 `result.json.limitations` 沿用了早期 probe 的通用说明，包含 subset 一词；本次实际验证范围以 `validation_scope=full_frozen_validation`、5,000 条记录和 1,206,758 个 target 为准。原始结果按实际输出保留，此处解释范围差异，避免改写运行证据。

## 6. 后续决定

本次门禁支持进入最终训练合同冻结。短跑 checkpoint 保留为实验产物；正式运行采用独立目录并从随机初始化执行一个完整 epoch。尚未启动正式训练，也未把 20 步结果记为本周预训练完成。
