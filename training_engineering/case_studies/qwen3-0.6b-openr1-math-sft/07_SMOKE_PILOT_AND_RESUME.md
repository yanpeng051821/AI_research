# 07 Smoke、Pilot、Checkpoint 与恢复

## 1. 为什么要分层放大

smoke、pilot 和正式训练解决不同问题：

| 类型 | 主要目的 | 能否形成正式能力结论 |
| --- | --- | --- |
| 1-step probe | 最坏样本、显存、AdamW 首步 | 否 |
| 5/20-step smoke | 训练、保存、生成链路是否通 | 否 |
| 100-step pilot | 吞吐、恢复、趋势、预算 | 只能形成工程/趋势结论 |
| 479-step S1 | 冻结合同下的正式训练 | 完成同合同评测后才可以 |

先小后大不是保守形式，而是让失败发生在便宜、易定位的阶段。

## 2. Smoke 应验证什么

一个有价值的 smoke 至少覆盖：

```text
真实模型和 tokenizer 加载
真实审计数据读取
completion-only labels
forward/backward
gradient clipping
optimizer/scheduler
至少一个 checkpoint
从 checkpoint 恢复
导出模型重新加载
短 generation health
终态 manifest
```

观察的不只是 loss：

- loss 和 grad norm 是否为有限值；
- LR 是否按 optimizer step 变化；
- global step 与 accumulation 对齐；
- checkpoint 中是否有 optimizer/scheduler/trainer state；
- 恢复后是否继续而不是重训第一个 batch；
- generation 是否为空、报错、无法停止或格式异常。

## 3. Pilot 的冻结原则

pilot 必须使用专用配置和输出目录，显式限制 optimizer steps。不要只在脑中记得“跑 100
步”；边界要进入 resolved config/run manifest。

正式 pilot 示例：

```bash
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_pilot100_16k.yaml \
  --output-dir runs/<pilot-id> \
  --stop-after-steps 50
```

恢复：

```bash
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_pilot100_16k.yaml \
  --output-dir runs/<pilot-id> \
  --resume-from-checkpoint runs/<pilot-id>/trainer/checkpoint-50
```

恢复前比较 run identity：数据、sample order、源码、配置、模型、tokenizer 均不能变化。

## 4. 本案例 20-step smoke

训练指标约为：

```text
train loss: 0.782 -> 0.646
grad norm: 4.58 -> 0.70
validation subset NLL: 0.8050 -> 0.6754
relative NLL reduction: about 16.1%
```

这证明真实训练链路能学习。但生成检查在 512 和 2048 token budget 下都没有在预期 EOS
处结束。正确结论是“训练链路通过、generation gate 未关闭”，而不是“模型训练失败”或
“模型能力提升”。训练计算、生成终止和任务正确率是三个不同问题。

## 5. 本案例 100-step pilot

最终从 `checkpoint-50` 恢复到 global step 100，exit code 为 0，恢复段没有 OOM、NaN 或
Inf。全量固定验证集结果：

```text
B0 NLL:       0.7804831795
Pilot-100 NLL: 0.5952246359
relative reduction: about 23.74%
```

两者使用同一 1,968 条 validation artifact 和 11,367,594 个有效 token，因此 NLL 可以
直接比较。它证明模型更贴合目标 SFT 分布，不证明 MATH-500 或通用能力已经改善。

按约 65 秒/optimizer step 估算，479-step 正式训练本体约 8.7 小时，还要预留加载、保存、
中断恢复和评测时间。

## 6. 真实遇到的工程问题

### DataLoader workers 失败

16K 数据配合 `num_workers=2` 出现 JSON 读取失败、`Too many open files` 和共享内存文件
描述符错误；服务器 `ulimit -n` 为 1024。主进程读取数据正常，因此问题属于多进程交付
路径，不是 JSONL 损坏。

处理：pilot 和 S1 固定 `dataloader_num_workers=0`。若未来要提高 workers，必须单独做
吞吐与正确性实验，不能在正式 S1 中临时修改。

### 执行到了旧源码

持久化 `.venv` 的 editable install 指向旧目录，traceback 路径与当前 Git clone 不一致。
第一次失败被归类为环境/可复现性失败，不是算法失败。后续规则是每次精确检出创建独立
环境，并在启动前打印模块路径。

### Pilot 边界没有真正冻结

早期配置没有清楚表达预期 max steps。修复后同时在 config 与 CLI stop boundary 中固定，
并由 dry-run manifest 检查实际 `max_steps`。

### 外部显存观察不能解释 allocator

运行中曾看到接近 79GB 的 `nvidia-smi`，后续恢复段又有约 51GB 的外部峰值记录。没有
allocator 级分阶段数据，就不能判断差异来自 reserved cache、采样时点或活跃 tensor。
因此正式 S1 前加入结构化 CUDA telemetry。

## 7. Checkpoint 验收

恢复前检查：

```text
checkpoint 属于同一个 run 输出目录
run identity 完全相同
global step 正确
optimizer state 存在
scheduler state 存在
trainer state 存在
sample order 没有变化
checkpoint 能被模型加载
```

恢复后检查：

```text
第一条日志 step 大于 checkpoint step
LR 延续而非重新 warmup
没有重复消费已训练样本
新 checkpoint 和 final model 正常写出
manifest 终态为 paused/completed/failed，而非永久 running
```

## 8. Pilot 何时允许进入 S1

- [ ] 完整到达预期 step，exit code 0。
- [ ] checkpoint/resume 通过。
- [ ] loss、grad norm、LR 和 token count 合理。
- [ ] validation NLL 使用与 B0 相同的 artifact。
- [ ] generation/export 基本可用。
- [ ] 时间和显存成本可接受。
- [ ] 失败与中断均有机器可读终态。
- [ ] 正式评测入口至少做过小规模 paired preflight。

在 pilot 收尾的当时，前六项大体已有证据，但 CUDA telemetry 实机验证和 MATH-500 paired
preflight 仍未关闭，所以当时的正式 S1 决策是 `NO-GO`。这句话是本阶段历史快照，不是
当前状态。后续训练记录显示，显存和训练恢复门被关闭后正式 S1 已经完成，但 paired MATH
评测风险没有被充分关闭，最终在正式评测阶段以长生成和预算超限的形式暴露；详见
`10_INCIDENTS_AND_REUSABLE_LESSONS.md` 与 `11_FORMAL_S1_RESULTS_AND_RELEASE.md`。
