# MiniMind mini 预训练 micro-batch 基准（已被取代）

> 状态：本轮没有显式锁定 AdamW 的 `foreach/fused` 后端，不能作为正式 batch 决策。原始结果保留用于说明问题，正式依据见后续 `pretrain_mini_microbatch_benchmark_v2`。

## 1. 为什么要做这次测量

官方脚本的默认物理 batch 面向其运行环境，已有 8-step 工程短跑中的 `micro_batch=2` 又只证明链路可以运行。两者都不能直接回答本机 RTX 4060 Ti 8GB 上应该选择多大的物理 batch。

物理 batch 太小会增加 Python 循环、kernel launch 和同步开销；太大则会挤占显存，甚至在 Windows 上退回共享内存，导致速度显著下降。因此这里固定优化器每次更新看到的记录数，只改变这些记录如何拆成 micro-batch。

## 2. 如何保证比较公平

四个候选原定为：

| micro-batch | 梯度累积 | 每次更新的记录数 |
| ---: | ---: | ---: |
| 2 | 128 | 256 |
| 4 | 64 | 256 |
| 8 | 32 | 256 |
| 16 | 16 | 256 |

每个可运行候选都重新执行以下过程：

1. 用同一个模型 seed 从随机初始化构造模型；
2. 用同一个 sampler seed 读取相同的前 768 条记录；
3. 执行一个预热更新，使 AdamW 创建 `m/v` 状态并完成 CUDA 预热；
4. 再测量两个完整更新；
5. 记录端到端 records/s、有效 targets/s、峰值显存和设备空闲显存；
6. 不执行验证，也不保存 checkpoint，避免无关开销影响比较。

三档已完成候选的初始化模型 SHA256 均为
`272d31cc9061d16cf6a00f2486ba97de6777eda428fcd08c061faeb10cde1ab4`，样本前缀 SHA256 均为
`e1ed44ecad28e78424b9c174a95688b66038889f530b6cd773b0006e3191210d`。这证明它们比较的是同一初始化和同一批样本。

## 3. 测量结果

| micro-batch | 梯度累积 | records/s | targets/s | 中位更新耗时 | 峰值 allocated | 峰值 reserved | 最低设备空闲比例 | 判断 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 128 | 21.83 | 5,136.51 | 11.60 s | 2.43 GiB | 2.45 GiB | 55.12% | 可运行，但循环次数更多 |
| 4 | 64 | **22.61** | **5,318.71** | **11.21 s** | 3.73 GiB | 3.79 GiB | 38.31% | 采用 |
| 8 | 32 | 21.44 | 5,044.96 | 11.82 s | 6.32 GiB | 6.44 GiB | 5.26% | 余量不足且更慢 |
| 16 | 16 | - | - | - | - | - | 接近耗尽 | Windows 共享内存换页，终止 |

`micro-batch=16` 没有及时抛出 CUDA OOM，而是长期占用约 7.91/8.19 GiB 显存，Python 进程私有内存增长到约 16.1 GB，数分钟不能完成。它在工程上等同于不可用，不能因为“进程还活着”就视为通过。

完整机器可读结果见 `microbatch_benchmark.json`。该 JSON 的 SHA256 为
`c458b44cae12244bcc5015ac0be6ce06945002d260dbf1fcae9d471f4b25db14`。

## 4. 为什么选择 4 而不是 8

选择规则不是“找刚好不 OOM 的最大 batch”，而是：在至少保留 10% 设备空闲显存的候选中，选择端到端 records/s 最高者。

`micro-batch=4` 比 `2` 快约 3.5%，仍保留约 38% 的设备空闲量。`8` 的 records/s 反而比 `4` 低约 5.1%，最低空闲比例仅 5.26%。但由于优化器后端未显式冻结，这些数值只作为第一轮观察，不能直接决定正式训练配置：

```text
micro_batch_size = 4
gradient_accumulation_steps = 64
effective_batch_records = 256
```

## 5. 对正式训练预算的影响

训练集共有 1,265,235 条记录，因此一个 epoch 包含：

```text
完整更新数 = 1,265,235 // 256 = 4,942
tail update = 83 条记录
总更新数 = 4,943
```

按本次短基准的 `22.61 records/s` 直接外推，纯训练约需 15.5 小时。该估计尚未计入完整验证、checkpoint 写盘和运行波动，只能用于下一步设计评测与保存频率，不能当作正式 ETA。

## 6. 复现方式

在 `implementation/` 下执行：

```powershell
uv run python scripts/benchmark_pretrain_microbatch.py `
  --config configs/pretrain_mini_microbatch_benchmark.json `
  --output-dir ../experiments/pretrain_mini_microbatch_benchmark_v2
```

输出目录必须是新目录。执行器逐个候选写入 `partial_results.json`，全部完成后再生成最终报告，避免某一高档位失速时丢失已经完成的结果。
