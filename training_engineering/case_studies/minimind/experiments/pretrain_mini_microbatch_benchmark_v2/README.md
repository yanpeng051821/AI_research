# MiniMind mini 预训练 micro-batch 基准

> 结论：在显式固定 AdamW `foreach=false`、`fused=false` 后，正式配方采用 `micro_batch_size=4`、`gradient_accumulation_steps=64`，每次完整更新消费 256 条记录。

## 1. 测量目标

官方脚本的默认物理 batch 面向其运行环境，已有工程短跑中的 `micro_batch=2` 又只证明链路可以运行。本基准要回答的是：在 RTX 4060 Ti 8GB、长度 768 和正式模型尺寸下，哪一种物理 batch 能在保留可靠显存余量的同时提供较高吞吐。

这里不同时搜索逻辑 batch。每个候选都让一次 optimizer update 消费 256 条记录，只改变这些记录被拆成多少次前向和反向计算。

## 2. 控制变量

| micro-batch | 梯度累积 | 每次更新的记录数 |
| ---: | ---: | ---: |
| 2 | 128 | 256 |
| 4 | 64 | 256 |
| 8 | 32 | 256 |
| 16 | 16 | 256 |

每个可运行候选均执行以下过程：

1. 使用相同模型 seed，从随机初始化重新构造模型；
2. 使用相同 sampler seed，读取相同的前 768 条记录；
3. 显式使用 AdamW `foreach=false`、`fused=false`；
4. 先执行一个预热更新，使 AdamW 创建 `m/v` 并完成 CUDA 预热；
5. 再测量两个完整更新；
6. 不执行验证，不保存 checkpoint，只测训练更新路径。

三档完成候选的初始化模型 SHA256 均为
`272d31cc9061d16cf6a00f2486ba97de6777eda428fcd08c061faeb10cde1ab4`，样本前缀 SHA256 均为
`e1ed44ecad28e78424b9c174a95688b66038889f530b6cd773b0006e3191210d`。

## 3. 测量结果

| micro-batch | 梯度累积 | records/s | targets/s | 中位更新耗时 | 峰值 allocated | 峰值 reserved | 最低设备空闲比例 | 判断 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 128 | 21.59 | 5,080.65 | 11.72 s | 2.43 GiB | 2.45 GiB | 55.12% | 可运行，但循环次数更多 |
| 4 | 64 | **22.59** | **5,314.62** | **11.22 s** | 3.73 GiB | 3.79 GiB | 38.31% | 采用 |
| 8 | 32 | 21.42 | 5,040.58 | 11.83 s | 6.32 GiB | 6.44 GiB | 5.26% | 余量不足且更慢 |
| 16 | 16 | - | - | - | - | - | 接近耗尽 | 预备运行进入共享内存换页，终止 |

完整机器可读结果见 `microbatch_benchmark.json`，其 SHA256 为
`eca79fcfb35b07dbfc170f1011c889aea5f60c856d842814fc2f6f3083f555bd`。

## 4. 配方决定

选择规则是在至少保留 10% 设备空闲显存的候选中，选端到端 records/s 最高者。`4 × 64` 比 `2 × 128` 快约 4.6%，并保留约 38% 的设备空闲量。`8 × 32` 更慢，且最低空闲比例只有约 5.26%。因此冻结：

```text
micro_batch_size = 4
gradient_accumulation_steps = 64
effective_batch_records = 256
```

训练集共有 1,265,235 条记录，因此一个 epoch 包含 4,942 个完整更新和一个消费剩余 83 条记录的 tail update，共 4,943 次 optimizer update。tail update 不补齐、不重复样本，也不丢弃。

按 `22.59 records/s` 直接外推，纯训练约需 15.6 小时。该估计尚未包括完整验证、checkpoint 写盘和长期运行波动。

## 5. 第一版为何被取代

第一版基准没有显式传入 AdamW `foreach/fused`，可能由 PyTorch 根据运行环境自动选择后端。虽然第二版结果与第一版接近，但正式实验不能依赖这种隐式选择，因此第一版保留为方法修订记录，不再作为训练合同依据。

## 6. 复现命令

在 `implementation/` 下执行：

```powershell
uv run python scripts/benchmark_pretrain_microbatch.py `
  --config configs/pretrain_mini_microbatch_benchmark.json `
  --output-dir ../experiments/pretrain_mini_microbatch_benchmark_v3
```

输出目录必须是新目录。执行器会在每档完成后更新 `partial_results.json`，全部候选完成后再写最终报告。
