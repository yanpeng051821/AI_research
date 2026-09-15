# 06 服务器环境、最长样本与显存预飞行

## 1. 目的

租到 GPU 不等于可以启动训练。服务器预飞行要在产生昂贵计算前证明四件事：

```text
代码和依赖是指定版本
数据盘中的文件是冻结 artifact
模型和评测缓存可用
最坏长度样本能完成真实 optimizer step
```

本案例状态：`COMPLETE`。22,295-token 的 32K 路径探针失败，之后生成新的 16K artifact
并重新冻结合同；最终 16K 预飞行通过，正式 S1 在 A100 80GB 上完成。

## 2. 实例与磁盘规划

本案例使用单卡 A100/A800 80GB。16K 长序列、FP32 主参数、BF16 autocast、gradient
checkpointing 和 AdamW 的组合在 80GB 上余量不宽，因此不能选择“刚好能加载模型”的卡。

实例创建前估算：

- 模型、优化器状态和 checkpoint 数量；
- 训练/验证 artifact 大小；
- Hugging Face、uv、vLLM 缓存；
- 日志、逐样本评测和临时文件；
- 至少 50GB 持久盘余量；
- preflight、训练、保存、评测分别需要的时间。

正式 S1 启动前的数据盘约有 211GB 可用。这个数字只属于当时实例；下次训练仍要重新检查，
不能依赖本案例的历史截图。

## 3. 干净环境安装

从最终 Git commit 创建新检出。不要复用另一检出的 editable `.venv`。

```bash
cd /workspace/<exact-checkout>/independent_implementation

command -v uv || python -m pip install uv
uv sync --frozen --all-groups
uv pip install -r requirements-server.txt
uv pip install setuptools
uv pip install 'flash-attn==2.7.4.post1' --no-build-isolation

uv run python -c "import post_training_core; print(post_training_core.__file__)"
uv pip freeze > evidence/server/pip-freeze.txt
nvidia-smi -q > evidence/server/nvidia-smi.txt
```

`post_training_core.__file__` 必须指向当前检出。若指向数据盘中的旧目录，立即停止；这说明
editable install 漂移，继续训练会让 Git commit 与实际执行代码不一致。

## 4. 服务器 preflight

```bash
uv run python scripts/server_preflight.py \
  --train-artifact artifacts/<frozen-id>/train.jsonl \
  --validation-artifact artifacts/<frozen-id>/validation.jsonl \
  --data-manifest artifacts/<frozen-id>/data_manifest.json \
  --bundle-manifest evidence/server/server_bundle_manifest.json \
  --minimum-free-gib 50 \
  --output evidence/server/server_preflight.json
```

报告必须 `passed: true`。检查范围包括 CUDA/BF16、包版本、LightEval、vLLM、Math Verify、
FlashAttention、数据 hash、lock hash、源码树和磁盘余量。

bundle manifest 要在完整数据 artifact、人工复核 summary 和 decision artifact 都可访问的环境
中生成，再随交付包传到服务器。只复制 review summary 而没有它指向的 decision artifact，
无法通过人工复核完整性校验。

如果评测将在另一台机器执行，可以显式使用训练-only 选项，但必须在实验合同中记录；不能
因为环境失败临时跳过检查。

## 5. 最长保留样本探针

不要用平均长度或最短样本估显存。探针必须包含：

```text
真实最长 input_ids
真实 assistant labels
model forward
backward
gradient clipping
AdamW 第一次 step
scheduler step
zero_grad
```

```bash
uv run python scripts/probe_longest_training_sample.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output evidence/server/s1-longest-sample-memory.json
```

验收：`status=passed`、sequence length 不超过并接近 16,384、loss/grad norm 有限、第一次
AdamW step 完成、显存统计已落盘。

## 6. 为什么第一次 optimizer step 特别重要

模型加载完成时，AdamW 的一阶和二阶动量状态可能还没有为全部参数分配。第一次
`optimizer.step()` 才可能创建 `m/v`。因此：

```text
forward/backward 能跑
!=
完整训练 step 能跑
```

只测 forward 或只看到 CUDA 可用，不能关闭显存门禁。

## 7. Allocated、reserved 与 nvidia-smi

至少记录：

```text
memory_allocated
memory_reserved
max_memory_allocated
max_memory_reserved
mem_get_info 的 free/total
```

观察点：模型放置后、优化器创建后、第一次 micro-batch 前后、第一次 optimizer step 前后、
zero-grad 后、checkpoint/eval 前后。

解释：

| 现象 | 更可能的含义 |
| --- | --- |
| `reserved` 高但 `allocated` 随 batch 回落 | PyTorch 缓存池，未必泄漏 |
| `allocated` 每个 micro-batch 固定增长 | 可能保留计算图或带梯度 tensor |
| 长样本时升高、结束后回落 | 正常 activation 峰值 |
| 首个 optimizer step 突然增加 | AdamW 状态懒创建 |
| `nvidia-smi` 79GB，但 allocated 明显更低 | 仅凭外部观察不能判定真实活跃张量 |

## 8. 本案例的真实转折

最初 32K artifact 的最长样本为 22,295 token，在 A100-80GB 探针中失败。我们没有把
异常解释成“80GB 也不够所以项目失败”，而是回到目标：第一次实验是建立可靠流程，不是
保留每一条极端长样本。于是版本化生成 16K 派生 artifact，并隔离 984/32 条超长记录。

这是一项实验设计变更，因此重新计算了数据 hash、样本数、step、warmup 和成本。它不是
运行时偷偷减小 `max_length`。

## 9. 验收门与落盘物

- [x] 最终代码检出、import path 和依赖 lock 一致。
- [x] 16K 正式配置对应的 `server_preflight.json` 全部通过。
- [x] 16K artifact 的最长样本完成包含 AdamW 的完整 step。
- [x] allocator telemetry 有命名阶段和 GiB/bytes 字段。
- [x] 最终 16K 路径没有 NaN、Inf、OOM 或 silent fallback。
- [x] attention backend 与合同一致。
- [x] 磁盘和时间预算满足正式训练及 checkpoint 恢复。

必须保存环境报告、命令、stdout/stderr、exit code、显存 JSONL/summary、探针结果和任何失败
快照。本案例依靠这些证据进入了 smoke；下次运行仍必须重新产生，而不是复用本次勾选结果。
