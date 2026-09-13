# 09 正式 S1 启动、观察、恢复与收尾手册

## 1. 文档用途与当前状态

这是一份执行手册，不是已经完成的实验报告。它负责把已经验证过的数据、代码、训练
合同和评测合同装配成一次正式 S1，并保证失败后仍然可以知道发生了什么、从哪里恢复。

在本手册编写时，当前决策是 `NO-GO`。正式 S1 只能在以下两个硬门关闭后启动：

1. 使用最终代码、最终配置和真实 16K 数据完成 GPU allocator telemetry 的一至两个
   optimizer update，覆盖 AdamW 第一次创建 `m/v` 状态、checkpoint 和最终模型导出。
2. 使用与正式 S1 相同的评测入口，分别对 B0 和一个 pilot/final-like checkpoint 完成
   小规模 MATH-500 预飞行，确认上下文预算、KV cache、vLLM 启动和终态 manifest 正常。

这两个门未关闭时，可以租 GPU 做预飞行，但不能把一次长时间训练称为正式 S1。

后续实际执行中，训练和 allocator telemetry 门被关闭，正式 S1 完成了 479 个 optimizer
updates；但 MATH paired preflight 没有形成足够可靠的长生成预算证据，正式训练仍然启动，
这构成了本案例的流程偏离。结果是训练工程本身完成，而 MATH-500 评测在 S1 后被预算门中止。
当前状态与最终结论见 `11_FORMAL_S1_RESULTS_AND_RELEASE.md`；本手册中的 `NO-GO` 只表示
启动前的历史状态，不能覆盖最终运行记录。

## 2. 正式 S1 的冻结身份

| 项目 | 冻结值 |
| --- | --- |
| Base 模型 | `Qwen/Qwen3-0.6B-Base` |
| 模型与 tokenizer revision | `311c62e88814bff7206909ccd330bab0a784743b` |
| 数据 | `gate0b-16k-derived-a` |
| 训练记录 | `61,224` |
| 验证记录 | `1,968` |
| 最大训练长度 | `16,384`，超长样本排除，不静默截断 |
| 训练阶段 | 一个 epoch，从 Base 开始，不从 pilot checkpoint 开始 |
| micro batch | `1` |
| gradient accumulation | `128` |
| optimizer updates | `479` |
| warmup updates | `15` |
| learning rate | `4e-5` |
| precision | FP32 主参数，BF16 autocast |
| attention | `flash_attention_2` |
| checkpoint cadence | 每 50 optimizer steps，最多保留 2 个 |
| 训练入口 | `scripts/train_sft_trl.py` |
| 配置 | `configs/gate0b/sft_s1_16k.yaml` |

当前 CPU dry-run 绑定过一次身份：

```text
source commit: e063e04ea5dcedaee4408cda994091d016a2a1c9
train SHA-256: 34f919f75ef49dcb45d2fa6bead80fa93667d04f709de48653a31f0fe2de3439
validation SHA-256: 60bbe71d35d86504527ee30860ad54ff83745de1549196c27ba86ea3678f3ff9
sample order SHA-256: 9af3e9338d993471846cc8de3bdca760eaf2f3242c5e536840dff8cd5346c2910
```

注意：只要正式启动前又修改了运行时代码、配置、`uv.lock` 或数据，上述 dry-run 身份就
已经过期。正确做法是重新生成 manifest，而不是为了匹配旧 hash 回退合理修复。

## 3. 预算与实例规划

已完成 pilot 的速度约为每个 optimizer step 65 秒。只计算 479 个 update，预计约
8.7 小时；环境检查、模型加载、checkpoint、验证和故障余量不在其中。

建议把租用分成两个窗口：

| 窗口 | 建议预算 | 目标 |
| --- | ---: | --- |
| 预飞行窗口 | 2 至 3 小时 | 关闭 GPU telemetry 与评测入口门禁，不启动正式 S1 |
| 正式窗口 | 12 至 14 小时 | 训练约 9 小时，保留 checkpoint、恢复和基础评测余量 |

MATH-500 的 B0 全量 2,000 次生成曾耗时约 108.6 分钟，S1 仍需同合同重跑。完整的
MATH-500、GSM8K 和 regression 可以在独立评测窗口完成，不应因为训练即将到期而修改
评测合同。

硬件最低采用单卡 A100/A800 80GB，并保留至少 50GB 可用持久化磁盘。80GB 是实测可行
边界，不是宽裕配置；不能以 `nvidia-smi` 的单一数字替代 allocator 证据。

## 4. 新实例初始化

以下命令从 `independent_implementation` 根目录执行。不要复制本地 `.venv`，也不要复用
来源不明的 editable installation。

```bash
set -euo pipefail

git status --short
git rev-parse HEAD
command -v uv || python -m pip install uv

uv sync --frozen --all-groups
uv pip install -r requirements-server.txt
uv pip install setuptools
uv pip install 'flash-attn==2.7.4.post1' --no-build-isolation

uv run python -c "import post_training_core; print(post_training_core.__file__)"
uv pip freeze > evidence/server/pip-freeze.txt
nvidia-smi -q > evidence/server/nvidia-smi.txt
```

验收条件：

- `git rev-parse HEAD` 是准备启动的明确 commit。
- `post_training_core.__file__` 指向当前 checkout，不指向旧数据盘或旧 clone。
- `uv sync --frozen` 没有改写 lock。
- FlashAttention、TRL、Transformers、PyTorch 和 LightEval 都可以导入。
- 单卡运行时只暴露一张 GPU。

如果模型与数据已在持久化盘，可以使用本地 snapshot/cache，但 manifest 中仍必须保存原始
repo id 和 revision。路径变化不能改变模型身份。

## 5. 数据与代码身份复核

先确认以下文件存在：

```text
artifacts/gate0b-16k-derived-a/train.jsonl
artifacts/gate0b-16k-derived-a/validation.jsonl
artifacts/gate0b-16k-derived-a/data_manifest.json
evidence/data_review_decisions.summary.json
```

server bundle manifest 必须在以下三者同时可访问的环境中生成：最终源码、完整 16K artifact、
人工复核 summary 及其指向的 decision artifact。通常在本地或持久化数据盘先生成，再把结果
与数据一起交付服务器。不能只把 summary 复制过去，因为 builder 会重新校验 decision 文件。

在该环境中执行：

```bash
uv run python scripts/build_server_bundle_manifest.py \
  --data-manifest artifacts/gate0b-16k-derived-a/data_manifest.json \
  --review-summary evidence/data_review_decisions.summary.json \
  --output evidence/server/server_bundle_manifest.json
```

如果正式启动 commit 相比生成 bundle 时发生变化，bundle 的 runtime tree 已过期，必须在
新 commit 上重新生成。把生成后的 `server_bundle_manifest.json` 上传到服务器相同路径。

运行服务器预检：

```bash
uv run python scripts/server_preflight.py \
  --train-artifact artifacts/gate0b-16k-derived-a/train.jsonl \
  --validation-artifact artifacts/gate0b-16k-derived-a/validation.jsonl \
  --data-manifest artifacts/gate0b-16k-derived-a/data_manifest.json \
  --bundle-manifest evidence/server/server_bundle_manifest.json \
  --minimum-free-gib 50 \
  --output evidence/server/formal-s1-server-preflight.json
```

`passed` 必须为 `true`。不要用 `--allow-missing-*` 绕过正式训练或同机评测所需依赖。

## 6. 重新生成正式 S1 dry-run 身份

正式 output 目录必须是新的空目录。先选择一个从未使用过的独立 preflight 目录执行数据和
合同 dry-run，不要删除或覆盖旧证据：

```bash
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output-dir runs/preflight/formal-s1-dry-run-<RUN-ID> \
  --dry-run
```

人工核对 `run_manifest.json`：

- `record_count = 61224`
- `max_steps = 479`
- `warmup_steps = 15`
- model/tokenizer revision 正确
- train/validation SHA-256 正确
- runtime tree、config、`uv.lock` 和 sample order hash 非空
- status 为 `dry_run`

把这个 manifest 连同 server preflight 一起复制到本次正式 run 的 evidence 目录。dry-run
不加载模型、不构建 GPU 专用 `SFTConfig`，因此它不能替代下一步 GPU 预飞行。

## 7. GPU 容量与 allocator telemetry 预飞行

### 7.1 最长保留样本

探针必须使用正式 16K 配置，而不是历史 22,295-token 配置：

```bash
uv run python scripts/probe_longest_training_sample.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output evidence/server/formal-s1-longest-sample.json
```

验收条件：最长保留序列为 16,384；forward、backward、gradient clipping 和 AdamW 第一次
step 全部完成；没有 OOM、NaN 或 Inf。

### 7.2 使用正式入口跑一个 optimizer update

使用新的、一次性目录，绝不能写入正式 S1 目录，也不要删除旧 probe：

```bash
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output-dir runs/preflight/formal-s1-telemetry-step1-<RUN-ID> \
  --stop-after-steps 1
```

检查：

```text
run_manifest.json
attempt-*.json
cuda_memory.jsonl
cuda_memory_summary.json
trainer/checkpoint-1 或等价 checkpoint
```

关键判断：

| 现象 | 解释与动作 |
| --- | --- |
| `allocated` 随 micro-batch 上下变化，`reserved` 高位稳定 | 通常是 activation 变化与缓存池复用 |
| `allocated` 每个 micro-batch 固定增长且不回落 | 怀疑保存了带梯度 tensor 或计算图，停止启动 |
| 第一个 `optimizer.step()` 突然 OOM | AdamW `m/v` 延迟分配暴露容量不足，停止启动 |
| `reserved` 接近 80GB，但 `allocated` 明显较低且 step 完成 | 不能仅凭 `nvidia-smi` 判为泄漏；继续分析 allocator 记录 |
| telemetry 缺 phase、step 或峰值字段 | 证据不合格，即使训练没报错也不能关闭门禁 |

预飞行后加载一次 step-1 checkpoint，并由生成检查显式设置 `use_cache=True` 做短生成。
训练时 `use_cache=False` 是为 gradient checkpointing；正式训练完成后的 `final_model` 也
必须通过相同加载和生成检查。

## 8. 配对评测入口预飞行

预飞行的目标不是得到有意义的 benchmark 分数，而是证明同一入口能对 B0 和训练后形态的
checkpoint 都写出终态记录。两边必须使用相同 `--max-samples`。

```bash
MODEL=/path/to/pinned/qwen3-0.6b-base-snapshot
PILOT_CHECKPOINT=/path/to/a/loadable/pilot-or-step1-checkpoint

for NAME in b0 checkpoint; do
  if [ "$NAME" = b0 ]; then TARGET="$MODEL"; else TARGET="$PILOT_CHECKPOINT"; fi
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  HF_HOME=/workspace/hf-cache \
  uv run python scripts/run_frozen_eval.py \
    --suite math500 \
    --model "$TARGET" \
    --max-samples 4 \
    --output-dir "runs/preflight/math500-$NAME"
done
```

不要全局设置 `HF_HUB_OFFLINE=1`。此前 LightEval 需要联网把 `ai2_arc` 等旧式短名解析到
数据集仓库；数据 payload 可以来自缓存，但名称解析仍可能访问 Hub。

验收条件：

- B0 和 checkpoint 都没有 prompt 被截断成零可生成长度。
- vLLM 成功分配 KV cache，没有 fork/CUDA 或 `-lcuda` 链接错误。
- `invocation_manifest.json` 从 `running` 进入明确的成功或失败终态。
- 失败时保存命令、return code 和 traceback，不留下无法解释的永久 `running`。
- 两个预飞行的任务、prompt、采样和 scorer 合同一致。

## 9. GO / NO-GO 决策

在 `evidence/server/formal-s1-go-decision.md` 中逐项写明：

```text
source/data/config identity: PASS / FAIL
server preflight: PASS / FAIL
longest sample: PASS / FAIL
first AdamW step: PASS / FAIL
allocator telemetry: PASS / FAIL
checkpoint reload/export: PASS / FAIL
paired MATH entry: PASS / FAIL
disk/time budget: PASS / FAIL
decision: GO / NO-GO
reviewer and timestamp
```

任何一项失败都是 `NO-GO`。可以修复后重新做该门禁，但不能在原结果上口头豁免。

## 10. 正式启动

正式 S1 必须从冻结 Base 开始，使用新的空目录：

```bash
test ! -e runs/gate0b-16k-s1 || test -z "$(ls -A runs/gate0b-16k-s1 2>/dev/null)"

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/workspace/hf-cache \
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output-dir runs/gate0b-16k-s1
```

运行方式必须由云平台实际验证。历史上后台进程是否在 SSH 断开后继续运行并不稳定，因此
不要只看到 `nohup` 或 `setsid` 启动成功就离开。优先使用平台支持的持久终端；至少先用
短任务断开并重连，证明进程仍在运行、日志继续增长、退出码可以落盘。

启动包装应额外记录：

```text
PID
start/end UTC time
完整命令和环境变量白名单
stdout/stderr 路径
return code
每分钟 nvidia-smi CSV
Git commit、config 和 artifact hash
```

不要把密钥、SSH 密码或访问 token 写入 manifest 和 Git。

## 11. 训练中观察

每个 optimizer step 至少查看：

- `global_step` 是否单调增加。
- loss、gradient norm 和 learning rate 是否为有限值。
- scheduler 是否只随 optimizer step 更新，而不是每个 micro-batch 更新。
- sampler/sample order 是否保持冻结身份。
- allocator 的 allocated、reserved 和 peak 是否可以解释。
- checkpoint 是否按第 50、100 等 step 保存并能加载。
- 磁盘余量和日志是否持续增长。

硬停止条件：

```text
NaN / Inf
数据、配置、代码或 sample-order hash 改变
训练从 pilot checkpoint 启动
allocator allocated 持续无界增长
checkpoint 无法加载或状态不完整
输出目录混入另一 run 的文件
磁盘将耗尽且不能安全保存 checkpoint
```

loss 短期抖动、`reserved` 长期高位、GPU 利用率波动和 checkpoint 保存时的短暂停顿，本身
不是停止理由。先依据 telemetry、CPU 数据阶段和日志定位具体阶段。

## 12. 中断与恢复

恢复必须使用同一 output 目录中的 TRL 原生 checkpoint：

```bash
uv run python scripts/train_sft_trl.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --output-dir runs/gate0b-16k-s1 \
  --resume-from-checkpoint runs/gate0b-16k-s1/trainer/checkpoint-<STEP>
```

runner 会比较恢复前后的模型、数据、配置、runtime tree、`uv.lock` 和 sample order identity。
不一致时应拒绝恢复。恢复后检查：

- `global_step` 从 checkpoint 继续，不回到零。
- optimizer 的 `m/v`、scheduler、RNG 和 Trainer state 被加载。
- learning rate 连续，没有重新 warmup。
- 已提交的 sample 不会从 epoch 开头再训练。
- 新 attempt 记录与原 run manifest 都存在。

如果为修复 bug 必须修改运行时代码，则原 checkpoint 和新代码不再天然属于同一实验。
需要单独评估兼容性并记录为新 attempt，不能静默继续。

## 13. 训练结束后的最低收尾

训练完成的最低条件是：

```text
run_manifest.status = completed
global_step = 479
final_model 存在并可加载
cuda_memory_summary.json 存在
trainer state 和最后 checkpoint 可读
进程 exit code = 0
```

先运行生成健康检查和全量 validation NLL：

```bash
S1_CHECKPOINT=runs/gate0b-16k-s1/final_model

uv run python scripts/verify_checkpoint_generation.py \
  --checkpoint "$S1_CHECKPOINT" \
  --prompts configs/gate0b/smoke_prompts.jsonl \
  --output-dir runs/gate0b-16k-s1/evals/generation-health \
  --minimum-eos-rate 0.0

uv run python scripts/evaluate_validation_nll.py \
  --config configs/gate0b/sft_s1_16k.yaml \
  --model "$S1_CHECKPOINT" \
  --output-dir runs/gate0b-16k-s1/evals/validation-nll
```

然后严格重放 B0 的 MATH-500、GSM8K 和 regression 合同。完成后做配对比较：

```bash
uv run python scripts/compare_validation_nll.py \
  --baseline-records runs/gate0b-16k/b0/validation-nll/records.jsonl \
  --trained-records runs/gate0b-16k-s1/evals/validation-nll/records.jsonl \
  --output runs/gate0b-16k-s1/evals/b0-vs-s1-validation-nll.json

uv run python scripts/compare_lighteval_results.py \
  --baseline-root runs/gate0b-16k/b0/evals \
  --trained-root runs/gate0b-16k-s1/evals \
  --output-dir runs/gate0b-16k-s1/evals/b0-vs-s1-lighteval
```

路径需按服务器实际归档位置核对，不能为了让命令运行而指向不同合同的 pilot 结果。

## 14. 必须保留的交付物

```text
GO/NO-GO decision
source/config/data/sample-order identity
server preflight
longest-sample report
allocator telemetry JSONL and summary
stdout/stderr and exit marker
run manifest and attempt manifests
resolved config
checkpoints and final_model metadata
generation health
full validation NLL records
MATH-500/GSM8K/regression manifests and per-sample outputs
B0/S1 paired comparisons
incident notes
final experiment report
```

大模型权重和大体积逐样本输出可以保存在数据盘或对象存储，Git 中保存 manifest、摘要、
hash 和稳定路径。只保留一个聚合分数，不能支持恢复、配对分析和错误定位。

## 15. 独立执行检查

下次不依赖他人执行时，应能够回答：

1. 我现在启动的是预飞行、pilot 还是正式 S1？证据字段在哪里？
2. 这个 run 的模型、数据、代码、配置和 sample order 如何唯一标识？
3. 第一个 AdamW step 是否发生过，它对显存做了什么？
4. 如果 SSH 断开，怎样证明进程仍在运行或已经失败？
5. 从哪个 checkpoint 恢复，为什么不会重训前面的 sample？
6. S1 的每个分数与哪个 B0 结果同合同配对？
7. 哪些结论由 NLL 支持，哪些必须由任务评测和错误分析支持？

这些问题都有落盘证据后，正式 S1 才是一项可信工程，而不只是一次成功运行的命令。
