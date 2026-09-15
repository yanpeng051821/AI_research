# 05 B0 基线与评测合同

## 1. 为什么训练前必须先跑 B0

训练后单独得到一个分数，没有办法回答分数变化来自训练、评测配置、模型 revision 还是随机
采样。B0 是正式训练前、在同一评测合同下对 Base 模型得到的可追溯结果。

本案例状态：`COMPLETE`。本篇保存已经执行并冻结的 B0，不是尚待运行的评测计划。

本案例不只评估一个指标，而是分为三层：

```text
目标分布拟合：validation assistant-only completion NLL
目标任务能力：MATH-500、GSM8K
原有能力保护：MMLU、ARC-Challenge、HellaSwag 等 regression panel
```

## 2. 冻结评测合同

执行前固定 `configs/gate0b/evaluation.yaml`，至少包含：

- task 和数据 revision；
- chat template / few-shot 模式；
- scorer；
- generation 或 log-likelihood 请求类型；
- temperature、top-p、seed；
- max model length、max new tokens；
- 每题采样数；
- vLLM memory utilization；
- 输出目录与 invocation manifest schema。

正式 S1 之后不能为了让模型跑通而只修改 S1 的评测参数。若合同确实错误，应建立新版本并
重新执行 B0。

## 3. 先 dry-run，再正式运行

```bash
uv run python scripts/run_frozen_eval.py \
  --suite <math500|gsm8k|regression> \
  --model Qwen/Qwen3-0.6B-Base \
  --model-revision 311c62e88814bff7206909ccd330bab0a784743b \
  --output-dir runs/<run-id>/b0/evals/<suite> \
  --dry-run
```

检查生成的 invocation manifest：模型、revision、task、prompt 模式、采样参数、数据 revision
均正确后，使用新的正式输出目录去掉 `--dry-run`。不要在原 dry-run 目录中混写结果。

validation NLL 单独执行：

```bash
uv run python scripts/evaluate_validation_nll.py \
  --config configs/gate0b/sft_train_16k.yaml \
  --model Qwen/Qwen3-0.6B-Base \
  --model-revision 311c62e88814bff7206909ccd330bab0a784743b \
  --output-dir runs/<run-id>/b0/validation-nll
```

## 4. 本案例 B0

| 指标 | 结果 | 样本/口径 |
| --- | ---: | --- |
| validation completion NLL | 0.7804831795 | 1,968 条，11,367,594 valid tokens |
| MATH-500 pass@1:1 | 0.432 | 500 题 |
| MATH-500 pass@1:4 | 0.449 | 500 题，每题 4 次生成 |
| GSM8K QEM | 0.4761182714 | 1,319 题 |
| MMLU average acc | 0.5451570633 | 57 学科 |
| ARC-Challenge acc_norm | 0.4539249147 | regression panel |
| HellaSwag acc_norm | 0.5334594702 | regression panel |
| regression aggregate acc | 0.5407047139 | 冻结任务集合 |

这些结果都有模型 revision、数据 revision、评测配置、运行目录和机器可读结果。历史中无法
核对的 MATH-500 约 26% 至 28% 已被降级，不再作为 B0。

## 5. 评测过程中的真实波折

### Hugging Face offline mode 不是总能直接使用

`HF_HUB_OFFLINE=1` 曾让 datasets 无法解析旧式数据集短名。即使数据似乎已缓存，评测框架
仍可能需要元数据解析。修复是预下载所需数据，并在正式调用中使用可工作的联网/缓存策略，
而不是笼统地把所有进程设为 offline。

### FlashAttention 导入改变了 vLLM 子进程行为

导入相关库后父进程可能已经初始化 CUDA，而 vLLM 默认 fork 子进程会报 CUDA 不能在 fork
后重新初始化。实际工作配方需要：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
```

这属于运行环境问题，不是模型能力问题。

### 生成式评测存在长尾

MATH-500 的 2,000 次生成耗时约 108.6 分钟。completion token 的中位数约 526，但 p95
接近 32K，约 6% 的长生成占用了大量 GPU 时间。预算不能只按平均答案长度估计。

## 6. 验收门与产物

- [x] 每个纳入正式 B0 的 suite，其 invocation manifest 为终态成功。
- [x] 模型和 dataset revision 已记录。
- [x] 保存汇总指标和逐样本输出。
- [x] 生成健康信息包含 EOS、空输出、运行错误和长度分布。
- [x] NLL 保存逐样本 loss sum、valid token count 和 sample ID。
- [x] B0 数值量级经过公开模型能力或 sanity test 检查。
- [x] 正式 B0 与任何 smoke/probe 目录明确分开。

只保存终端里的一个百分数，不算完成 B0。

## 7. 可复用边界

可复用的是“目标指标 + 回归指标 + 逐样本证据 + 同合同”的结构。具体 benchmark、few-shot、
采样参数和 scorer 必须围绕新任务重新选择，不能机械沿用数学 SFT 的评测集。
