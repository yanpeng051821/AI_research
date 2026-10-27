---
pipeline_tag: text-generation
datasets:
  - jingyaogong/minimind_dataset
tags:
  - minimind
  - pretraining
  - from-scratch
  - research-checkpoint
  - custom-runtime
---

**English** | [简体中文](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain/blob/main/README_zh.md)

# MiniMind Dense 64M: One-Epoch Mini Pretraining

> Public research checkpoint. This card documents the actual training run and native loading interface. It does not claim full reproduction of the official model, reliable chat capabilities, or cleared commercial rights. No blanket model-weight license has been selected. Exact release revisions and verification results are recorded in the linked GitHub release records.

## 1. Model Overview

This model comes from a small-scale pretraining experiment starting from random initialization. The learner implemented model and training components with guidance, aligned the architecture with MiniMind-3 Dense, and trained for one epoch on the official mini corpus. The objective was to complete a traceable training, recovery, evaluation, and delivery workflow and check whether NLL decreased on a fixed, in-distribution validation set, not to reproduce all capabilities of the official released model.

| Item | This Model |
| --- | --- |
| Type | Dense decoder-only autoregressive language model |
| Unique parameters | 63,912,192, approximately 63.91M |
| Architecture | 8 layers, hidden size 768, FFN intermediate size 2,432 |
| Attention | 8 query heads, 4 KV heads, Q/K RMSNorm, RoPE base 1,000,000 |
| Vocabulary | 6,400 tokens, using the MiniMind tokenizer |
| Weight tying | Embedding and LM head share one Parameter |
| Starting point | Random initialization with seed 42; no official pretrained weights were loaded |
| Training stages completed | This pretraining run only; this model has not undergone SFT, DPO, or RL |
| Export | Native FP32 safetensors package, approximately 256.20 MB |
| Loading interface | Included load_model.py; not a Transformers AutoModel adapter |

The architecture and tokenizer reference is pinned to [MiniMind commit f659b557](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f). Referencing it does not imply that this model inherits the official weights or benchmark scores.

## 2. Training Data

| Item | Actual Version or Policy |
| --- | --- |
| Dataset | [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5) |
| Pinned revision | 312afb4f76391145c6902f765bb51691c09a12f5 |
| Source file | pretrain_t2t_mini.jsonl, 1,270,238 records |
| Training split | 1,265,235 records |
| Fixed validation split | 5,000 records; not an independent final test set |
| Excluded records | 3; manual constraints apply to entire groups with identical text |
| Split policy | Group by exact original-text hash; sample validation groups with seed 43; keep duplicate texts within the same split |
| Encoding | BOS + at most 766 body tokens + EOS; right-pad to 768 |
| Supervision | Body and EOS targets after the causal shift; PAD does not contribute to loss |

An exact comparison of complete encoded input sequences found no train/validation overlap. Approximately 3.92% of training records were truncated, removing approximately 8.55% of the original body tokens; these percentages have different denominators. Training consumed 301,810,690 valid prediction targets. The fixed validation set contains 1,206,758 valid targets.

The project did not complete exhaustive fact-checking, near-duplicate removal, original-document-level isolation, or comprehensive personal-information and copyright review. Exact non-overlap does not rule out semantic overlap or contamination. Raw and derived JSONL files are not redistributed with this model package.

The full data rules and identities are in the [pinned data contract](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/data_recipe/DATA_CONTRACT.md). The objective remains full-text pretraining; question-and-answer text in the corpus does not make this assistant-only SFT.

## 3. Training Recipe

| Item | Actual Configuration |
| --- | --- |
| Hardware | One RTX 4060 Ti with 8 GB VRAM |
| State and computation | FP32 parameter state, CUDA BF16 autocast, deterministic algorithms, TF32 disabled |
| Updated parameters | All unique trainable parameters, in one AdamW parameter group |
| Micro-batch / accumulation | 4 / 64; 256 records per full update window |
| Loss normalization | loss_sum divided by valid causal targets across the entire current accumulation window |
| Optimizer | AdamW, lr=5e-4, betas=(0.9,0.999), eps=1e-8, weight decay=0.01 |
| Learning-rate schedule | No warmup; optimizer-update-indexed cosine decay to 5e-5 |
| Gradient clipping | Global norm limit 1.0 |
| Budget | One epoch, 4,943 updates; final window: 83 records in 21 micro-batches |
| Validation | At initialization, every 500 updates, and at completion; all 5,000 validation records |
| Checkpointing | Every 100 updates and at completion; 50 full recovery checkpoints |

Training started on 2026-10-25 at 10:43:09 and finished on 2026-10-26 at 02:56:35, both in Asia/Shanghai (UTC+08:00). Wall time was approximately 16 hours, 13 minutes, and 26 seconds. The formal run exited normally and did not start from the intermediate weights used in the recovery probe.

Before launch, a four-update continuous run was compared against a two-plus-two resumed run under the full formal schedule. A separate short run reached update 20. The checks passed under the tested conditions, not as a guarantee of bitwise recovery across devices.

See the [pinned training contract](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/training_recipe/TRAINING_CONTRACT.md) and [executable configuration](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/training_recipe/training_config.json). This Git commit contains the verified core runtime, recipes, and export/publication tools. It is not asserted to have been the Git HEAD at training launch.

## 4. Results and Cost

### 4.1 Fixed Validation NLL

| Metric | Random Initialization | Final Model |
| --- | ---: | ---: |
| Token-weighted NLL on the full fixed validation set | 8.912246 | 1.998545 |

NLL decreased at all 11 measured validation points, with smaller improvements later in training. Per-update training loss uses changing batches and parameter states; its difference from fixed-checkpoint validation NLL is not interpreted directly as a generalization gap.

These results support improved text prediction on this fixed distribution. They do not establish improvements in chat, factual accuracy, mathematics, code, tool use, or safety. No independent final test or downstream benchmark scores are reported.

### 4.2 Measured Cost

Training windows accounted for approximately 96.01% of this run's wall time, checkpoint saving for 2.33%, and validation for 1.64%. End-to-end throughput was approximately 5,167 valid targets per second. The largest logged update peak allocated memory was approximately 3.77 GiB; update-end reserved memory ranged from approximately 3.37 to 3.88 GiB. These are not continuous memory measurements covering every phase. Integrated electricity usage and billing were not collected, so no monetary cost is reported.

### 4.3 Manual Generation Spot Checks

On 2026-10-27, the learner personally ran load_model.py from the native FP32 v2 package and supplied the following terminal outputs. Conditions: CPU FP32, eval mode, gradients disabled, greedy decoding, and at most 32 new tokens. Inputs consist of BOS followed by a text prefix, without a conversation template.

Chinese prompts and continuations are reproduced verbatim below, including incomplete endings. They have not been translated into supposed model-generated English. The language of this card is not evidence of English generation performance.

**Scene continuation**

Prompt: `春天的风吹过田野，`

```text
树叶在微风中轻轻摇曳。小草在微风中轻轻摇曳，阳光照耀着大地，微
```

The continuation is relevant to a spring scene and broadly fluent, but repeats a phrase about swaying in the breeze and is somewhat formulaic.

**Greeting continuation**

Prompt: `你好，`

```text
我有一个问题想问。您好，欢迎来到这里。请问您有什么问题需要帮助吗？我最近在学习编程，但是不知道从
```

The output contains greetings and question/answer-like text, but switches between questioner and responder rather than maintaining a stable assistant role. This is not recorded as reliable conversational ability.

**Concept continuation**

Prompt: `机器学习是什么，`

```text
它如何工作？机器学习是一种人工智能的分支，它使用算法和统计模型来让计算机从数据中学习，并根据这些数据做出决策。
```

The model completes the question before providing a high-level explanation, consistent with full-text continuation. This explanation is broadly reasonable, but one example is not a systematic factual evaluation.

These are three ad hoc manual observations, not a pre-registered independent test, and no accuracy is calculated. They show some short Chinese continuation ability alongside repetition and unstable speaker boundaries; they do not prove reliable chat, instruction following, or broad knowledge generalization. Similarity to training texts was not checked. Actual generated token counts and stopping causes were not logged. The first two incomplete endings may reflect the 32-token cap; EOS termination or an inability to end cannot be inferred from them.

## 5. Loading and Use

Download the entire package from a specific revision of [this repository](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain), then run from its directory. Verified release revisions are listed in the [GitHub release records](https://github.com/yanpeng051821/AI_research/tree/main/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/release). Downloading only the weights is insufficient because the custom runtime is required.

```shell
python -I -B load_model.py --prompt "春天的风吹过田野，" --max-new-tokens 8
```

The included requirements.txt records PyTorch 2.14.0, Transformers 4.57.6, tokenizers 0.22.2, and safetensors 0.8.0. Select an appropriate PyTorch build for your platform. Loading was tested in the existing local dependency environment with CPU FP32, not through a clean installation on another machine.

The loader constructs the model from the included configuration, loads model.safetensors, and uses the included tokenizer. It defaults to CPU FP32, eval mode, and disabled gradients. This is a pretrained continuation interface, not an SFT chat interface. It conservatively limits prompt tokens plus the generation budget to the trained length of 768. This is not an absolute architectural context limit; longer-context quality has not been evaluated.

The package contains custom Python runtime code that should be reviewed before execution. AutoModel.from_pretrained is not a replacement for this entry point, and automatic support by HF online inference services is not claimed.

## 6. Export Verification and Identities

Model A, loaded directly from the final checkpoint using the project runtime, was compared with model B, reconstructed in a fresh process from the package's runtime, configuration, and safetensors weights. Both used identical CPU FP32 conditions. Full model-state digests and embedding/LM-head Parameter sharing matched. Tokens, masks, and logits for two short right-padded prefixes, plus a short greedy continuation, matched exactly. Maximum absolute logit difference was zero. A standalone UTF-8 continuation command also passed.

This checks export correctness under the tested conditions, not downstream ability, CUDA/BF16 equivalence, or, by itself, post-upload download integrity. The original public release separately passed a pinned-revision download and loading check. Native export v2 fixes Windows standard-output encoding without changing logical parameter values.

| Identity | SHA256 |
| --- | --- |
| Final source checkpoint | 8d3cb52867e0c74b35c5931bfb9c90de3a7b6589fd9466fd68d00c9593431e38 |
| model.safetensors | 3f1d1e6f585d516149d62d33670ef20cb29caa11c1759786586879e19fa3cc22 |
| Internal model-state digest | 8d10d76090b2714c055a95123c3583c6941b1ecb6af84a10f0f68dbe21cb783f |
| Core runtime contract | b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58 |

File hashes identify serialized bytes; the internal state digest includes tensor names, dtypes, shapes, and values. Documentation or packaging updates produce a new file manifest, rather than reusing an old manifest for modified content.

## 7. Sources and Licensing

The pinned MiniMind source commit includes an [Apache-2.0 license](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE). The tokenizer comes from model/tokenizer.json and model/tokenizer_config.json in that commit. Parsed JSON content matches upstream; local file bytes differ only in line endings. Training and export identities use the actual file hashes recorded by this project's encoding audit.

The [pinned dataset card](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md) lists both Apache-2.0 and CC-BY-NC-2.0 and describes multiple data sources. The reviewed materials do not establish an explicit mapping of every source license to the mini file used here. This review has not established the full scope of redistribution rights for the trained weights.

No blanket package or model-weight license is selected, no unconditional commercial-use rights are claimed, and public availability does not grant unspecified additional rights. The package includes the upstream code license and attribution, not a declaration that this is the model-weight license. The full licensing scope of original contributions and model weights remains unresolved. Research, noncommercial, or private labels do not themselves supply missing permissions. This records provenance and risks, not legal advice.

## 8. Contributions and Limitations

The learner implemented core model components and early data/training components with guidance. The assistant performed explicitly delegated integration, repeated tests, data/recipe checks, run operations, analysis, export, and documentation. The whole project is not represented as independently implemented by the learner.

This model is intended for learning and controlled research, not as a reliable source of facts or a high-stakes decision system. Systematic safety, bias, training-text memorization, and personal-information leakage evaluations have not been completed. The inference package omits optimizer, scheduler, sampler, and RNG state and cannot replace a full recovery checkpoint.

## 9. Release Files

recipes/ contains the actual training configuration, core contract, and a public data-recipe summary without local paths. evidence/ contains validation NLL history, cost statistics, learning curves, original manual observations, and local export checks. The fixed GitHub links above provide full source code, frozen recipes, and data-preparation entry points. Upstream licensing and attribution are in licenses/ and THIRD_PARTY_NOTICES.md; they do not replace unresolved package-wide or model-weight terms.

README.md is the default English model card. [README_zh.md](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain/blob/main/README_zh.md) preserves the Chinese version; the language links navigate between the two documents and do not change model behavior.
