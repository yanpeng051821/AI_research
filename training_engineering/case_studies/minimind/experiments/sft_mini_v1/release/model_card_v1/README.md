---
language:
  - zh
pipeline_tag: text-generation
base_model: yanpeng051821/minimind-dense-64m-mini-pretrain
base_model_relation: finetune
datasets:
  - jingyaogong/minimind_dataset
tags:
  - minimind
  - sft
  - full-finetuning
  - research-checkpoint
  - custom-runtime
---

**English** | [简体中文](README_zh.md)

# MiniMind Dense 64M: One-Epoch Mini SFT

> Local model-card draft, updated 2026-11-01. This SFT model has not been uploaded to a new Hub repository. It is an educational research checkpoint, not a reliable assistant. Validation NLL and some stopping behavior improved, but task completion did not improve over its own pretrained baseline on the fixed diagnostic set. Scores are assistant-authored initial judgments, with one unresolved case; no independent review is claimed.

## 1. Model Overview

This is a full-parameter supervised fine-tune of [our own MiniMind mini pretrained model](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain), trained and operated by yanpeng with AI assistance. Its pretrained weights were trained from random initialization, not initialized from official MiniMind model weights. The architecture and tokenizer reference MiniMind; this experiment does not reproduce the complete official model or inherit its reported capabilities.

The experiment asks whether assistant-only supervision can turn the existing continuation model into a better assistant under fixed conditions. It completed the engineering workflow, but **did not meet the task-improvement objective**. This card preserves that negative result alongside measurable partial changes.

| Item | Actual Model |
| --- | --- |
| Type | Dense decoder-only causal language model |
| Unique parameters | 63,912,192, approximately 63.91M |
| Architecture | 8 layers; hidden size 768; FFN intermediate size 2,432 |
| Attention | 8 query heads, 4 KV heads; Q/K RMSNorm; RoPE base 1,000,000 |
| Tokenizer | MiniMind, vocabulary 6,400; BOS=1, EOS=2, PAD=0 |
| Tied weights | Embedding and LM head share one Parameter |
| SFT initialization | Our own pretraining final step 4,943; fresh SFT AdamW/scheduler |
| Stages completed | Our mini pretraining, then this one-epoch SFT; no DPO or RL |
| Evaluated S1 | Final SFT step 49,087; not selected by comparing checkpoint scores |
| Export | Native FP32 safetensors; tensor file 255,659,304 bytes, approximately 255.66 MB |
| Interface | Local Python runtime; not a Transformers AutoModel adapter |
| Context scope | Trained and evaluated within 768 tokens; longer-context quality untested |

The upstream architecture/tokenizer reference is [MiniMind commit f659b557](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f). The pretraining architecture reference and SFT training-recipe choices are provenance, not claims of an exact official end-to-end reproduction.

## 2. Data and Supervision

### 2.1 Data Version

| Item | Actual Data |
| --- | --- |
| Upstream source | [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5) |
| Pinned revision | 312afb4f76391145c6902f765bb51691c09a12f5 |
| Source file | sft_t2t_mini.jsonl, 905,718 records |
| Accepted conversations | 787,425 |
| Train / validation | 785,377 / 2,048 |
| Limited revisions | 14 identity introductions and 2 specific content corrections |
| Final preparation holds | 21,073; some are unresolved contexts, not proven errors |
| Post-revision duplicates | 15 set aside |
| Train / validation supervised targets | 326,249,045 / 833,052, counted after the causal shift |

Preparation checked structure, declared-tool protocols and parameter schemas, real template encoding, length and valid targets. Earlier stages excluded overlength conversations, exact duplicates, two confirmed bad records, and undeclared calls. The final preparation applied bounded revisions/holds, re-encoded, deduplicated and split; these counts refer to different stages and should not be subtracted independently from the original total.

Split groups are connected components sharing any normalized user message or normalized full conversation. Normalization uses NFKC, case folding and collapsed whitespace for grouping only. Seed 20261031 selects whole groups within the validation budget; development/review examples are protected on the training side. Exact dialogue and defined normalized-question overlap across splits was zero. This is not semantic near-duplicate or external benchmark contamination clearance, and the validation set is not an independent blind test.

### 2.2 Encoding and Identity Policy

Complete encoded conversations longer than 768 are excluded, not truncated. The template is rendered once and tokenized without adding special tokens again. Right padding reaches 768. Assistant bodies, retained thinking, call text and the complete message-ending sequence are supervised; role headers, user/system text, tool results and padding are not targets at those positions. Context is still visible to the model; labels shift only in the causal loss.

The 14 bounded identity revisions identify yanpeng as the trainer and credit MiniMind's architecture/project; they do not globally replace people or project names. Their obsolete reasoning is cleared. This is a data policy, **not evidence that the model reliably recalls its identity**; the post-hoc identity probes below failed.

Content truth was reviewed only in bounded samples. Anonymous bundled tool results have limited structural correspondence guarantees. Semantic overlap, comprehensive factual quality, safety, personal-information leakage and full source-rights review were not cleared. Raw and derived training JSONL files are not part of the intended inference release.

## 3. Training Recipe and Cost

| Item | Actual Configuration |
| --- | --- |
| Hardware | One NVIDIA RTX 4060 Ti, 8 GB VRAM |
| Training system | Project's own PyTorch loop; not TRL Trainer |
| Parameter / compute precision | FP32 parameter state; CUDA BF16 autocast |
| Updated parameters | All unique trainable parameters, one AdamW group |
| Micro-batch / accumulation | 4 / 4; usually 16 conversations per update |
| Loss | Causal cross-entropy; loss sum divided by all valid assistant targets in the current accumulation window |
| AdamW | lr=1e-5; betas=(0.9,0.999); eps=1e-8; weight decay=0.01; foreach/fused=False |
| Schedule | No warmup; optimizer-update-indexed cosine decay to 1e-6 |
| Clipping | Global gradient norm limit 1.0 after token normalization; logged norm is pre-clip |
| Seeds / data workers | seed=42, sampler_seed=42; num_workers=0 |
| Training amount | One epoch, 49,087 updates and 196,345 micro-batches |
| Final update | One conversation, one micro-batch, 413 valid targets; not dropped |
| Validation | Full 2,048 records initially, every 5,000 updates and finally; BF16 monitoring |
| Checkpoints | Every 1,000 updates and completion; 50 full recovery checkpoints |
| Stop policy | 14-hour cumulative active-run budget with boundary save/pause; not reached |

Training ran from 2026-10-31 17:43:07 to 2026-11-01 04:26:24, Asia/Shanghai (UTC+08:00), and exited successfully. Wall time was approximately 10 hours 43 minutes 17 seconds: update windows approximately 10h12m24s, checkpoint saving/receipts 22m46s, validation 6m53s, and other startup/orchestration overhead approximately 1m13s. No extra rental or paid evaluator was used for this run; electricity and monetary operating cost were not measured.

The largest logged update peak allocated memory was approximately 3.77 GiB; post-update allocated memory was approximately 0.789 GiB and reserved memory ranged approximately 3.37–3.88 GiB. These are sampled allocator readings, not continuous whole-device measurements.

The final code passed a same-device continuous-20 versus 10-plus-resume-to-20 GPU gate under the full schedule. The final run's update/sample/target totals, tail, scheduler, AdamW state, sampler order, finite tensors and final checkpoint bytes were audited. All 50 checkpoint/receipt pairs were inventoried; not every historical checkpoint was fully hashed and loaded. This is not a guarantee of cross-device bitwise reproducibility.

## 4. Evaluation and Findings

### 4.1 Fixed Paired Diagnostic

B0 is our own pretrained model; S1 is the final SFT model. The 32 core cases cover everyday questions, instruction constraints, supplied material and fixed multi-turn contexts, eight each. Four tool cases are supplementary and do not execute real services.

Both use identical actual prompt text/token IDs, system, fixed histories, tools, tokenizer, template and an empty prefilled think region. Generation is CUDA FP32, eval/inference mode, greedy, batch 1, at most 256 new tokens, total context budget 768, EOS=2; TF32/autocast are disabled. NLL uses all 2,048 validation records, batch 4, and 833,052 assistant targets. Scoring is the original per-case checklist plus the same pre-SFT-approved V2 location-only patch.

| Metric | B0 | S1 |
| --- | ---: | ---: |
| Full FP32 validation NLL | 2.3954561617 | 1.7210819774 |
| Core task judgments | 2 pass, 30 fail | 0 confirmed pass, 31 fail, 1 uncertain |
| Confirmed core pass rate | 6.25% | 0%; upper bound 3.125% if the uncertain case passes |
| Supplementary tool judgments | 0/4 pass | 0/4 pass |
| EOS / length-limit stops | 0 / 36 | 17 / 19 |
| Mean / median new tokens | 256 / 256 | 155.19 / 256 |

NLL decreased by approximately 28.15%. It measures prediction of real assistant targets under supplied context, not whole-answer correctness in free generation. These are SFT validation numbers; they are not comparable to the different validation set and full-text objective in the pretraining model card.

Core pairing contains 30 both-nonpass cases, one confirmed regression, and one uncertain pair, with no confirmed positive task flip. The uncertain case copies the correct key-location context but does not clearly answer the final location; it remains unresolved. Even counting it as a pass does not exceed B0's two passes.

All semantic judgments are assistant-authored initial reviews, not independently or exhaustively reviewed by the user. Hash, coverage, quote and aggregation checks do not certify semantic correctness. This small, already inspected diagnostic set is not a public benchmark or blind test.

### 4.2 Partial Changes and Failures

A location prompt requests only the location from a supplied record. S1 outputs:

```text
青禾图书角周三开放，开始时间14:20，结束时间17:10，地点是二楼东侧。
```

It preserves the given times and stops with EOS, whereas B0 repeated text with an incorrect start time. It still fails the location-only instruction because it adds dates/times. This is a partial change, not a task pass.

For a radio-repair summary, S1 preserves the person, battery replacement and working-again outcome, but copies the two-sentence source instead of producing the requested one-sentence summary. Some other outputs merely echo the question and then end.

Tool-related risks remain: using lookup_book for weather, malformed call JSON, and inventing humidity/wind absent from a supplied weather result. No tool was actually executed. Reduced repetition or more EOS stops must not be read as reliable instruction following, factuality or safe tool use.

### 4.3 Post-Hoc Identity Diagnostic

After the main evaluation, three additional prompts ask who the model is, who trained it, and whether pretraining began randomly or continued official weights. Under the same S1 FP32 greedy settings, all three reach the 256-token limit without a correct source attribution; none identifies yanpeng/MiniMind as intended. One claims that it has no actual training data. The third question supplies two alternatives and is not an unprompted-recall test.

These are post-hoc diagnostics, not added to the 36 main cases, not blind, and not paired against B0. Trainer/source facts in this card come from experiment records, not the model's self-description.

## 5. Local Loading

The native S1 bundle and this card's chat helper are separate local artifacts at this draft stage; a final upload payload has not been assembled. The intended package contains model.safetensors, config.json, tokenizer/, runtime/, load_model.py, requirements.txt, the two cards and [chat.py](chat.py).

After these files are assembled together and dependencies installed, run from the package directory:

```shell
python -I -B chat.py --prompt "你好，请简单介绍一下你自己。" --max-new-tokens 256
```

The example is a way to inspect behavior, **not a promise of a correct introduction**. chat.py defaults to CPU FP32, uses the same neutral Chinese assistant system as evaluation, renders the packaged chat template with add_generation_prompt=True and open_thinking=False, and tokenizes with add_special_tokens=False. It generates one reply, executes no tools, performs no sampling/retries, and rejects prompt-plus-budget beyond 768 without silent truncation. CPU performance/outputs are not asserted bitwise identical to the CUDA evaluation.

The original load_model.py --prompt interface remains a raw continuation/loading check; it does not add the SFT chat template. Do not substitute AutoModel.from_pretrained or assume automatic support by Hub inference providers.

Recorded dependencies are PyTorch 2.14.0, Transformers 4.57.6, tokenizers 0.22.2 and safetensors 0.8.0. Select the appropriate PyTorch build for the platform; only the existing local environment has been tested, not a clean installation elsewhere. Review the included custom Python before execution; file hashes are not signatures or a code sandbox.

## 6. Export and Reproduction

The final checkpoint was strictly loaded into the matching architecture and exported without changing its FP32 logical parameters. A fresh offline CPU process reconstructed the model with the package's runtime/config/tokenizer. The full state digest and tied Parameter matched; inputs, masks, logits for two short prefixes and a short greedy continuation were exactly equal, with zero maximum logit difference. This is a bounded export check, not all-input or cross-device equivalence.

| Identity | SHA256 |
| --- | --- |
| Final SFT checkpoint | 181ce5bbc64c0c6ad0f3b79682f71df33206665c8212182422ce1363fec1e03a |
| model.safetensors | 4fdc2b4d4f2bbc1f85ca0a7a68eb615fea96cfaf3c7ddcf9cb9c8e03dcb995b5 |
| Logical model-state digest | 733c18c9775eadd7f67330fabb0568ab61103f9d5fba995e6d967405c904cc62 |
| Initial B0 state digest | 8d10d76090b2714c055a95123c3583c6941b1ecb6af84a10f0f68dbe21cb783f |
| SFT training contract | 3ac716dee78ce33d5d418e7f1f5e7e6b176dcdad6d06c52ac608322172c1abe9 |
| SFT data contract | 55bea98df2af3f6105fc03e06711b2602ba38e678b412720dcfff928e9f1da4d |

The source project is [AI_research](https://github.com/yanpeng051821/AI_research), with this experiment under training_engineering/case_studies/minimind/experiments/sft_mini_v1. This draft has not been pushed: exact accessible Git/HF revisions, release-file inventory and post-download verification must be recorded before publication. No nonexistent S1 Hub repository or unpublished commit is presented as accessible.

Changing documentation or adding chat.py requires a new release manifest; the evaluated native bundle and its existing manifest remain unchanged. Full recovery checkpoints retain optimizer/scheduler/sampler/RNG state locally and are not replaced by this inference package.

## 7. Intended Use and Limitations

This checkpoint is useful for learning, inspecting training effects and reproducing a bounded negative-result experiment. It is not a production assistant, trusted knowledge source, verified identity responder or autonomous tool agent. Strong repetition, question/source copying, missed constraints, unsupported claims and tool misuse were observed.

Long-context quality beyond 768, English proficiency, systematic safety/bias, privacy leakage and memorization have not been established. The language of this English card is not evidence of English generation ability. No extra training, checkpoint selection, changed decoding or relaxed rubric was used to manufacture an improvement after seeing the result.

Core components were implemented by the learner with guidance; delegated integration, audit, operations, evaluation, export and documentation were assisted by an AI coding agent. The project is not represented as entirely independently implemented by the learner.

## 8. Sources and License Status

MiniMind's pinned [source license](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE) and tokenizer provenance are credited separately from this model's weights. The [pinned dataset card](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md) declares Apache-2.0 and CC-BY-NC-2.0; the existing review has not established a complete per-file source-license mapping or the full weight-redistribution scope.

No blanket model-weight/package license or unconditional commercial rights are asserted, and no license tag has been selected automatically. Research or noncommercial labeling alone does not resolve missing permissions. Applicable notices, public redistribution scope and original-contribution terms remain release-review items; this card records the uncertainty rather than giving legal advice.

**Release status:** local draft only; no S1 Hub upload, pinned-download verification or public-release approval is claimed.
