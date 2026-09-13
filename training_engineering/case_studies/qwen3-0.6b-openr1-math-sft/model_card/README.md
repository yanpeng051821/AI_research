---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B-Base
datasets:
- open-r1/OpenR1-Math-220k
tags:
- qwen3
- sft
- math
- research-checkpoint
---

# Qwen3-0.6B OpenR1-Math SFT S1

This repository contains a research checkpoint produced by one epoch of full-parameter supervised
fine-tuning of `Qwen/Qwen3-0.6B-Base` on a deterministically filtered OpenR1-Math artifact.

This checkpoint is published to make the training run reproducible and auditable. It is **not** a claim
of across-the-board mathematical or general-purpose improvement.

## Training

| Field | Value |
| --- | --- |
| Base revision | `311c62e88814bff7206909ccd330bab0a784743b` |
| Train records | 61,224 |
| Maximum artifact length | 16,384 tokens; longer records excluded, not truncated |
| Epochs / optimizer steps | 1 / 479 |
| Effective batch | 128 sequences (`1 x 128` gradient accumulation) |
| Learning rate | `4e-5`, cosine schedule, 3% warmup |
| Precision | BF16 compute; full-parameter training |
| Hardware | 1 x NVIDIA A100-SXM4-80GB |
| Runtime | 31,867 seconds (about 8 h 51 min) |
| Train loss | 0.568926 |

The supervised loss is completion-only causal cross-entropy with global valid-token normalization.

## Evaluation

All deltas below compare the frozen base model and S1 under paired contracts.

| Evaluation | B0 | S1 | Delta |
| --- | ---: | ---: | ---: |
| Validation NLL, token weighted | 0.780483 | 0.546614 | -29.96% relative |
| GSM8K QEM, 1,319 samples | 0.476118 | 0.514784 | +3.87 pp |
| 59-task regression macro | 0.543412 | 0.528562 | -1.49 pp |
| MMLU 57-task macro | 0.545157 | 0.530376 | -1.48 pp |
| ARC-Challenge `acc_norm` | 0.453925 | 0.431741 | -2.22 pp |
| HellaSwag `acc_norm` | 0.533459 | 0.522008 | -1.15 pp |

The checkpoint fits the target SFT distribution substantially better and improves GSM8K under the
recorded contract, while the broad regression panel declines. Further error analysis is required before
selecting a follow-up recipe.

## MATH-500 limitation

A 20-question, four-generation smoke test with a 512-token cap was strongly length-truncated: all 80 S1
generations reached the cap. The S1 pass@1:1 value was 0.05 versus 0.25 for B0, but this result mixes
answer quality with truncation and is not treated as a formal MATH-500 score.

The original 500-question, four-generation, 32K-token evaluation was stopped after 204/2000 generations
because projected paid GPU time exceeded the budget. No full MATH-500 result is reported.

## Generation and stop tokens

The saved `generation_config.json` uses token `151645` (`<|im_end|>`) as EOS, while the underlying model
`config.json` retains the base token `151643` (`<|endoftext|>`). Some inference engines do not honor the
saved generation config. For bounded generation, explicitly supply both stop strings or their token IDs
and set an application-appropriate maximum output length.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "yanpeng051821/qwen3-0.6b-openr1-math-sft-s1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "Solve: If 3x + 5 = 20, find x."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    eos_token_id=[
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    ],
)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Intended use

Use this checkpoint for research on small-model SFT, data and error analysis, generation-length behavior,
and reproducible B0/S1 comparisons. It should not be deployed as a general assistant or used where
mathematical correctness is safety critical without independent validation.

## Reproducibility evidence

The `evidence/` directory contains the resolved training configuration, run manifest, compact paired
evaluation summaries, the MATH budget-abort record, and SHA-256 checksums. Full optimizer checkpoints,
training data, and large per-sample evaluation artifacts are intentionally stored outside this model repo.

Implementation: https://github.com/yanpeng051821/small_model_post_training

Engineering case study: https://github.com/yanpeng051821/AI_research/tree/main/training_engineering/case_studies/qwen3-0.6b-openr1-math-sft
