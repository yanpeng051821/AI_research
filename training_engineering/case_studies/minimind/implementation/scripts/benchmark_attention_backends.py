"""Isolated backend probe; never edits the historical Attention implementation."""

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model

import minimind_lab.models.attention as attention
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss_sum_and_count
from minimind_lab.data.pretrain import PretrainCollator
from minimind_lab.data.sft import SFTCollator
from benchmark_training_capacity import encode_record, run_case, write_json


ORIGINAL = attention.scaled_dot_product_attention


def sdpa_masked(q, k, v, causal=True, attention_mask=None):
    allowed = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device)
    if causal:
        allowed = allowed.tril()
    if attention_mask is not None:
        allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0), None


def flash_right_padded(q, k, v, causal=True, attention_mask=None):
    if not causal or q.shape[-2] != k.shape[-2]:
        raise ValueError("probe flash adapter supports causal self-attention only")
    # Data is checked before entering this adapter. Right-pad keys are in the
    # future of every real query; ignored pad-query outputs need not match.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    except ImportError:
        context = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    with context:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True), None


def check_right_padding(batch):
    mask = batch["attention_mask"].bool()
    if not mask[:, 0].all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("right-padding invariant failed")
    if (batch["labels"][~mask] != -100).any():
        raise ValueError("padding labels must be ignored")


def compare(root, stage, batch):
    check_right_padding(batch)
    torch.manual_seed(42)
    config = json.loads((root / "model_config.json").read_text())["model"]
    model = MinimindForCausalLM(ModelConfig(**config)).float().cuda()
    if stage == "sft":
        load_model(model, str(root / "pretrained.safetensors"), strict=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    reports = {}
    reference = None
    for name, implementation in [("manual", ORIGINAL), ("sdpa_masked_auto", sdpa_masked),
                                  ("flash_causal_right_padding", flash_right_padded)]:
        attention.scaled_dot_product_attention = implementation
        model.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss_sum, count = causal_lm_loss_sum_and_count(logits.float(), batch["labels"])
            loss = loss_sum / count
            loss.backward()
            active_logits = logits[batch["attention_mask"].bool()].detach().float().cpu()
            gradients = {key: param.grad.detach().float().cpu().clone()
                         for key, param in model.named_parameters() if param.grad is not None}
            if not all(torch.isfinite(value).all() for value in gradients.values()):
                raise FloatingPointError("nonfinite gradients")
            if reference is None:
                reference = {"loss": loss.item(), "logits": active_logits, "gradients": gradients}
                reports[name] = {"status": "passed", "nll": loss.item(), "valid_targets": count.item()}
            else:
                diff = (active_logits - reference["logits"]).abs()
                squared_diff = sum((gradients[key].double() - value.double()).square().sum().item()
                                   for key, value in reference["gradients"].items())
                squared_reference = sum(value.double().square().sum().item()
                                        for value in reference["gradients"].values())
                reports[name] = {"status": "measured_not_formal_equivalence_certification",
                                 "nll": loss.item(), "nll_abs_difference": abs(loss.item() - reference["loss"]),
                                 "real_position_logits_max_abs_difference": diff.max().item(),
                                 "real_position_logits_mean_abs_difference": diff.mean().item(),
                                 "gradient_global_l2_relative_difference": math.sqrt(squared_diff / squared_reference),
                                 "gradients_finite": True, "pad_query_logits_compared": False}
            del logits, loss, loss_sum, count, gradients, active_logits
        except Exception as error:
            reports[name] = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
    attention.scaled_dot_product_attention = ORIGINAL
    del model, reference
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=["pretrain", "sft"], default=["pretrain", "sft"])
    parser.add_argument("--pretrain-micro", type=int, default=16)
    parser.add_argument("--sft-micro", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Measure compute without writing another large probe checkpoint.")
    parser.add_argument("--backends", nargs="+", choices=["manual", "sdpa_masked_auto", "flash_causal_right_padding"],
                        default=["sdpa_masked_auto", "flash_causal_right_padding"])
    args = parser.parse_args()
    if (args.pretrain_micro <= 0 or 256 % args.pretrain_micro
            or args.sft_micro <= 0 or 16 % args.sft_micro or args.seconds <= 0):
        parser.error("positive micro-batches must divide the original effective batch; seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.root / "tokenizer", local_files_only=True)
    report = {"status": "running", "formal_training": False, "comparisons": {}, "performance": [],
              "checkpoint_skipped": args.skip_checkpoint,
              "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(0),
                              "total_memory": torch.cuda.get_device_properties(0).total_memory,
                              "flash_attention_compiled": (torch.backends.cuda.is_flash_attention_available()
                                                           if hasattr(torch.backends.cuda, "is_flash_attention_available") else None),
                              "cpu_threads": torch.get_num_threads(), "tf32": False,
                              "deterministic_algorithms": True},
              "invariant": "Strict right padding, causal self-attention, positionwise remaining blocks, ignored padding targets.",
              "limits": ["No returned attention weights in candidate backends.",
                         "Floating-point operation order changes; not bitwise-equivalent training or an approved resume backend.",
                         "All correctness comparisons use real token positions; pad-query outputs can differ.",
                         "Performance sample is prepared mini data, not a full-data quality experiment."]}
    variants = [("pretrain", args.pretrain_micro, 256 // args.pretrain_micro),
                ("sft", args.sft_micro, 16 // args.sft_micro)]
    for stage, micro, accumulation in variants:
        if stage not in args.stages:
            continue
        records = [json.loads(line) for line in (args.root / f"{stage}_sample.jsonl").read_text(encoding="utf-8").splitlines()]
        features = [encode_record(stage, row, tokenizer) for row in records]
        collator = PretrainCollator(tokenizer, 768) if stage == "pretrain" else SFTCollator(tokenizer, 768)
        for begin in range(0, len(features), 4):
            check_right_padding(collator(features[begin:begin + 4]))
        comparison = compare(args.root, stage, collator(features[:4]))
        report["comparisons"][stage] = comparison
        write_json(args.output_dir / "summary.json", report)
        print(stage, "comparison", json.dumps(comparison), flush=True)
        for name, implementation in [("manual", ORIGINAL), ("sdpa_masked_auto", sdpa_masked),
                                      ("flash_causal_right_padding", flash_right_padded)]:
            if name not in args.backends:
                continue
            if comparison[name]["status"] == "failed":
                continue
            attention.scaled_dot_product_attention = implementation
            target = args.output_dir / name
            target.mkdir(exist_ok=True)
            try:
                result = run_case(args.root, target, stage, micro, accumulation,
                                  records, features, tokenizer, seconds=args.seconds, max_updates=1000,
                                  save_probe_checkpoint=not args.skip_checkpoint)
                report["performance"].append({"backend": name, **result})
            finally:
                attention.scaled_dot_product_attention = ORIGINAL
            write_json(args.output_dir / "summary.json", report)
    report["status"] = "completed"
    write_json(args.output_dir / "summary.json", report)


if __name__ == "__main__":
    main()
