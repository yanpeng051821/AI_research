"""Bounded capacity probe using unchanged project model, encoding and update code."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import PretrainCollator, encode_pretrain_text
from minimind_lab.data.sft import SFTCollator, encode_sft_conversation
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss_sum_and_count
from minimind_lab.training.step import train_update
from minimind_lab.training.checkpoint import save_checkpoint


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_record(stage, record, tokenizer):
    if stage == "pretrain":
        return {"input_ids": encode_pretrain_text(record["text"], tokenizer, 768)}
    return encode_sft_conversation(record["conversations"], tokenizer)


def run_case(root, out, stage, micro, accumulation, records, features, tokenizer, seconds, max_updates,
             save_probe_checkpoint=True):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = json.loads((root / "model_config.json").read_text())["model"]
    model = MinimindForCausalLM(ModelConfig(**config)).float()
    if stage == "sft":
        load_model(model, str(root / "pretrained.safetensors"), strict=True)
    model = model.cuda()
    lr = 5e-4 if stage == "pretrain" else 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01, foreach=False, fused=False)
    # The old full-epoch schedule is only a timing surrogate, not a new frozen full-data recipe.
    total_steps = 4943 if stage == "pretrain" else 49087
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 + 0.9 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)) / 2)
    collator = PretrainCollator(tokenizer, 768) if stage == "pretrain" else SFTCollator(tokenizer, 768)
    tag = f"{stage}_micro{micro}_accum{accumulation}"
    case_dir = out / tag
    case_dir.mkdir(exist_ok=False)
    cursor = 0

    def batches(online=False):
        nonlocal cursor
        result = []
        for _ in range(accumulation):
            indices = [(cursor + i) % len(features) for i in range(micro)]
            cursor += micro
            items = ([encode_record(stage, records[i], tokenizer) for i in indices]
                     if online else [features[i] for i in indices])
            result.append(collator(items))
        return result

    start = time.perf_counter()
    history = []
    try:
        for _ in range(2):
            train_update(model, optimizer, batches(), scheduler, device="cuda", autocast_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured_start = time.perf_counter()
        for index in range(max_updates):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            pending = batches()
            built = time.perf_counter()
            result = train_update(model, optimizer, pending, scheduler, device="cuda", autocast_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            row = {**result, "index": index, "seconds": elapsed,
                   "batch_build_seconds": built - tick,
                   "allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                   "peak_allocated": torch.cuda.max_memory_allocated()}
            history.append(row)
            with (case_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if len(history) >= 3 and time.perf_counter() - measured_start >= seconds:
                break
        online = []
        for _ in range(3):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            pending = batches(online=True)
            built = time.perf_counter()
            result = train_update(model, optimizer, pending, scheduler, device="cuda", autocast_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            online.append({"seconds": time.perf_counter() - tick, "encode_collate_seconds": built - tick,
                           "valid_tokens": result["valid_tokens"]})
        checkpoint_seconds = checkpoint_size = checkpoint_sha = hash_seconds = None
        if save_probe_checkpoint:
            checkpoint = case_dir / "probe_checkpoint.pt"
            tick = time.perf_counter()
            save_checkpoint(checkpoint, model, optimizer, scheduler, len(history) + 5,
                            metadata={"purpose": "capacity_probe_not_formal_training"})
            checkpoint_seconds = time.perf_counter() - tick
            checkpoint_size = checkpoint.stat().st_size
            tick = time.perf_counter()
            checkpoint_sha = digest(checkpoint)
            hash_seconds = time.perf_counter() - tick
        model.eval()
        validation_records = 64
        validation_targets = 0
        validation_loss = 0.0
        tick = time.perf_counter()
        with torch.inference_mode():
            for begin in range(0, validation_records, 4):
                batch = {k: v.cuda() for k, v in collator(features[begin:begin + 4]).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss, count = causal_lm_loss_sum_and_count(logits.float(), batch["labels"])
                validation_loss += loss.item()
                validation_targets += count.item()
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - tick
        count_records = micro * accumulation * len(history)
        measured_seconds = sum(row["seconds"] for row in history)
        report = {"status": "passed", "stage": stage, "micro_batch": micro,
                  "gradient_accumulation": accumulation, "effective_batch": micro * accumulation,
                  "measured_updates": len(history), "timed_records": count_records,
                  "cached_update_median_seconds": statistics.median(row["seconds"] for row in history),
                  "cached_records_per_second": count_records / measured_seconds,
                  "online_update_median_seconds": statistics.median(row["seconds"] for row in online),
                  "online_records_per_second": micro * accumulation * len(online) / sum(row["seconds"] for row in online),
                  "online_encode_collate_median_seconds": statistics.median(row["encode_collate_seconds"] for row in online),
                  "supervised_targets_per_second": sum(row["valid_tokens"] for row in history) / measured_seconds,
                  "peak_allocated_bytes": max(row["peak_allocated"] for row in history),
                  "reserved_bytes": max(row["reserved"] for row in history),
                  "checkpoint_skipped": not save_probe_checkpoint,
                  "checkpoint_bytes": checkpoint_size, "checkpoint_save_seconds": checkpoint_seconds,
                  "checkpoint_sha256": checkpoint_sha, "checkpoint_hash_seconds": hash_seconds,
                  "bf16_validation_64_records_seconds": validation_seconds,
                  "bf16_validation_nll": validation_loss / validation_targets,
                  "total_case_seconds": time.perf_counter() - start,
                  "online_samples": online,
                  "limits": ["Mini prepared-data sample, not cleaned full source distribution.",
                             "BF16 validation is throughput only, not held-out quality evaluation.",
                             "No formal sampler/resume/receipt gate or full-data recipe freeze.",
                             "Source PyTorch differs from the local training environment.",
                             "Cached timing excludes encoding; online includes encoding/collation but not full dataset I/O."]}
    except torch.cuda.OutOfMemoryError:
        report = {"status": "out_of_memory", "stage": stage, "micro_batch": micro,
                  "gradient_accumulation": accumulation}
    except Exception as exc:
        (case_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report = {"status": "failed", "stage": stage, "micro_batch": micro,
                  "gradient_accumulation": accumulation, "error_type": type(exc).__name__, "error": str(exc)}
    write_json(case_dir / "result.json", report)
    print(json.dumps(report), flush=True)
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--max-updates", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    write_json(args.output_dir / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": props.name, "total_memory": props.total_memory, "sm_count": props.multi_processor_count,
               "cpu_threads": torch.get_num_threads(), "deterministic": True, "tf32": False,
               "formal_training": False})
    tokenizer = AutoTokenizer.from_pretrained(args.root / "tokenizer", local_files_only=True)
    summary = []
    for stage, variants in [("pretrain", [(4, 64), (16, 16), (32, 8)]),
                            ("sft", [(4, 4), (8, 2), (16, 1)])]:
        records = [json.loads(line) for line in (args.root / f"{stage}_sample.jsonl").read_text(encoding="utf-8").splitlines()]
        tick = time.perf_counter()
        features = [encode_record(stage, row, tokenizer) for row in records]
        write_json(args.output_dir / f"{stage}_sample_encoding.json", {
            "records": len(records), "seconds": time.perf_counter() - tick,
            "min_length": min(len(f["input_ids"]) for f in features),
            "max_length": max(len(f["input_ids"]) for f in features),
            "mean_length": statistics.mean(len(f["input_ids"]) for f in features),
            "source_sha256": digest(args.root / f"{stage}_sample.jsonl")})
        for micro, accumulation in variants:
            summary.append(run_case(args.root, args.output_dir, stage, micro, accumulation,
                                    records, features, tokenizer, args.seconds, args.max_updates))
            write_json(args.output_dir / "summary.json", {"status": "running", "results": summary})
    write_json(args.output_dir / "summary.json", {"status": "completed", "results": summary})


if __name__ == "__main__":
    main()
