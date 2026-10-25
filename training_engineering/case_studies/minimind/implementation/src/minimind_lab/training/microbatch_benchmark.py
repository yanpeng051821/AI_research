"""Controlled CUDA micro-batch benchmark for the frozen MiniMind pretraining input."""

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import PretrainCollator, PretrainDataset
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.experiment import (
    memory_snapshot,
    model_state_hash,
    prepare_contract,
    sha256_file,
    write_json,
)
from minimind_lab.training.step import train_update


def resolve_benchmark_config(path: Path, root: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "purpose",
        "source_config",
        "source_config_sha256",
        "effective_batch_records",
        "candidates",
        "excluded_candidates",
        "warmup_updates",
        "measured_updates",
        "minimum_device_free_ratio",
    }
    if set(raw) != required:
        raise ValueError("unknown or missing benchmark configuration fields")
    if raw["schema_version"] != 1 or raw["purpose"] != "microbatch_benchmark":
        raise ValueError("unsupported benchmark configuration")

    source_config = (root / raw["source_config"]).resolve()
    if sha256_file(source_config) != raw["source_config_sha256"]:
        raise ValueError("source configuration identity mismatch")

    for name in ("effective_batch_records", "warmup_updates", "measured_updates"):
        if type(raw[name]) is not int or raw[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    candidates = raw["candidates"]
    if not candidates or any(type(value) is not int or value <= 0 for value in candidates):
        raise ValueError("candidates must contain positive integers")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if any(raw["effective_batch_records"] % value for value in candidates):
        raise ValueError("every candidate must divide effective_batch_records")
    excluded = raw["excluded_candidates"]
    if not isinstance(excluded, list) or any(not isinstance(value, dict) for value in excluded):
        raise ValueError("excluded_candidates must be a list of records")
    ratio = raw["minimum_device_free_ratio"]
    if not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("minimum_device_free_ratio must be between zero and one")

    return {**raw, "source_config_path": source_config}


def select_candidate(results: list[dict], minimum_free_ratio: float) -> dict | None:
    eligible = [
        result
        for result in results
        if result["status"] == "completed"
        and result["minimum_device_free_ratio"] >= minimum_free_ratio
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda result: (
            result["measured_records_per_second"],
            result["minimum_device_free_ratio"],
        ),
    )


def _sampler_prefix_hash(indices: list[int], count: int) -> str:
    payload = json.dumps(indices[:count], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_candidate(
    *,
    candidate: int,
    effective_batch: int,
    warmup_updates: int,
    measured_updates: int,
    source: dict,
    paths: dict,
    tokenizer,
    dataset,
    collator,
) -> dict:
    train = source["training"]
    device = torch.device(train["device"])
    accumulation = effective_batch // candidate
    total_updates = warmup_updates + measured_updates
    sample_count = effective_batch * total_updates

    torch.manual_seed(train["seed"])
    model = MinimindForCausalLM(ModelConfig(**source["model"]))
    initial_model_sha256 = model_state_hash(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["learning_rate"],
        betas=tuple(train["betas"]),
        eps=train["eps"],
        weight_decay=train["weight_decay"],
        foreach=train["foreach"],
        fused=train["fused"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = StatefulSampler(len(dataset), seed=train["sampler_seed"])
    prefix_sha256 = _sampler_prefix_hash(sampler.indices, sample_count)
    loader = DataLoader(
        dataset,
        batch_size=candidate,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(train["seed"] + 1),
    )
    iterator = iter(loader)
    dtype = torch.bfloat16 if train["precision"] == "bf16" else None
    updates = []

    try:
        for update_index in range(total_updates):
            window_started = time.perf_counter()
            window = [next(iterator) for _ in range(accumulation)]
            records = sum(batch["input_ids"].shape[0] for batch in window)
            if records != effective_batch:
                raise RuntimeError("benchmark window did not contain the effective batch")

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            metrics = train_update(
                model,
                optimizer,
                window,
                scheduler,
                train["max_grad_norm"],
                device=device,
                autocast_dtype=dtype,
            )
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            window_seconds = time.perf_counter() - window_started
            sampler.commit(records)
            memory = memory_snapshot(device)
            memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            updates.append(
                {
                    "update": update_index + 1,
                    "measured": update_index >= warmup_updates,
                    "records": records,
                    **metrics,
                    "update_seconds": update_seconds,
                    "window_seconds": window_seconds,
                    "records_per_update_second": records / update_seconds,
                    "records_per_window_second": records / window_seconds,
                    "valid_tokens_per_window_second": metrics["valid_tokens"] / window_seconds,
                    "memory": memory,
                }
            )
            del window

        measured = [row for row in updates if row["measured"]]
        measured_records = sum(row["records"] for row in measured)
        measured_tokens = sum(row["valid_tokens"] for row in measured)
        measured_window_seconds = sum(row["window_seconds"] for row in measured)
        total_device_bytes = updates[-1]["memory"]["total_device_bytes"]
        minimum_free_ratio = min(
            row["memory"]["free_device_bytes"] / total_device_bytes for row in updates
        )
        return {
            "micro_batch_size": candidate,
            "gradient_accumulation_steps": accumulation,
            "status": "completed",
            "initial_model_sha256": initial_model_sha256,
            "sampler_prefix_sha256": prefix_sha256,
            "minimum_device_free_ratio": minimum_free_ratio,
            "median_measured_update_seconds": statistics.median(row["update_seconds"] for row in measured),
            "median_measured_window_seconds": statistics.median(row["window_seconds"] for row in measured),
            "measured_records_per_second": measured_records / measured_window_seconds,
            "measured_valid_tokens_per_second": measured_tokens / measured_window_seconds,
            "updates": updates,
        }
    except torch.cuda.OutOfMemoryError as error:
        return {
            "micro_batch_size": candidate,
            "gradient_accumulation_steps": accumulation,
            "status": "oom",
            "initial_model_sha256": initial_model_sha256,
            "sampler_prefix_sha256": prefix_sha256,
            "error": str(error),
            "updates": updates,
        }
    finally:
        del iterator, loader, sampler, scheduler, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def run_microbatch_benchmark(config_path: Path, output_dir: Path, root: Path) -> dict:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    benchmark = resolve_benchmark_config(config_path, root)
    contract, paths, _ = prepare_contract(benchmark["source_config_path"], root)
    source = contract["resolved_config"]
    train = source["training"]
    if train["device"] != "cuda":
        raise ValueError("micro-batch benchmark requires CUDA")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if train["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unsupported on this CUDA device")

    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = datetime.now(timezone.utc)
    try:
        tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_dir"], local_files_only=True)
        dataset = PretrainDataset(paths["train"], tokenizer, source["data"]["max_length"])
        collator = PretrainCollator(tokenizer, source["data"]["max_length"])
        results = list(benchmark["excluded_candidates"])
        for candidate in benchmark["candidates"]:
            print(f"benchmarking micro_batch_size={candidate}", flush=True)
            result = _run_candidate(
                    candidate=candidate,
                    effective_batch=benchmark["effective_batch_records"],
                    warmup_updates=benchmark["warmup_updates"],
                    measured_updates=benchmark["measured_updates"],
                    source=source,
                    paths=paths,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    collator=collator,
                )
            results.append(result)
            write_json(
                output_dir / "partial_results.json",
                {"status": "running", "completed_candidates": results},
            )
            print(f"completed micro_batch_size={candidate}: {result['status']}", flush=True)
        chosen = select_candidate(results, benchmark["minimum_device_free_ratio"])
        report = {
            "schema_version": 1,
            "status": "completed",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_config_sha256": sha256_file(config_path),
            "source_contract_sha256": contract["contract_sha256"],
            "effective_batch_records": benchmark["effective_batch_records"],
            "warmup_updates": benchmark["warmup_updates"],
            "measured_updates": benchmark["measured_updates"],
            "minimum_device_free_ratio": benchmark["minimum_device_free_ratio"],
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "precision": train["precision"],
            "sequence_length": source["data"]["max_length"],
            "results": results,
            "recommended_micro_batch_size": chosen["micro_batch_size"] if chosen else None,
            "recommended_gradient_accumulation_steps": chosen["gradient_accumulation_steps"] if chosen else None,
            "recommendation_rule": "highest end-to-end measured records/s among successful candidates retaining the configured device-free ratio",
        }
        write_json(output_dir / "microbatch_benchmark.json", report)
        (output_dir / "partial_results.json").unlink()
        return report
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
