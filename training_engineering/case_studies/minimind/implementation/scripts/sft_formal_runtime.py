"""Formal single-pass SFT orchestration over existing data/update/checkpoint components."""

import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset

from train_sft import ROOT, prepare, load_weights
from minimind_lab.data.sft import SFTDataset, SFTCollator
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.experiment import canonical_hash, sha256_file, write_json, lr_multiplier, memory_snapshot
from minimind_lab.training.step import train_update


RUNTIME_SCRIPTS = ("sft_formal_runtime.py", "train_frozen_sft.py", "freeze_sft_training.py")


def build_contract(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if set(cfg) != {"schema_version", "purpose", "base_candidate", "operations"} or cfg["schema_version"] != 1 or cfg["purpose"] != "sft_formal_recipe":
        raise ValueError("invalid formal recipe schema")
    ops = cfg["operations"]
    if set(ops) != {"checkpoint_interval", "validation_interval", "validation_batch_size", "validation_precision",
                    "checkpoint_dir", "checkpoint_retention", "wall_clock_budget_hours", "budget_scope", "minimum_checkpoint_free_gib"}:
        raise ValueError("unknown or missing operations fields")
    for key in ("checkpoint_interval", "validation_interval", "validation_batch_size"):
        if type(ops[key]) is not int or ops[key] <= 0:
            raise ValueError(f"invalid {key}")
    for key in ("wall_clock_budget_hours", "minimum_checkpoint_free_gib"):
        if type(ops[key]) not in (int, float) or not math.isfinite(ops[key]) or ops[key] <= 0:
            raise ValueError(f"invalid {key}")
    if ops["checkpoint_retention"] != "keep_all" or ops["validation_precision"] != "bf16" or ops["budget_scope"] != "cumulative_active_launch_time_excluding_paused_downtime_soft_update_boundary":
        raise ValueError("unsupported operations policy")
    candidate_path = (ROOT / cfg["base_candidate"]).resolve()
    core, data = prepare(candidate_path)
    identity = {"core": core["identity"], "core_contract_sha256": core["contract_sha256"],
                "operations": ops, "base_candidate_sha256": sha256_file(candidate_path),
                "formal_code_sha256": {name: sha256_file(ROOT / "scripts" / name) for name in RUNTIME_SCRIPTS}}
    return {"contract_sha256": canonical_hash(identity), "identity": identity, "config": cfg}, data


def run(contract, data, output, *, resume=None, gate_stop=None, gate_validation_records=64, clock=time.perf_counter):
    started = clock()
    core, ops = contract["identity"]["core"], contract["identity"]["operations"]
    cfg = core["candidate"]
    train, budget = cfg["training"], core["planned_updates"]
    is_gate = gate_stop is not None
    if is_gate and (type(gate_stop) is not int or not 0 < gate_stop < budget):
        raise ValueError("engineering stop must be strictly within full plan")
    target = gate_stop if is_gate else budget
    scope = "engineering_gate" if is_gate else "formal_sft"
    count = gate_validation_records if is_gate else core["validation_records"]
    if type(count) is not int or not 0 < count <= core["validation_records"]:
        raise ValueError("invalid validation coverage")
    checkpoint_root = output if is_gate else Path(ops["checkpoint_dir"]).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not is_gate and shutil.disk_usage(checkpoint_root).free < ops["minimum_checkpoint_free_gib"] * 1024**3:
        raise ValueError("insufficient checkpoint disk reserve")
    if not is_gate and resume is None and any(checkpoint_root.iterdir()):
        raise FileExistsError("fresh formal run requires an empty dedicated checkpoint directory")
    device = torch.device(train["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if train["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unavailable")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(train["seed"])
        model, tokenizer, _, _ = load_weights((ROOT / cfg["initial_bundle"]).resolve(), data, device)
        model.requires_grad_(True)
        dataset = SFTDataset(data["artifacts"]["train"]["path"], tokenizer)
        validation = SFTDataset(data["artifacts"]["validation"]["path"], tokenizer)
        if len(dataset) != core["train_records"] or len(validation) != core["validation_records"]:
            raise ValueError("dataset counts changed")
        sampler = StatefulSampler(len(dataset), train["sampler_seed"])
        if canonical_hash(sampler.indices) != core["sample_order_sha256"]:
            raise ValueError("sample order changed")
        collator = SFTCollator(tokenizer, data["encoding"]["max_length"])
        loader = DataLoader(dataset, batch_size=train["micro_batch_size"], sampler=sampler, collate_fn=collator,
                            num_workers=0, drop_last=False, generator=torch.Generator().manual_seed(123))
        valid_loader = DataLoader(Subset(validation, range(count)), batch_size=ops["validation_batch_size"], collate_fn=collator,
                                  num_workers=0, generator=torch.Generator().manual_seed(456))
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], betas=tuple(train["betas"]),
                                     eps=train["eps"], weight_decay=train["weight_decay"], foreach=train["foreach"], fused=train["fused"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, budget, train["warmup_updates"], train["min_lr_ratio"]))
        metadata = {"contract_sha256": contract["contract_sha256"], "execution_kind": scope,
                    "initial_model_state_sha256": cfg["initial_model_state_sha256"], "validation_records": count}
        step, tokens, prior_seconds = 0, 0, 0.0
        last_checkpoint = None
        if resume is not None:
            resume = Path(resume).resolve()
            if not is_gate and resume.parent != checkpoint_root:
                raise ValueError("formal resume must use its dedicated checkpoint directory")
            receipt = json.loads(Path(str(resume) + ".receipt.json").read_text(encoding="utf-8"))
            if receipt["checkpoint_sha256"] != sha256_file(resume) or receipt["contract_sha256"] != contract["contract_sha256"]:
                raise ValueError("checkpoint receipt mismatch")
            step, restored = load_checkpoint(resume, model, optimizer, scheduler, sampler=sampler,
                                             expected_metadata=metadata, return_metadata=True)
            tokens, prior_seconds = restored["valid_tokens_seen"], receipt["active_run_seconds"]
            if (not math.isfinite(prior_seconds) or prior_seconds < 0 or scheduler.last_epoch != step
                    or receipt["global_step"] != step or receipt["committed_samples"] != sampler.committed_position
                    or canonical_hash(sampler.indices) != core["sample_order_sha256"]
                    or sampler.committed_position != min(step * core["effective_batch_records"], len(dataset))):
                raise ValueError("restored progress mismatch")
            last_checkpoint = resume
        if not 0 <= step <= budget or (is_gate and step >= target):
            raise ValueError("invalid start or engineering boundary")
        dtype = torch.bfloat16 if train["precision"] == "bf16" else None
        active = lambda: prior_seconds + clock() - started
        expired = lambda: active() >= ops["wall_clock_budget_hours"] * 3600

        def log(event):
            with (output / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event), flush=True)

        def manifest(status, **extra):
            value = {"status": status, "run_kind": scope, "global_step": step,
                     "contract_sha256": contract["contract_sha256"], "committed_samples": sampler.committed_position,
                     "valid_tokens_seen": tokens, "active_run_seconds": active(), **extra}
            write_json(output / "run_manifest.json", value)
            return value

        def validate(event):
            start = clock()
            result = evaluate_nll(model, valid_loader, device=device, autocast_dtype=dtype)
            log({"event": event, "global_step": step, "records": count, "precision": train["precision"],
                 "scope": scope, "seconds": clock() - start, **result})
            return result

        def checkpoint():
            nonlocal last_checkpoint
            path = checkpoint_root / f"checkpoint-{step}.pt"
            if last_checkpoint is not None and path == last_checkpoint:
                return path
            if path.exists() or Path(str(path) + ".receipt.json").exists():
                raise FileExistsError(path)
            before = clock()
            save_checkpoint(path, model, optimizer, scheduler, step, sampler=sampler,
                            metadata={**metadata, "valid_tokens_seen": tokens})
            receipt = {"contract_sha256": contract["contract_sha256"], "checkpoint_sha256": sha256_file(path),
                       "global_step": step, "committed_samples": sampler.committed_position,
                       "active_run_seconds": active()}
            write_json(Path(str(path) + ".receipt.json"), receipt)
            last_checkpoint = path
            log({"event": "checkpoint_saved", "global_step": step, "seconds": clock() - before, "path": str(path)})
            return path

        manifest("running", training_started=False)
        initial = None if expired() else validate("initial_or_resume_validation")
        iterator = iter(loader)
        stop_reason = "completed"
        while step < target:
            if expired():
                stop_reason = "time_budget"
                break
            before = clock()
            window = []
            for _ in range(train["gradient_accumulation_steps"]):
                batch = next(iterator, None)
                if batch is None:
                    break
                window.append(batch)
            if not window:
                raise RuntimeError("data exhausted before full plan")
            manifest("running", training_started=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            updated = train_update(model, optimizer, window, scheduler, train["max_grad_norm"], device=device, autocast_dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sampler.commit(sum(b["input_ids"].shape[0] for b in window))
            step += 1
            tokens += updated["valid_tokens"]
            optimizer.zero_grad(set_to_none=True)
            log({"event": "update", "global_step": step, **updated, "window_seconds": clock() - before,
                 "committed_samples": sampler.committed_position, "memory": memory_snapshot(device)})
            manifest("running", training_started=True)
            if step % ops["checkpoint_interval"] == 0:
                checkpoint()
            if step < target and not expired() and step % ops["validation_interval"] == 0:
                validate("periodic_validation")
        if step == budget:
            if sampler.committed_position != len(dataset) or tokens != data["split_statistics"]["train"]["total_targets_after_shift"]:
                raise ValueError("full epoch sample/target coverage mismatch")
        elif step == target:
            stop_reason = "engineering_boundary"
        final = None if stop_reason == "time_budget" or expired() else validate("final_validation")
        checkpoint()
        return {**manifest("completed" if step == budget else "paused"), "stop_reason": stop_reason,
                "checkpoint": str(last_checkpoint), "resume_from": str(resume) if resume else None,
                "initial_or_resume_validation": initial, "final_validation": final,
                "planned_updates": budget, "training_started": step > 0,
                "formal_training_started": not is_gate and step > 0}
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
