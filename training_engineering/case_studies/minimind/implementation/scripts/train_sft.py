"""Prepare the SFT candidate, or explicitly run a bounded engineering probe.

This candidate entry deliberately cannot launch the full training plan.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, Subset

from evaluate_sft import load_weights
from freeze_sft_data_recipe import check as check_data_freeze
from load_exported_pretrain import verify_bundle
from minimind_lab.data.sft import SFTCollator, SFTDataset
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.training.checkpoint import load_checkpoint, save_checkpoint
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.experiment import (
    canonical_hash, lr_multiplier, memory_snapshot, positive_integer,
    sha256_file, write_json,
)
from minimind_lab.training.step import train_update


ROOT = Path(__file__).resolve().parents[1]
TRAINING_KEYS = {
    "epochs", "device", "precision", "seed", "sampler_seed", "micro_batch_size",
    "gradient_accumulation_steps", "learning_rate", "weight_decay", "betas", "eps",
    "foreach", "fused", "warmup_updates", "min_lr_ratio", "max_grad_norm", "num_workers",
}


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version", "status", "purpose", "data_freeze", "data_freeze_sha256",
                "data_contract_sha256", "initial_bundle", "initial_model_state_sha256",
                "scoring_patch", "scoring_patch_sha256", "training", "operations", "pending_confirmation"}
    if set(cfg) != expected or cfg["schema_version"] != 1 or cfg["purpose"] != "sft":
        raise ValueError("unknown SFT candidate schema")
    if cfg["status"] != "candidate_not_frozen" or cfg["operations"]["formal_training_authorized"] is not False:
        raise ValueError("this entry accepts only an unfrozen, non-authorized candidate")
    train = cfg["training"]
    if set(train) != TRAINING_KEYS:
        raise ValueError("missing or unknown training fields")
    if train["epochs"] != 1 or train["num_workers"] != 0:
        raise ValueError("this entry supports one data pass and num_workers=0")
    for key in ("micro_batch_size", "gradient_accumulation_steps"):
        positive_integer(train[key], key)
    for key in ("seed", "sampler_seed", "warmup_updates"):
        if type(train[key]) is not int or train[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("learning_rate", "eps", "max_grad_norm"):
        if type(train[key]) not in (int, float) or not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("invalid weight decay")
    if len(train["betas"]) != 2 or any(not 0 <= b < 1 for b in train["betas"]):
        raise ValueError("invalid AdamW betas")
    if not 0 <= train["min_lr_ratio"] <= 1:
        raise ValueError("invalid minimum LR ratio")
    if any(type(train[k]) is not bool for k in ("foreach", "fused")):
        raise ValueError("optimizer execution flags must be booleans")
    if train["device"] not in ("cpu", "cuda") or train["precision"] not in ("fp32", "bf16"):
        raise ValueError("unsupported device or precision")
    if train["precision"] == "bf16" and train["device"] != "cuda":
        raise ValueError("BF16 requires CUDA in this entry")
    return cfg


def prepare(config_path, root=ROOT):
    cfg = read_config(config_path)
    marker_path = (root / cfg["data_freeze"]).resolve()
    patch_path = (root / cfg["scoring_patch"]).resolve()
    for path, expected in ((marker_path, cfg["data_freeze_sha256"]),
                           (patch_path, cfg["scoring_patch_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError(f"candidate reference identity mismatch: {path}")
    checked = check_data_freeze(marker_path)
    if checked["data_contract_sha256"] != cfg["data_contract_sha256"]:
        raise ValueError("data contract identity mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    data = json.loads(Path(marker["files"]["contract"]["path"]).read_text(encoding="utf-8"))
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    if patch["version"] != "scoring_v2" or patch["status"] != "user_approved_before_sft":
        raise ValueError("expected the approved V2 scoring patch")
    bundle = (root / cfg["initial_bundle"]).resolve()
    bundle_manifest = verify_bundle(bundle)
    if bundle_manifest["model_state_sha256"] != cfg["initial_model_state_sha256"]:
        raise ValueError("initial model is not the agreed B0")
    model_config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    train = cfg["training"]
    count = checked["train_records"]
    effective = train["micro_batch_size"] * train["gradient_accumulation_steps"]
    budget = math.ceil(count / effective)
    if train["warmup_updates"] >= budget:
        raise ValueError("warmup covers the whole training plan")
    sampler = StatefulSampler(count, train["sampler_seed"])
    code_paths = sorted((root / "src/minimind_lab").rglob("*.py")) + [
        root / "scripts" / name for name in (
            "train_sft.py", "evaluate_sft.py", "freeze_sft_data_recipe.py",
            "load_exported_pretrain.py", "inspect_sft_eval_draft.py",
        )
    ]
    identity = {
        "candidate": cfg, "model": model_config,
        "bundle_manifest_sha256": sha256_file(bundle / "export_manifest.json"),
        "sample_order_sha256": canonical_hash(sampler.indices),
        "train_records": count, "validation_records": checked["validation_records"],
        "planned_updates": budget, "effective_batch_records": effective,
        "loss_normalization": "sum loss / valid assistant causal targets in the entire update window",
        "runtime": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers", "safetensors")},
        "code_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in code_paths},
    }
    return {"identity": identity, "contract_sha256": canonical_hash(identity),
            "status": "candidate_prepared_not_frozen", "training_started": False}, data


def run_probe(contract, data, output, *, stop_after, validation_records, resume=None, root=ROOT):
    """Keep the original full-pass scheduler horizon, even for a bounded probe."""
    positive_integer(stop_after, "probe_updates")
    positive_integer(validation_records, "validation_records")
    cfg = contract["identity"]["candidate"]
    train = cfg["training"]
    budget = contract["identity"]["planned_updates"]
    if stop_after >= budget:
        raise ValueError("candidate probes must stop before the full training budget")
    if validation_records > contract["identity"]["validation_records"]:
        raise ValueError("validation probe exceeds the frozen validation set")
    device = torch.device(train["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent CPU fallback")
    if train["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unavailable")
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(train["seed"])
        model, tokenizer, _, _ = load_weights((root / cfg["initial_bundle"]).resolve(), data, device)
        model.requires_grad_(True)
        dataset = SFTDataset(data["artifacts"]["train"]["path"], tokenizer)
        validation = SFTDataset(data["artifacts"]["validation"]["path"], tokenizer)
        if (len(dataset) != contract["identity"]["train_records"]
                or len(validation) != contract["identity"]["validation_records"]):
            raise ValueError("SFT Dataset count differs from frozen contract")
        sampler = StatefulSampler(len(dataset), train["sampler_seed"])
        if canonical_hash(sampler.indices) != contract["identity"]["sample_order_sha256"]:
            raise ValueError("sample order mismatch")
        collator = SFTCollator(tokenizer, data["encoding"]["max_length"])
        loader = DataLoader(dataset, batch_size=train["micro_batch_size"], sampler=sampler,
                            collate_fn=collator, num_workers=0, drop_last=False,
                            generator=torch.Generator().manual_seed(123))
        valid_loader = DataLoader(Subset(validation, range(validation_records)), batch_size=4,
                                 collate_fn=collator, num_workers=0,
                                 generator=torch.Generator().manual_seed(456))
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"],
                                     weight_decay=train["weight_decay"], betas=tuple(train["betas"]),
                                     eps=train["eps"], foreach=train["foreach"], fused=train["fused"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: lr_multiplier(s, budget, train["warmup_updates"], train["min_lr_ratio"]))
        metadata = {"contract_sha256": contract["contract_sha256"],
                    "initial_model_state_sha256": cfg["initial_model_state_sha256"],
                    "validation_probe_records": validation_records}
        step, tokens_seen = 0, 0
        if resume is not None:
            receipt = json.loads(Path(str(resume) + ".receipt.json").read_text(encoding="utf-8"))
            if receipt["checkpoint_sha256"] != sha256_file(resume) or receipt["contract_sha256"] != contract["contract_sha256"]:
                raise ValueError("resume receipt mismatch")
            step, restored = load_checkpoint(resume, model, optimizer, scheduler, sampler=sampler,
                                             expected_metadata=metadata, return_metadata=True)
            tokens_seen = restored["valid_tokens_seen"]
            if (receipt["global_step"] != step or receipt["committed_samples"] != sampler.committed_position
                    or sampler.committed_position != min(step * contract["identity"]["effective_batch_records"], len(dataset))
                    or canonical_hash(sampler.indices) != contract["identity"]["sample_order_sha256"]):
                raise ValueError("restored data position or order mismatch")
        if not 0 <= step < stop_after:
            raise ValueError("probe stop must be beyond restored progress")
        dtype = torch.bfloat16 if train["precision"] == "bf16" else None

        def log(event):
            with (output / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event), flush=True)

        def validate(label):
            started = time.perf_counter()
            result = evaluate_nll(model, valid_loader, device=device, autocast_dtype=dtype)
            log({"event": label, "global_step": step, "records": validation_records,
                 "scope": "fixed_prefix_engineering_probe_not_final_paired_evaluation",
                 "precision": train["precision"], "seconds": time.perf_counter() - started, **result})
            return result

        initial_validation = validate("initial_or_resume_validation")
        iterator = iter(loader)
        while step < stop_after:
            window_start = time.perf_counter()
            window = []
            for _ in range(train["gradient_accumulation_steps"]):
                batch = next(iterator, None)
                if batch is None:
                    break
                window.append(batch)
            if not window:
                raise RuntimeError("data exhausted before probe boundary")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            write_json(output / "run_manifest.json", {"status": "running", "global_step": step,
                       "contract_sha256": contract["contract_sha256"], "training_started": True,
                       "committed_samples": sampler.committed_position,
                       "run_kind": "engineering_probe_not_formal_sft"})
            started = time.perf_counter()
            result = train_update(model, optimizer, window, scheduler, train["max_grad_norm"],
                                  device=device, autocast_dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            sampler.commit(sum(b["input_ids"].shape[0] for b in window))
            step += 1
            tokens_seen += result["valid_tokens"]
            optimizer.zero_grad(set_to_none=True)
            log({"event": "update", "global_step": step, **result,
                 "update_seconds": elapsed, "window_seconds": time.perf_counter() - window_start,
                 "committed_samples": sampler.committed_position, "memory": memory_snapshot(device)})
            write_json(output / "run_manifest.json", {"status": "running", "global_step": step,
                       "contract_sha256": contract["contract_sha256"], "training_started": True,
                       "run_kind": "engineering_probe_not_formal_sft"})
        final_validation = validate("final_probe_validation")
        checkpoint = output / f"checkpoint-{step}.pt"
        started = time.perf_counter()
        save_checkpoint(checkpoint, model, optimizer, scheduler, step, sampler=sampler,
                        metadata={**metadata, "valid_tokens_seen": tokens_seen})
        write_json(Path(str(checkpoint) + ".receipt.json"), {
            "contract_sha256": contract["contract_sha256"], "checkpoint_sha256": sha256_file(checkpoint),
            "global_step": step, "committed_samples": sampler.committed_position})
        log({"event": "checkpoint_saved", "global_step": step, "seconds": time.perf_counter() - started})
        return {"status": "paused", "run_kind": "engineering_probe_not_formal_sft",
                "global_step": step, "planned_updates": budget,
                "committed_samples": sampler.committed_position, "valid_tokens_seen": tokens_seen,
                "contract_sha256": contract["contract_sha256"], "training_started": True,
                "resume_from": str(resume) if resume else None, "checkpoint": str(checkpoint),
                "initial_or_resume_validation": initial_validation, "final_validation": final_validation}
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--probe-updates", type=int)
    parser.add_argument("--validation-records", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.prepare_only and (args.resume is not None or args.validation_records is not None):
        parser.error("prepare-only does not load a checkpoint or run validation")
    if not args.prepare_only and args.validation_records is None:
        parser.error("probe requires explicit --validation-records")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "preparing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_started": False, "formal_training_authorized": False}
    write_json(output / "run_manifest.json", manifest)
    try:
        contract, data = prepare(args.config.resolve())
        write_json(output / "candidate_contract.json", contract)
        if args.prepare_only:
            result = {"status": "prepared_not_trained", "training_started": False,
                      "contract_sha256": contract["contract_sha256"],
                      "planned_updates": contract["identity"]["planned_updates"],
                      "formal_training_authorized": False}
        else:
            result = run_probe(contract, data, output, stop_after=args.probe_updates,
                               validation_records=args.validation_records, resume=args.resume)
        write_json(output / "result.json", result)
        write_json(output / "run_manifest.json", {**manifest, **result})
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        progress = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        write_json(output / "run_manifest.json", {**manifest, **progress, "status": "failed",
                   "error_type": type(exc).__name__, "error": str(exc)})
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
