"""Resume-capable single-GPU training for full-data v1; never edits old entries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import minimind_lab.models.attention as attention
from minimind_lab.data.full_source import OffsetJSONLDataset
from minimind_lab.data.pretrain import PretrainCollator
from minimind_lab.data.sft import SFTCollator
from minimind_lab.data.full_source import BlockShuffleSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.checkpoint import load_checkpoint, save_checkpoint
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.experiment import lr_multiplier, sha256_file, write_json
from minimind_lab.training.step import train_update


def install_flash_backend():
    def flash_attention(q, k, v, causal=True, attention_mask=None):
        if not causal or q.shape[-2] != k.shape[-2]:
            raise ValueError("full-data Flash adapter only permits causal self-attention")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True), None

    attention.scaled_dot_product_attention = flash_attention


def checked_collator(collator):
    def collate(records):
        batch = collator(records)
        mask = batch["attention_mask"].bool()
        if (not mask[:, 0].all()
                or ((~mask[:, :-1]) & mask[:, 1:]).any()
                or (batch["labels"][~mask] != -100).any()):
            raise ValueError("Flash adapter requires strict right padding and ignored pad labels")
        return batch
    return collate


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure(stage, recipe):
    item = recipe[stage]
    model = ModelConfig(**{key: value for key, value in recipe["model"].items()
                           if key not in {"source", "tokenizer"}})
    training = item
    if training["epochs"] != 2:
        raise ValueError("confirmed full-data recipe requires exactly two epochs per stage")
    if training["micro_batch_size"] * training["gradient_accumulation_steps"] != training["effective_batch_records"]:
        raise ValueError("microbatch and accumulation do not preserve the declared effective batch")
    return model, training


def newest_checkpoint(checkpoint_dir):
    candidates = sorted(Path(checkpoint_dir).glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def prune_checkpoints(checkpoint_dir, keep=2):
    paths = list(Path(checkpoint_dir).glob("step-*.pt"))
    periodic = sorted((p for p in paths if ".epoch-" not in p.name), key=lambda p: p.stat().st_mtime)
    for path in periodic[:-keep]:
        path.unlink()
        Path(str(path) + ".receipt.json").unlink(missing_ok=True)


def run(args):
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    model_config, training = configure(args.stage, recipe)
    source = args.source_dir.resolve() / recipe["data"][args.stage]["filename"]
    prepared = args.prepared_dir.resolve() / args.stage
    report_path = prepared / "audit_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"full-source audit is required: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_file = recipe["data"][args.stage]
    if (report["status"] != "completed" or report["source"]["sha256"] != expected_file["lfs_sha256"]
            or report["source"]["actual_sha256"] != expected_file["lfs_sha256"]):
        raise ValueError("full-source audit identity does not match the pinned source")
    split = "train"
    offsets = prepared / f"{split}.offsets.u64"
    valid_offsets = prepared / "validation.offsets.u64"
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/minimind_tokenizer", local_files_only=True)
    dataset = OffsetJSONLDataset(source, offsets, args.stage, tokenizer, training["max_length"])
    validation = OffsetJSONLDataset(source, valid_offsets, args.stage, tokenizer, training["max_length"])
    if args.smoke_records:
        if args.smoke_records <= 0:
            raise ValueError("smoke_records must be positive")
        dataset.limit_records = min(args.smoke_records, len(dataset))
        validation.limit_records = min(32, len(validation))
    records = len(dataset)
    if records <= 0 or len(validation) <= 0:
        raise ValueError("train and validation data must both contain eligible examples")

    effective = training["effective_batch_records"]
    updates_per_epoch = math.ceil(records / effective)
    total_updates = updates_per_epoch * training["epochs"]
    micro = training["micro_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    seed = training["seed"]
    identity = {
        "stage": args.stage,
        "recipe_sha256": sha256_file(recipe_path),
        "audit_report_sha256": sha256_file(report_path),
        "source_sha256": expected_file["lfs_sha256"],
        "initial_model_sha256": (sha256_file(args.initial_model)
                                 if args.stage == "sft" and args.initial_model is not None else None),
        "train_offsets_sha256": sha256_file(offsets),
        "validation_offsets_sha256": sha256_file(valid_offsets),
        "max_length": training["max_length"],
        "epochs": training["epochs"],
        "records": records,
        "effective_batch_records": effective,
        "planned_updates": total_updates,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "attention_backend": "forced_torch_sdpa_flash_attention",
        "smoke_records": args.smoke_records,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_path = output / "metrics.jsonl"
    manifest_path = output / "run_manifest.json"
    manifest = {
        "status": "initializing", "training_started": False, "stage": args.stage,
        "identity_sha256": identity_hash, "identity": identity,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable; refusing fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    if not torch.backends.cuda.is_flash_attention_available():
        raise RuntimeError("PyTorch FlashAttention is not compiled; refusing fallback")
    install_flash_backend()
    model = MinimindForCausalLM(model_config).to(device)
    if args.stage == "sft":
        if args.initial_model is None or not Path(args.initial_model).is_file():
            raise FileNotFoundError("SFT requires this experiment's verified pretrain final model")
        initial = torch.load(args.initial_model, map_location="cpu", weights_only=True)
        model.load_state_dict(initial, strict=True)
        del initial
    model_hash_before = state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"], betas=tuple(training["betas"]),
        eps=training["eps"], foreach=False, fused=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.1 + 0.45 * (1.0 + math.cos(math.pi * step / total_updates)),
    )
    collator = PretrainCollator(tokenizer, training["max_length"]) if args.stage == "pretrain" else SFTCollator(tokenizer, training["max_length"])
    collate_fn = checked_collator(collator)
    validation_loader = DataLoader(validation, batch_size=micro, shuffle=False,
                                    collate_fn=collate_fn, num_workers=0)
    autocast_dtype = torch.bfloat16
    sampler = None
    global_step = 0
    start_epoch = 0
    valid_tokens_seen = 0
    if args.resume is not None:
        resume_receipt_path = Path(str(args.resume) + ".receipt.json")
        resume_receipt = json.loads(resume_receipt_path.read_text(encoding="utf-8"))
        if (resume_receipt["checkpoint_sha256"] != sha256_file(args.resume)
                or resume_receipt["identity_sha256"] != identity_hash):
            raise ValueError("resume receipt or identity check failed")
        start_epoch = resume_receipt["epoch"]
        sampler = BlockShuffleSampler(records, seed + start_epoch)
        global_step, metadata = load_checkpoint(
            args.resume, model, optimizer, scheduler, sampler=sampler,
            expected_metadata={"identity_sha256": identity_hash, "stage": args.stage},
            return_metadata=True,
        )
        valid_tokens_seen = metadata["valid_tokens_seen"]
        if metadata["epoch"] != start_epoch:
            raise ValueError("checkpoint epoch does not match its receipt")
        if metadata.get("model_hash") != state_hash(model):
            raise ValueError("resume checkpoint model-state digest mismatch")
    else:
        model_hash_before = state_hash(model)

    manifest.update({
        "status": "running", "training_started": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "initial_model_sha256": model_hash_before,
        "actual_attention_backend": "torch.nn.functional.scaled_dot_product_attention forced with SDPBackend.FLASH_ATTENTION",
        "records": records, "validation_records": len(validation),
        "updates_per_epoch": updates_per_epoch, "planned_updates": total_updates,
    })
    write_json(manifest_path, manifest)
    stop_requested = {"value": False}

    def request_stop(_sig, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_boundary(epoch, current_sampler, status="running", epoch_boundary=False):
        suffix = f".epoch-{epoch + 1}" if epoch_boundary else ""
        checkpoint = checkpoint_dir / f"step-{global_step:09d}{suffix}.pt"
        metadata = {
            "identity_sha256": identity_hash, "stage": args.stage, "epoch": epoch,
            "valid_tokens_seen": valid_tokens_seen, "model_hash": state_hash(model),
        }
        save_checkpoint(checkpoint, model, optimizer, scheduler, global_step,
                        sampler=current_sampler, metadata=metadata)
        receipt = {"checkpoint_sha256": sha256_file(checkpoint), "global_step": global_step,
                   "epoch": epoch, "committed_samples": current_sampler.committed_position,
                   "identity_sha256": identity_hash, "valid_tokens_seen": valid_tokens_seen}
        write_json(Path(str(checkpoint) + ".receipt.json"), receipt)
        prune_checkpoints(checkpoint_dir)
        manifest.update({"status": status, "global_step": global_step,
                         "epoch": epoch, "committed_samples": current_sampler.committed_position,
                         "latest_checkpoint": str(checkpoint), "latest_checkpoint_sha256": receipt["checkpoint_sha256"]})
        write_json(manifest_path, manifest)
        return checkpoint

    try:
        if start_epoch < training["epochs"]:
            if sampler is None:
                sampler = BlockShuffleSampler(records, seed + start_epoch)
            for epoch in range(start_epoch, training["epochs"]):
                if epoch != start_epoch:
                    sampler = BlockShuffleSampler(records, seed + epoch)
                loader = DataLoader(dataset, batch_size=micro, sampler=sampler,
                                    collate_fn=collate_fn, num_workers=0, drop_last=False,
                                    generator=torch.Generator().manual_seed(123 + epoch))
                validation_result = evaluate_nll(model, validation_loader, device=device,
                                                 autocast_dtype=autocast_dtype)
                with log_path.open("a", encoding="utf-8", newline="\n") as log:
                    log.write(json.dumps({"event": "validation", "epoch": epoch,
                                          "global_step": global_step, **validation_result}) + "\n")
                pending = []
                window_start = time.perf_counter()
                for batch in loader:
                    pending.append(batch)
                    if len(pending) < accumulation and sampler.position < records:
                        continue
                    result = train_update(model, optimizer, pending, scheduler,
                                          training["max_grad_norm"], device=device,
                                          autocast_dtype=autocast_dtype)
                    committed = sum(item["input_ids"].shape[0] for item in pending)
                    sampler.commit(committed)
                    pending = []
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    global_step += 1
                    valid_tokens_seen += result["valid_tokens"]
                    row = {"event": "update", "stage": args.stage, "epoch": epoch,
                           "global_step": global_step, **result,
                           "committed_samples": sampler.committed_position,
                           "seconds_since_previous_update": time.perf_counter() - window_start,
                           "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                           "cuda_reserved_bytes": torch.cuda.memory_reserved(device)}
                    with log_path.open("a", encoding="utf-8", newline="\n") as log:
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                    window_start = time.perf_counter()
                    if global_step % recipe["checkpoint"]["interval_updates"] == 0:
                        save_boundary(epoch, sampler)
                    elif global_step % 10 == 0:
                        manifest.update({"status": "running", "global_step": global_step,
                                         "epoch": epoch, "committed_samples": sampler.committed_position})
                        write_json(manifest_path, manifest)
                    if args.stop_after_updates is not None and global_step >= args.stop_after_updates:
                        checkpoint = save_boundary(epoch, sampler, "paused")
                        return {"status": "paused", "global_step": global_step,
                                "checkpoint": str(checkpoint), "identity_sha256": identity_hash}
                    if stop_requested["value"]:
                        checkpoint = save_boundary(epoch, sampler, "paused")
                        return {"status": "paused", "global_step": global_step,
                                "checkpoint": str(checkpoint), "identity_sha256": identity_hash}
                if pending:
                    raise RuntimeError("uncommitted microbatches remain after epoch iterator")
                save_boundary(epoch, sampler, epoch_boundary=True)
            final_validation = evaluate_nll(model, validation_loader, device=device,
                                            autocast_dtype=autocast_dtype)
        else:
            final_validation = evaluate_nll(model, validation_loader, device=device,
                                            autocast_dtype=autocast_dtype)

        final_model = output / "final_model.pt"
        temp_model = output / "final_model.pt.tmp"
        torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, temp_model)
        temp_model.replace(final_model)
        loaded = MinimindForCausalLM(model_config)
        loaded.load_state_dict(torch.load(final_model, map_location="cpu", weights_only=True), strict=True)
        final_hash = state_hash(loaded)
        if final_hash != state_hash(model):
            raise ValueError("final model strict reload digest mismatch")
        result = {
            "status": "completed", "stage": args.stage,
            "epochs": training["epochs"], "global_step": global_step,
            "planned_updates": total_updates, "train_records_per_epoch": records,
            "validation_records": len(validation), "valid_tokens_seen": valid_tokens_seen,
            "final_validation": final_validation, "final_model": str(final_model),
            "final_model_sha256": sha256_file(final_model), "model_state_sha256": final_hash,
            "identity_sha256": identity_hash, "attention_backend": manifest["actual_attention_backend"],
        }
        write_json(output / "result.json", result)
        manifest.update(result)
        write_json(manifest_path, manifest)
        return result
    except BaseException as exc:
        manifest.update({"status": "failed", "global_step": global_step,
                         "error_type": type(exc).__name__, "error": str(exc)})
        write_json(manifest_path, manifest)
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pretrain", "sft"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-records", type=int)
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
