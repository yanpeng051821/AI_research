"""Single-device, single-data-pass pretraining with explicit identities and update boundaries."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import PretrainCollator, PretrainDataset
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.checkpoint import load_checkpoint, save_checkpoint
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.step import train_update


@dataclass
class DataSettings:
    split_manifest: str
    split_manifest_sha256: str
    encoding_report: str
    encoding_report_sha256: str
    tokenizer_dir: str
    max_length: int


@dataclass
class TrainingSettings:
    device: str = "cpu"
    precision: str = "fp32"
    seed: int = 42
    sampler_seed: int = 42
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_updates: int | None = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    foreach: bool = False
    fused: bool = False
    warmup_updates: int = 0
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    checkpoint_interval: int = 4
    num_workers: int = 0


@dataclass
class EvaluationSettings:
    batch_size: int = 2
    sample_limit: int | None = 64
    interval: int = 4


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def resolve_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != {"schema_version", "purpose", "data", "model", "training", "evaluation"}:
        raise ValueError("unknown or missing top-level configuration fields")
    if raw["schema_version"] != 1 or raw["purpose"] not in ("engineering_probe", "pretrain"):
        raise ValueError("unsupported configuration version or purpose")
    data = DataSettings(**raw["data"])
    model = ModelConfig(**raw["model"])
    training = TrainingSettings(**raw["training"])
    evaluation = EvaluationSettings(**raw["evaluation"])
    for name in ("micro_batch_size", "gradient_accumulation_steps", "checkpoint_interval"):
        positive_integer(getattr(training, name), name)
    for name in ("batch_size", "interval"):
        positive_integer(getattr(evaluation, name), name)
    if training.max_updates is not None:
        positive_integer(training.max_updates, "max_updates")
    if evaluation.sample_limit is not None:
        positive_integer(evaluation.sample_limit, "sample_limit")
    positive_integer(data.max_length, "max_length")
    if data.max_length < 3 or training.num_workers != 0:
        raise ValueError("require length >= 3 and num_workers=0 in this entry")
    if type(training.warmup_updates) is not int or training.warmup_updates < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    if type(training.seed) is not int or type(training.sampler_seed) is not int:
        raise ValueError("seeds must be integers")
    for value, name in [(training.learning_rate, "learning_rate"), (training.eps, "eps"), (training.max_grad_norm, "max_grad_norm")]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(training.weight_decay) or training.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if len(training.betas) != 2 or any(not 0 <= beta < 1 for beta in training.betas):
        raise ValueError("AdamW betas must be in [0, 1)")
    if type(training.foreach) is not bool or type(training.fused) is not bool:
        raise ValueError("AdamW foreach and fused flags must be booleans")
    if not 0 <= training.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if training.precision not in ("fp32", "bf16") or training.device not in ("cpu", "cuda"):
        raise ValueError("supported devices: cpu/cuda; supported precision: fp32/bf16")
    if training.precision == "bf16" and training.device != "cuda":
        raise ValueError("BF16 probe requires CUDA; CPU reference uses FP32")
    if model.hidden_size % model.num_heads or model.num_heads % model.kv_heads:
        raise ValueError("model head dimensions are inconsistent")
    return {
        "schema_version": 1, "purpose": raw["purpose"], "data": asdict(data),
        "model": asdict(model), "training": asdict(training), "evaluation": asdict(evaluation),
    }


def prepare_contract(config_path, root):
    config = resolve_config(config_path)
    data = config["data"]
    paths = {name: (root / data[name]).resolve() for name in ("split_manifest", "encoding_report", "tokenizer_dir")}
    for name in ("split_manifest", "encoding_report"):
        if sha256_file(paths[name]) != data[name + "_sha256"]:
            raise ValueError(f"{name} identity mismatch")
    split = json.loads(paths["split_manifest"].read_text(encoding="utf-8"))
    audit = json.loads(paths["encoding_report"].read_text(encoding="utf-8"))
    if split["status"] != "text_split_complete_encoding_not_checked" or audit["status"] != "passed":
        raise ValueError("text split and encoding checks must have passed")
    if audit["split_manifest_sha256"] != data["split_manifest_sha256"] or audit["max_length"] != data["max_length"]:
        raise ValueError("encoding report does not match split or length")
    for name, info in split["files"].items():
        path = paths["split_manifest"].parent / info["file"]
        if sha256_file(path) != info["sha256"] or audit["split_files_sha256"][name] != info["sha256"]:
            raise ValueError(f"{name} data identity mismatch")
        paths[name] = path
    for filename, expected in audit["tokenizer_files_sha256"].items():
        if sha256_file(paths["tokenizer_dir"] / filename) != expected:
            raise ValueError(f"tokenizer identity mismatch: {filename}")
    encoder = root / "src/minimind_lab/data/pretrain.py"
    if sha256_file(encoder) != audit["code_sha256"]["dataset_and_collator"]:
        raise ValueError("Dataset/collator changed since encoding audit")
    expected_versions = audit.get("runtime", {})
    for name in ("transformers", "tokenizers"):
        if expected_versions.get(name) != importlib.metadata.version(name):
            raise ValueError(f"tokenizer runtime changed since encoding audit: {name}")
    training = config["training"]
    effective_batch = training["micro_batch_size"] * training["gradient_accumulation_steps"]
    data_updates = math.ceil(split["counts"]["train"] / effective_batch)
    budget = training["max_updates"] if training["max_updates"] is not None else data_updates
    if budget > data_updates or training["warmup_updates"] >= budget:
        raise ValueError("budget exceeds one data pass or warmup covers the entire budget")
    limit = config["evaluation"]["sample_limit"]
    if limit is not None and limit > split["counts"]["validation"]:
        raise ValueError("validation sample_limit exceeds frozen validation set")
    runtime_files = sorted((root / "src/minimind_lab").rglob("*.py")) + [root / "scripts/train_pretrain.py"]
    code = {path.relative_to(root).as_posix(): sha256_file(path) for path in runtime_files}
    data_identity = {
        "split_manifest_sha256": data["split_manifest_sha256"],
        "encoding_report_sha256": data["encoding_report_sha256"], "max_length": data["max_length"],
    }
    identity = {
        "schema_version": 1, "purpose": config["purpose"], "data": data_identity,
        "model": config["model"], "training": training, "evaluation": config["evaluation"],
        "max_updates_resolved": budget, "effective_batch_records": effective_batch,
        "loss_normalization": "sum loss / valid causal targets across the entire accumulation window",
        "optimizer_parameter_policy": "all unique trainable parameters in one AdamW group",
        "scheduler": "optimizer-update indexed warmup then cosine to min_lr_ratio",
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "datasets")},
        "code_sha256": code,
    }
    contract = {"contract_sha256": canonical_hash(identity), "identity": identity, "resolved_config": config}
    return contract, paths, split


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lr_multiplier(step, budget, warmup, minimum):
    if step < warmup:
        return (step + 1) / warmup
    if warmup == 0:
        progress = step / max(1, budget - 1)
    else:
        progress = (step - warmup + 1) / (budget - warmup)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (1 - minimum) * (1 + math.cos(math.pi * progress))


def memory_snapshot(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "free_device_bytes": free, "total_device_bytes": total,
    }


def run_experiment(
    config_path,
    output_dir,
    root,
    *,
    resume=None,
    stop_after_updates=None,
    prepare_only=False,
    checkpoint_dir=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "preparing", "started_at_utc": datetime.now(timezone.utc).isoformat(), "global_step": 0}
    state_path = output_dir / "run_manifest.json"
    write_json(state_path, manifest)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    metrics = []
    try:
        contract, paths, split = prepare_contract(Path(config_path), root)
        write_json(output_dir / "contract.json", contract)
        write_json(output_dir / "resolved_config.json", contract["resolved_config"])
        manifest["contract_sha256"] = contract["contract_sha256"]
        manifest["purpose"] = contract["identity"]["purpose"]
        if prepare_only:
            manifest["status"] = "prepared_not_trained"
            write_json(state_path, manifest)
            return manifest
        checkpoint_root = output_dir if checkpoint_dir is None else Path(checkpoint_dir).resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        manifest["checkpoint_dir"] = str(checkpoint_root)
        cfg = contract["resolved_config"]
        train = cfg["training"]
        evaluation = cfg["evaluation"]
        device = torch.device(train["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if train["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unsupported on this CUDA device")
        if stop_after_updates is not None:
            positive_integer(stop_after_updates, "stop_after_updates")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = torch.bfloat16 if train["precision"] == "bf16" else None
        tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_dir"], local_files_only=True)
        if len(tokenizer) != cfg["model"]["vocab_size"]:
            raise ValueError("model vocab_size must match tokenizer")
        if {name: getattr(tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")} != json.loads(paths["encoding_report"].read_text(encoding="utf-8"))["special_token_ids"]:
            raise ValueError("tokenizer special-token IDs differ from encoding audit")
        if tokenizer.truncation_side != "right":
            raise ValueError("right truncation required by encoding contract")
        dataset = PretrainDataset(paths["train"], tokenizer, cfg["data"]["max_length"])
        validation = PretrainDataset(paths["validation"], tokenizer, cfg["data"]["max_length"])
        if len(dataset) != split["counts"]["train"] or len(validation) != split["counts"]["validation"]:
            raise ValueError("loaded Dataset counts differ from manifest")
        limit = evaluation["sample_limit"]
        if limit is not None:
            validation = Subset(validation, range(limit))
        sampler = StatefulSampler(len(dataset), train["sampler_seed"])
        collator = PretrainCollator(tokenizer, cfg["data"]["max_length"])
        loader = DataLoader(dataset, batch_size=train["micro_batch_size"], sampler=sampler, collate_fn=collator,
                            num_workers=0, generator=torch.Generator().manual_seed(123))
        validation_loader = DataLoader(validation, batch_size=evaluation["batch_size"], shuffle=False, collate_fn=collator,
                                       num_workers=0, generator=torch.Generator().manual_seed(456))
        torch.manual_seed(train["seed"])
        model = MinimindForCausalLM(ModelConfig(**cfg["model"]))
        initialization_hash = model_state_hash(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"],
                                     betas=tuple(train["betas"]), eps=train["eps"],
                                     foreach=train["foreach"], fused=train["fused"])
        budget = contract["identity"]["max_updates_resolved"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_multiplier(step, budget, train["warmup_updates"], train["min_lr_ratio"]),
        )
        step = 0
        tokens_seen = 0
        if resume is not None:
            receipt_path = Path(str(resume) + ".receipt.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["contract_sha256"] != contract["contract_sha256"] or receipt["checkpoint_sha256"] != sha256_file(resume):
                raise ValueError("resume checkpoint identity mismatch")
            step, metadata = load_checkpoint(
                resume, model, optimizer, scheduler, sampler=sampler,
                expected_metadata={"contract_sha256": contract["contract_sha256"], "initialization_sha256": initialization_hash},
                return_metadata=True,
            )
            tokens_seen = metadata["valid_tokens_seen"]
        target = min(stop_after_updates, budget) if stop_after_updates is not None else budget
        if step > budget or (target <= step and step < budget):
            raise ValueError("stop boundary must be beyond restored progress and within the original plan")
        manifest.update({
            "status": "running", "global_step": step, "resume_from": str(resume) if resume else None,
            "initialization_sha256": initialization_hash,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "validation_records": len(validation),
            "validation_scope": "fixed_prefix_engineering_subset" if limit is not None else "full_frozen_validation",
        })
        write_json(state_path, manifest)

        def log(event):
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event), flush=True)

        def validate(label):
            started = time.perf_counter()
            result = evaluate_nll(model, validation_loader, device=device, autocast_dtype=dtype)
            log({"event": label, "global_step": step, **result, "seconds": time.perf_counter() - started,
                 "records": len(validation), "scope": manifest["validation_scope"]})
            return result

        def checkpoint():
            path = checkpoint_root / f"checkpoint-{step}.pt"
            receipt_path = Path(str(path) + ".receipt.json")
            if path.exists() or receipt_path.exists():
                raise FileExistsError(f"checkpoint destination already exists: {path}")
            started = time.perf_counter()
            save_checkpoint(path, model, optimizer, scheduler, step, sampler=sampler, metadata={
                "contract_sha256": contract["contract_sha256"], "initialization_sha256": initialization_hash,
                "valid_tokens_seen": tokens_seen,
            })
            receipt = {"contract_sha256": contract["contract_sha256"], "global_step": step,
                       "committed_samples": sampler.committed_position, "checkpoint_sha256": sha256_file(path)}
            write_json(receipt_path, receipt)
            manifest["last_checkpoint"] = path.name
            write_json(state_path, manifest)
            log({"event": "checkpoint_saved", **receipt, "file": path.name,
                 "path": str(path), "seconds": time.perf_counter() - started})

        baseline = validate("initial_validation" if resume is None else "resume_validation")
        iterator = iter(loader)
        last_saved = None
        while step < target:
            window_start = time.perf_counter()
            window = []
            for _ in range(train["gradient_accumulation_steps"]):
                batch = next(iterator, None)
                if batch is None:
                    break
                window.append(batch)
            if not window:
                raise RuntimeError("data exhausted before the frozen update budget")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = train_update(model, optimizer, window, scheduler, train["max_grad_norm"], device=device, autocast_dtype=dtype)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - started
            sampler.commit(sum(batch["input_ids"].shape[0] for batch in window))
            step += 1
            tokens_seen += result["valid_tokens"]
            optimizer.zero_grad(set_to_none=True)
            record = {"event": "update", "global_step": step, **result, "update_seconds": update_seconds,
                      "window_seconds": time.perf_counter() - window_start,
                      "valid_tokens_per_update_second": result["valid_tokens"] / update_seconds,
                      "committed_samples": sampler.committed_position, "memory": memory_snapshot(device)}
            metrics.append(record)
            log(record)
            manifest.update({"global_step": step, "committed_samples": sampler.committed_position, "valid_tokens_seen": tokens_seen})
            write_json(state_path, manifest)
            if step % train["checkpoint_interval"] == 0 or step == target:
                checkpoint()
                last_saved = step
            if step % evaluation["interval"] == 0 and step != target:
                validate("periodic_validation")
        if last_saved != step:
            checkpoint()
        final_validation = validate("final_validation")
        manifest["status"] = "completed" if step == budget else "paused"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report = {
            **manifest, "updates_this_launch": len(metrics), "initial_or_resume_validation": baseline,
            "final_validation": final_validation,
            "median_update_seconds_excluding_first": statistics.median([row["update_seconds"] for row in metrics[1:] or metrics]) if metrics else None,
            "peak_allocated_bytes_this_launch": max((row["memory"].get("peak_allocated_bytes", 0) for row in metrics), default=0),
            "limitations": ["Engineering probe and validation subset do not establish full-data convergence or natural-language ability."],
        }
        write_json(output_dir / "result.json", report)
        write_json(state_path, manifest)
        return report
    except BaseException as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(state_path, manifest)
        raise
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
