"""Inspect checkpoint artifacts before selecting a resume point."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import torch


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt$")
RECEIPT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt\.receipt\.json$")
TEMPORARY_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt\.tmp$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "global_step",
    "torch_rng_state",
    "cuda_rng_state_all",
    "sampler_state",
    "metadata",
}
REQUIRED_RECEIPT_KEYS = {
    "contract_sha256",
    "global_step",
    "committed_samples",
    "checkpoint_sha256",
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _record(step: int, checkpoint_path: Path) -> dict:
    return {
        "step": step,
        "checkpoint": str(checkpoint_path.resolve()),
        "receipt": str(Path(str(checkpoint_path) + ".receipt.json").resolve()),
    }


def _reject(record: dict, status: str, reason: str) -> dict:
    return {**record, "status": status, "reason": reason}


def inspect_checkpoint(checkpoint_path, expected_contract_sha256: str) -> dict:
    """Return a strict, read-only resume assessment for one checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint filename: {checkpoint_path.name}")
    step = int(match.group(1))
    record = _record(step, checkpoint_path)
    receipt_path = Path(record["receipt"])
    if not receipt_path.is_file():
        return _reject(record, "missing_receipt", "checkpoint has no receipt")

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _reject(record, "invalid_receipt", f"receipt cannot be read: {type(exc).__name__}")
    if not isinstance(receipt, dict) or not REQUIRED_RECEIPT_KEYS <= receipt.keys():
        return _reject(record, "invalid_receipt", "receipt is missing required fields")
    if type(receipt["global_step"]) is not int or receipt["global_step"] != step:
        return _reject(record, "step_mismatch", "receipt step differs from filename")
    if type(receipt["committed_samples"]) is not int or receipt["committed_samples"] < 0:
        return _reject(record, "invalid_receipt", "committed_samples must be a nonnegative integer")
    if receipt["contract_sha256"] != expected_contract_sha256:
        return _reject(record, "contract_mismatch", "receipt belongs to another training contract")
    if not isinstance(receipt["checkpoint_sha256"], str) or SHA256_PATTERN.fullmatch(receipt["checkpoint_sha256"]) is None:
        return _reject(record, "invalid_receipt", "checkpoint_sha256 is not a lowercase SHA256 digest")

    try:
        actual_sha256 = sha256_file(checkpoint_path)
    except OSError as exc:
        return _reject(record, "unreadable_checkpoint", f"checkpoint cannot be read: {type(exc).__name__}")
    record["checkpoint_sha256"] = actual_sha256
    if actual_sha256 != receipt["checkpoint_sha256"]:
        return _reject(record, "hash_mismatch", "checkpoint bytes differ from receipt")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return _reject(record, "invalid_checkpoint", f"checkpoint cannot be loaded: {type(exc).__name__}")
    if not isinstance(checkpoint, dict) or not REQUIRED_CHECKPOINT_KEYS <= checkpoint.keys():
        return _reject(record, "invalid_checkpoint", "checkpoint is missing required state")
    if checkpoint["global_step"] != step:
        return _reject(record, "step_mismatch", "checkpoint step differs from filename")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict) or metadata.get("contract_sha256") != expected_contract_sha256:
        return _reject(record, "contract_mismatch", "checkpoint metadata belongs to another training contract")
    if not isinstance(metadata.get("initialization_sha256"), str):
        return _reject(record, "invalid_checkpoint", "checkpoint has no initialization identity")
    if type(metadata.get("valid_tokens_seen")) is not int or metadata["valid_tokens_seen"] < 0:
        return _reject(record, "invalid_checkpoint", "valid_tokens_seen must be a nonnegative integer")
    sampler_state = checkpoint["sampler_state"]
    if not isinstance(sampler_state, dict) or "indices" not in sampler_state or "committed_position" not in sampler_state:
        return _reject(record, "invalid_checkpoint", "checkpoint has no resumable sampler state")
    committed = sampler_state["committed_position"]
    if type(committed) is not int or committed != receipt["committed_samples"]:
        return _reject(record, "committed_position_mismatch", "sampler and receipt committed positions differ")
    indices = sampler_state["indices"]
    if not isinstance(indices, list) or committed < 0 or committed > len(indices):
        return _reject(record, "invalid_checkpoint", "sampler state has an invalid committed position")

    return {
        **record,
        "status": "valid",
        "reason": "receipt, contract, hash and serialized training state passed",
        "checkpoint_sha256": actual_sha256,
        "committed_samples": committed,
        "valid_tokens_seen": metadata["valid_tokens_seen"],
        "initialization_sha256": metadata["initialization_sha256"],
    }


def inspect_checkpoint_directory(checkpoint_dir, expected_contract_sha256: str) -> dict:
    """Classify checkpoint artifacts and select the newest valid resume point."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not isinstance(expected_contract_sha256, str) or SHA256_PATTERN.fullmatch(expected_contract_sha256) is None:
        raise ValueError("expected contract identity must be a lowercase SHA256 digest")

    checkpoints = {}
    receipts = {}
    temporary = []
    unrecognized = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        if match := CHECKPOINT_PATTERN.fullmatch(path.name):
            checkpoints[int(match.group(1))] = path
        elif match := RECEIPT_PATTERN.fullmatch(path.name):
            receipts[int(match.group(1))] = path
        elif match := TEMPORARY_PATTERN.fullmatch(path.name):
            temporary.append({"step": int(match.group(1)), "path": str(path.resolve()), "status": "orphan_temporary"})
        elif path.name.startswith("checkpoint-"):
            unrecognized.append({"path": str(path.resolve()), "status": "unrecognized_checkpoint_artifact"})

    records = [inspect_checkpoint(checkpoints[step], expected_contract_sha256) for step in sorted(checkpoints)]
    for step in sorted(receipts.keys() - checkpoints.keys()):
        records.append({
            "step": step,
            "checkpoint": str((checkpoint_dir / f"checkpoint-{step}.pt").resolve()),
            "receipt": str(receipts[step].resolve()),
            "status": "receipt_without_checkpoint",
            "reason": "receipt exists but checkpoint is missing",
        })
    records.sort(key=lambda row: (row["step"], row["status"]))
    artifacts = records + sorted(temporary, key=lambda row: row["step"]) + sorted(unrecognized, key=lambda row: row["path"])
    counts = Counter(row["status"] for row in artifacts)
    valid = [row for row in records if row["status"] == "valid"]
    latest = max(valid, key=lambda row: row["step"]) if valid else None
    return {
        "schema_version": 1,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "expected_contract_sha256": expected_contract_sha256,
        "artifact_counts": dict(sorted(counts.items())),
        "latest_valid_checkpoint": latest["checkpoint"] if latest else None,
        "latest_valid_step": latest["step"] if latest else None,
        "artifacts": artifacts,
    }
