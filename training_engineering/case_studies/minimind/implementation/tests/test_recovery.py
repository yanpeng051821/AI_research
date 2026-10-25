import hashlib
import json
from pathlib import Path

import torch

from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.training.checkpoint import save_checkpoint
from minimind_lab.training.recovery import inspect_checkpoint_directory


CONTRACT = "a" * 64
OTHER_CONTRACT = "b" * 64


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_checkpoint(directory, step, contract=CONTRACT, *, write_receipt=True):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sampler = StatefulSampler(8, seed=42)
    iterator = iter(sampler)
    for _ in range(step):
        next(iterator)
    sampler.commit(step)
    path = save_checkpoint(
        directory / f"checkpoint-{step}.pt",
        model,
        optimizer,
        scheduler,
        step,
        sampler=sampler,
        metadata={
            "contract_sha256": contract,
            "initialization_sha256": "c" * 64,
            "valid_tokens_seen": step * 10,
        },
    )
    if write_receipt:
        receipt = {
            "contract_sha256": contract,
            "global_step": step,
            "committed_samples": step,
            "checkpoint_sha256": sha256_file(path),
        }
        Path(str(path) + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_directory_inspection_selects_latest_valid_checkpoint(tmp_path):
    make_checkpoint(tmp_path, 1)
    latest = make_checkpoint(tmp_path, 2)

    report = inspect_checkpoint_directory(tmp_path, CONTRACT)

    assert report["artifact_counts"] == {"valid": 2}
    assert report["latest_valid_step"] == 2
    assert report["latest_valid_checkpoint"] == str(latest.resolve())


def test_directory_inspection_classifies_incomplete_and_foreign_artifacts(tmp_path):
    make_checkpoint(tmp_path, 1, write_receipt=False)
    foreign = make_checkpoint(tmp_path, 2, contract=OTHER_CONTRACT)
    corrupted = make_checkpoint(tmp_path, 3)
    with corrupted.open("ab") as stream:
        stream.write(b"changed")
    (tmp_path / "checkpoint-4.pt.tmp").write_bytes(b"incomplete")
    (tmp_path / "checkpoint-5.pt.receipt.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-latest.pt").write_bytes(b"unknown")

    report = inspect_checkpoint_directory(tmp_path, CONTRACT)
    by_step = {row["step"]: row["status"] for row in report["artifacts"] if "step" in row}

    assert by_step == {
        1: "missing_receipt",
        2: "contract_mismatch",
        3: "hash_mismatch",
        4: "orphan_temporary",
        5: "receipt_without_checkpoint",
    }
    assert report["latest_valid_checkpoint"] is None
    assert report["artifact_counts"] == {
        "contract_mismatch": 1,
        "hash_mismatch": 1,
        "missing_receipt": 1,
        "orphan_temporary": 1,
        "receipt_without_checkpoint": 1,
        "unrecognized_checkpoint_artifact": 1,
    }
    assert foreign.exists()


def test_directory_inspection_rejects_unloadable_checkpoint_with_matching_hash(tmp_path):
    path = tmp_path / "checkpoint-7.pt"
    path.write_bytes(b"not a torch checkpoint")
    receipt = {
        "contract_sha256": CONTRACT,
        "global_step": 7,
        "committed_samples": 7,
        "checkpoint_sha256": sha256_file(path),
    }
    (tmp_path / "checkpoint-7.pt.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    report = inspect_checkpoint_directory(tmp_path, CONTRACT)

    assert report["artifacts"][0]["status"] == "invalid_checkpoint"
    assert report["latest_valid_checkpoint"] is None


def test_directory_inspection_rejects_invalid_contract_identity(tmp_path):
    try:
        inspect_checkpoint_directory(tmp_path, "not-a-sha256")
    except ValueError as exc:
        assert "lowercase SHA256" in str(exc)
    else:
        raise AssertionError("invalid contract identity was accepted")
