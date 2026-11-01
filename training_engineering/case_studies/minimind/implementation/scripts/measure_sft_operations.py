"""Measure full engineering validation and flushed checkpoint-copy cost; no updates."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader

from evaluate_sft import load_weights
from freeze_sft_data_recipe import check as check_data_freeze
from minimind_lab.data.sft import SFTCollator, SFTDataset
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.experiment import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def run(run_dir, destination, output):
    if output.exists() or destination.exists():
        raise FileExistsError("measurement outputs must be new")
    contract_path = run_dir / "candidate_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cfg = contract["identity"]["candidate"]
    marker_path = (ROOT / cfg["data_freeze"]).resolve()
    checked = check_data_freeze(marker_path)
    if checked["data_contract_sha256"] != cfg["data_contract_sha256"]:
        raise ValueError("data version mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    data = json.loads(Path(marker["files"]["contract"]["path"]).read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    checkpoint = Path(result["checkpoint"])
    receipt = json.loads(Path(str(checkpoint) + ".receipt.json").read_text(encoding="utf-8"))
    digest = sha256_file(checkpoint)
    if receipt["checkpoint_sha256"] != digest:
        raise ValueError("source checkpoint identity mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with checkpoint.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    copy_seconds = time.perf_counter() - started
    if sha256_file(destination) != digest:
        raise ValueError("checkpoint copy mismatch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, tokenizer, _, _ = load_weights((ROOT / cfg["initial_bundle"]).resolve(), data, torch.device("cuda"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["metadata"]["contract_sha256"] != contract["contract_sha256"]:
            raise ValueError("checkpoint contract mismatch")
        model.load_state_dict(state["model"], strict=True)
        del state
        dataset = SFTDataset(data["artifacts"]["validation"]["path"], tokenizer)
        loader = DataLoader(dataset, batch_size=4, collate_fn=SFTCollator(tokenizer, data["encoding"]["max_length"]),
                            num_workers=0, generator=torch.Generator().manual_seed(456))
        torch.cuda.synchronize()
        started = time.perf_counter()
        validation = evaluate_nll(model, loader, device=torch.device("cuda"), autocast_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
    if len(dataset) != checked["validation_records"] or validation["valid_tokens"] != data["split_statistics"]["validation"]["total_targets_after_shift"]:
        raise ValueError("full validation coverage mismatch")
    report = {"status": "passed", "source_checkpoint_sha256": digest,
              "source_contract_sha256": contract["contract_sha256"], "source_run_dir": str(run_dir),
              "checkpoint_copy": str(destination), "checkpoint_bytes": checkpoint.stat().st_size,
              "copy_flush_seconds": copy_seconds, "copy_sha256_verified": True,
              "validation": {"records": len(dataset), "batch_size": 4, "precision": "bf16",
                             "seconds": seconds, **validation},
              "parameter_updates": 0, "script_sha256": sha256_file(Path(__file__)),
              "limitations": ["Flushed disk copy does not include GPU-to-CPU serialization overhead.",
                              "Step20 BF16 engineering validation is not final S1 FP32 evaluation."]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "checkpoint-copy", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.checkpoint_copy.resolve(), args.output.resolve())
