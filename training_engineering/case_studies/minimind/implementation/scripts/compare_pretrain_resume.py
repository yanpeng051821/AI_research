"""Compare a continuous probe to a pause/resume probe without trusting the printed loss alone."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import torch


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_tree(left, right, path="checkpoint"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"tensor metadata mismatch at {path}")
        if not torch.equal(left, right):
            raise ValueError(f"tensor values differ at {path}")
        return 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"dictionary keys differ at {path}")
        return sum(compare_tree(value, right[key], f"{path}.{key}") for key, value in left.items())
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"sequence differs at {path}")
        return sum(compare_tree(a, b, f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right)))
    if left != right:
        raise ValueError(f"value differs at {path}")
    return 0


def read_updates(directory):
    return [row for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))["event"] == "update"]


def checkpoint_path(directory, result):
    root = Path(result.get("checkpoint_dir", directory))
    return root / result["last_checkpoint"]


def compare_runs(continuous, paused, resumed):
    a = json.loads((continuous / "result.json").read_text(encoding="utf-8"))
    b = json.loads((resumed / "result.json").read_text(encoding="utf-8"))
    a_path = checkpoint_path(continuous, a)
    b_path = checkpoint_path(resumed, b)
    left = torch.load(a_path, map_location="cpu", weights_only=True)
    right = torch.load(b_path, map_location="cpu", weights_only=True)
    tensors = compare_tree(left, right)
    direct = read_updates(continuous)
    resumed_updates = read_updates(paused) + read_updates(resumed)
    fields = ["global_step", "loss", "valid_tokens", "micro_batches", "grad_norm", "lr_used", "next_lr", "committed_samples"]
    if len(direct) != len(resumed_updates):
        raise ValueError("update trace lengths differ")
    for a_row, b_row in zip(direct, resumed_updates):
        for field in fields:
            if a_row[field] != b_row[field]:
                raise ValueError(f"update trace mismatch: {field}")
    if a["final_validation"] != b["final_validation"] or a["contract_sha256"] != b["contract_sha256"]:
        raise ValueError("validation or contract differs")
    return {
        "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same_machine_same_contract_pause_resume_exact_equivalence_not_cross_hardware_guarantee",
        "contract_sha256": a["contract_sha256"],
        "updates_compared": len(direct),
        "global_step": left["global_step"], "committed_samples": left["sampler_state"]["committed_position"],
        "valid_tokens_seen": left["metadata"]["valid_tokens_seen"],
        "exactly_equal_tensor_leaves": tensors,
        "checked_state": ["model", "optimizer", "scheduler", "sampler", "torch_rng", "cuda_rng", "metadata"],
        "exact_update_fields": fields,
        "continuous_checkpoint_sha256": sha256_file(a_path),
        "resumed_checkpoint_sha256": sha256_file(b_path),
        "parameter_dtypes": sorted({str(value.dtype) for value in left["model"].values()}),
        "continuous_steady_median_update_seconds": statistics.median(row["update_seconds"] for row in direct[1:]),
        "continuous_peak_allocated_gib": max(row["memory"]["peak_allocated_bytes"] for row in direct) / 1024**3,
        "validation_scope": a["validation_scope"],
        "initial_validation": a["initial_or_resume_validation"],
        "final_validation": a["final_validation"],
        "comparison_script_sha256": sha256_file(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--paused", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new comparison report")
    report = compare_runs(args.continuous, args.paused, args.resumed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
