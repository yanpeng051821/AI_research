"""Summarize completed-run wall time, memory observations, and checkpoint storage."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


GIB = 1024 ** 3


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_seconds(rows, key):
    values = [row[key] for row in rows]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"invalid or missing durations: {key}")
    return values


def timing_summary(rows, result):
    if result["status"] != "completed":
        raise ValueError("cost summary requires a completed run")
    updates = [r for r in rows if r["event"] == "update"]
    validations = [r for r in rows if r["event"] in {
        "initial_validation", "periodic_validation", "final_validation",
    }]
    saves = [r for r in rows if r["event"] == "checkpoint_saved"]
    if [r["global_step"] for r in updates] != list(range(1, result["global_step"] + 1)):
        raise ValueError("incomplete update sequence")
    if (updates[-1]["committed_samples"] != result["committed_samples"]
            or sum(r["valid_tokens"] for r in updates) != result["valid_tokens_seen"]):
        raise ValueError("logged data budget differs from result")
    start = datetime.fromisoformat(result["started_at_utc"])
    finish = datetime.fromisoformat(result["finished_at_utc"])
    elapsed = (finish - start).total_seconds()
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid wall-clock interval")
    windows = positive_seconds(updates, "window_seconds")
    update_times = positive_seconds(updates, "update_seconds")
    validation_times = positive_seconds(validations, "seconds")
    save_times = positive_seconds(saves, "seconds")
    if any(a + 1e-6 < b for a, b in zip(windows, update_times)):
        raise ValueError("update duration must be contained in its window")
    components = {
        "training_windows": sum(windows),
        "validation": sum(validation_times),
        "checkpoint_save_hash_receipt": sum(save_times),
    }
    residual = elapsed - sum(components.values())
    if residual < -0.01:
        raise ValueError("timed components exceed run wall time")
    components["unattributed_run_overhead"] = residual
    return {
        "wall_seconds": elapsed,
        "wall_hours": elapsed / 3600,
        "components": {
            name: {"seconds": seconds, "percent_of_wall": seconds / elapsed * 100}
            for name, seconds in components.items()
        },
        "training_window_detail": {
            "updates": len(updates),
            "update_path_seconds": sum(update_times),
            "other_window_seconds": sum(windows) - sum(update_times),
            "median_update_seconds_excluding_first": statistics.median(update_times[1:] or update_times),
        },
        "validation": {
            "events": len(validations), "mean_seconds": statistics.mean(validation_times),
            "min_seconds": min(validation_times), "max_seconds": max(validation_times),
        },
        "checkpoint": {
            "events": len(saves), "mean_seconds": statistics.mean(save_times),
            "min_seconds": min(save_times), "max_seconds": max(save_times),
        },
        "throughput": {
            "training_window_records_per_second": result["committed_samples"] / sum(windows),
            "training_window_targets_per_second": result["valid_tokens_seen"] / sum(windows),
            "end_to_end_records_per_second": result["committed_samples"] / elapsed,
            "end_to_end_targets_per_second": result["valid_tokens_seen"] / elapsed,
        },
        "memory_observations_gib": {
            "max_peak_allocated": max(r["memory"]["peak_allocated_bytes"] for r in updates) / GIB,
            "min_update_end_allocated": min(r["memory"]["allocated_bytes"] for r in updates) / GIB,
            "max_update_end_allocated": max(r["memory"]["allocated_bytes"] for r in updates) / GIB,
            "min_update_end_reserved": min(r["memory"]["reserved_bytes"] for r in updates) / GIB,
            "max_update_end_reserved": max(r["memory"]["reserved_bytes"] for r in updates) / GIB,
            "min_logged_device_free": min(r["memory"]["free_device_bytes"] for r in updates) / GIB,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimated-hours", type=float, default=16.3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new report path; do not overwrite evidence")
    if not math.isfinite(args.estimated_hours) or args.estimated_hours <= 0:
        raise ValueError("estimated hours must be positive")
    metrics_path = args.run_dir / "metrics.jsonl"
    result_path = args.run_dir / "result.json"
    before = {"metrics_sha256": sha256_file(metrics_path), "result_sha256": sha256_file(result_path)}
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = timing_summary(rows, result)
    checkpoint_dir = Path(result["checkpoint_dir"])
    checkpoints = list(checkpoint_dir.glob("checkpoint-*.pt"))
    receipts = list(checkpoint_dir.glob("checkpoint-*.pt.receipt.json"))
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    receipt_bytes = sum(path.stat().st_size for path in receipts)
    summary.update({
        "contract_sha256": result["contract_sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"run_id": args.run_dir.name, **before,
                   "analysis_script_sha256": sha256_file(Path(__file__))},
        "estimate_comparison": {
            "estimated_hours_approximate": args.estimated_hours,
            "actual_minus_estimated_seconds": summary["wall_seconds"] - args.estimated_hours * 3600,
        },
        "storage_snapshot": {
            "checkpoint_files": len(checkpoints), "receipt_files": len(receipts),
            "checkpoint_bytes": checkpoint_bytes, "receipt_bytes": receipt_bytes,
            "total_gib": (checkpoint_bytes + receipt_bytes) / GIB,
            "temporary_files": [p.name for p in checkpoint_dir.glob("*.tmp")],
            "scope": "file counts and sizes; not a content audit of every checkpoint",
        },
        "electricity_or_rental_cost": None,
        "limitations": [
            "Wall interval is the training run; excludes earlier data work, probes, and later release work.",
            "Update path is nested inside training window; do not add both to wall time.",
            "Update time includes host/device transfer, Python dispatch, and CUDA synchronization; not pure GPU kernel time.",
            "Other window time includes batching, synchronization, sampler commit, and gradient cleanup; not pure data I/O time.",
            "Unattributed residual includes setup and logging outside measured regions; no per-operation attribution.",
            "Reserved/free-device memory are update-end snapshots; allocated peak is reset each update.",
            "No integrated power or billing measurements were collected; monetary cost is not estimated.",
        ],
    })
    if before != {"metrics_sha256": sha256_file(metrics_path), "result_sha256": sha256_file(result_path)}:
        raise ValueError("source files changed during analysis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
