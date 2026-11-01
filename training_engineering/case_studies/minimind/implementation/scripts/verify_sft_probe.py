"""Check saved probe evidence and checkpoint metadata; this is not a resume run."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, output):
    if output.exists():
        raise FileExistsError(output)
    paths = [directory / name for name in ("result.json", "run_manifest.json",
                                          "candidate_contract.json", "metrics.jsonl")]
    result, manifest, contract = [json.loads(p.read_text(encoding="utf-8")) for p in paths[:3]]
    checkpoint = Path(result["checkpoint"])
    receipt_path = Path(str(checkpoint) + ".receipt.json")
    paths.extend((checkpoint, receipt_path, Path(__file__)))
    identities = {str(p.resolve()): sha(p) for p in paths}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    events = [json.loads(line) for line in paths[3].read_text(encoding="utf-8").splitlines()]
    updates = [row for row in events if row["event"] == "update"]
    first = updates[0]["global_step"]
    last = result["global_step"]
    require(result["status"] == manifest["status"] == "paused", "expected a paused probe")
    require([u["global_step"] for u in updates] == list(range(first, last + 1)), "update coverage mismatch")
    require(receipt["checkpoint_sha256"] == identities[str(checkpoint.resolve())], "checkpoint hash mismatch")
    require(saved["global_step"] == receipt["global_step"] == last, "checkpoint step mismatch")
    require(saved["scheduler"]["last_epoch"] == last, "scheduler progress mismatch")
    require(saved["metadata"]["contract_sha256"] == receipt["contract_sha256"]
            == result["contract_sha256"] == contract["contract_sha256"], "contract mismatch")
    require(saved["metadata"]["initial_model_state_sha256"]
            == contract["identity"]["candidate"]["initial_model_state_sha256"], "B0 metadata mismatch")
    require(saved["sampler_state"]["committed_position"] == receipt["committed_samples"]
            == result["committed_samples"] == updates[-1]["committed_samples"], "data position mismatch")
    require(len(saved["sampler_state"]["indices"]) == contract["identity"]["train_records"], "sample count mismatch")
    order = json.dumps(saved["sampler_state"]["indices"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(hashlib.sha256(order.encode()).hexdigest() == contract["identity"]["sample_order_sha256"], "order mismatch")
    require(all(int(s["step"].item()) == last for s in saved["optimizer"]["state"].values()), "Adam step mismatch")
    require(saved["metadata"]["valid_tokens_seen"] == result["valid_tokens_seen"], "token count mismatch")
    if first == 1:
        require(sum(u["valid_tokens"] for u in updates) == result["valid_tokens_seen"], "log token sum mismatch")
    require(all(math.isfinite(u[k]) for u in updates for k in ("loss", "grad_norm", "lr_used")), "nonfinite metric")
    steady = updates[1:] or updates
    median = statistics.median(u["window_seconds"] for u in steady)
    report = {
        "status": "passed_saved_evidence_checks_not_resume_test", "inputs_sha256": identities,
        "global_step": last, "updates_this_launch": len(updates),
        "committed_samples": result["committed_samples"], "valid_tokens_seen": result["valid_tokens_seen"],
        "checkpoint_bytes": checkpoint.stat().st_size,
        "median_update_seconds_excluding_first": statistics.median(u["update_seconds"] for u in steady),
        "median_window_seconds_excluding_first": median,
        "rough_pure_training_hours": median * contract["identity"]["planned_updates"] / 3600,
        "estimate_excludes": ["validation", "checkpoint", "startup", "long_run_throughput_variation"],
        "peak_allocated_gib": max(u["memory"]["peak_allocated_bytes"] for u in updates) / 1024**3,
        "last_reserved_gib": updates[-1]["memory"]["reserved_bytes"] / 1024**3,
        "last_allocated_gib": updates[-1]["memory"]["allocated_bytes"] / 1024**3,
        "initial_validation": result["initial_or_resume_validation"],
        "final_validation": result["final_validation"],
        "actual_resume_test_completed": False,
    }
    require(all(sha(Path(p)) == h for p, h in identities.items()), "inputs changed during verification")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())
