"""Archive completed-run evidence without copying datasets or checkpoint tensors."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from minimind_lab.training.recovery import inspect_checkpoint
from train_frozen_pretrain import ROOT, verify_frozen_recipe


SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{25,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(path, relative_to):
    return {"path": path.relative_to(relative_to).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_run(run, process, contract, expected):
    result = read_json(run / "result.json")
    state = read_json(run / "run_manifest.json")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    updates = [r for r in rows if r["event"] == "update"]
    validations = [r for r in rows if r["event"] in {
        "initial_validation", "periodic_validation", "final_validation",
    }]
    saves = [r for r in rows if r["event"] == "checkpoint_saved"]
    budget = expected["max_updates"]
    if result["status"] != "completed" or any(result.get(k) != v for k, v in state.items()):
        raise ValueError("run/result are incomplete or inconsistent")
    for name in ("contract_sha256", "initialization_sha256", "model_parameters"):
        if result[name] != expected[name]:
            raise ValueError(f"run identity mismatch: {name}")
    if (result["global_step"] != budget or result["committed_samples"] != expected["train_records"]
            or result["valid_tokens_seen"] != expected["train_valid_targets"]
            or result["validation_records"] != expected["validation_records"]):
        raise ValueError("run budget differs from frozen recipe")
    if result["resume_from"] is not None:
        raise ValueError("this archive expects one uninterrupted formal run")
    if [r["global_step"] for r in updates] != list(range(1, budget + 1)):
        raise ValueError("incomplete update trace")
    if sum(r["valid_tokens"] for r in updates) != result["valid_tokens_seen"]:
        raise ValueError("target total differs from result")
    if any(not math.isfinite(r[name]) for r in updates for name in ("loss", "grad_norm")):
        raise ValueError("non-finite update values")
    config = contract["resolved_config"]
    effective = contract["identity"]["effective_batch_records"]
    for row in updates:
        if row["committed_samples"] != min(row["global_step"] * effective, expected["train_records"]):
            raise ValueError("committed sample trajectory differs from one epoch")
    expected_validation = [0, *range(config["evaluation"]["interval"], budget,
                                     config["evaluation"]["interval"]), budget]
    if [r["global_step"] for r in validations] != expected_validation:
        raise ValueError("validation schedule differs from frozen recipe")
    for row in validations:
        if (row["records"] != expected["validation_records"]
                or row["valid_tokens"] != expected["validation_valid_targets"]
                or row["scope"] != "full_frozen_validation" or not math.isfinite(row["nll"])):
            raise ValueError("validation scope or values differ")
    if (validations[0]["nll"] != result["initial_or_resume_validation"]["nll"]
            or validations[-1]["nll"] != result["final_validation"]["nll"]):
        raise ValueError("validation trace differs from result")
    interval = config["training"]["checkpoint_interval"]
    if [r["global_step"] for r in saves] != [*range(interval, budget, interval), budget]:
        raise ValueError("checkpoint schedule differs from frozen recipe")
    process_state = read_json(process / "process.json")
    if process_state["status"] != "exited_successfully" or process_state["exit_code"] != 0:
        raise ValueError("supervised process did not exit successfully")
    if (process / "stderr.log").stat().st_size:
        raise ValueError("nonempty stderr requires explicit review before archiving")
    return result, saves


def checkpoint_inventory(result, saves):
    directory = Path(result["checkpoint_dir"])
    checkpoints = {p.name: p for p in directory.glob("checkpoint-*.pt")}
    receipts = {p.name: p for p in directory.glob("checkpoint-*.pt.receipt.json")}
    if set(checkpoints) != {r["file"] for r in saves} or set(receipts) != {
        r["file"] + ".receipt.json" for r in saves
    } or list(directory.glob("*.tmp")):
        raise ValueError("checkpoint files differ from saved-event inventory")
    records = []
    for row in saves:
        receipt_path = receipts[row["file"] + ".receipt.json"]
        receipt = read_json(receipt_path)
        for key in ("global_step", "committed_samples", "checkpoint_sha256", "contract_sha256"):
            if receipt[key] != row[key]:
                raise ValueError("checkpoint receipt differs from saved event")
        records.append({"file": row["file"], "bytes": checkpoints[row["file"]].stat().st_size,
                        "receipt_sha256": sha256_file(receipt_path), **receipt,
                        "checkpoint_hash_scope": "declared_by_receipt_not_recomputed_in_inventory"})
    return {"checkpoint_directory": str(directory), "checkpoint_count": len(records),
            "total_checkpoint_bytes": sum(r["bytes"] for r in records), "records": records,
            "scope": "Files, sizes and receipt/event agreement; only final checkpoint is read and hashed separately."}


def git_reference(contract):
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    for name in contract["identity"]["code_sha256"]:
        path = ROOT / name
        relative = path.relative_to(repo).as_posix()
        current = subprocess.check_output(["git", "hash-object", "--no-filters", "--", relative], cwd=repo, text=True).strip()
        committed = subprocess.check_output(["git", "rev-parse", f"{head}:{relative}"], cwd=repo, text=True).strip()
        if current != committed:
            raise ValueError("frozen runtime source is not represented by archive-time HEAD")
    return {"head_at_archiving": head, "frozen_runtime_sources_match_head_bytes": True,
            "scope": "This commit contains the verified runtime sources; it is not asserted to be launch-time HEAD."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--process-dir", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new archive directory; do not overwrite evidence")
    freeze_path = args.experiment_dir / "training_recipe/training_freeze.json"
    postflight, freeze = verify_frozen_recipe(freeze_path)
    contract = read_json(args.run_dir / "contract.json")
    if contract != read_json(args.experiment_dir / "training_recipe/contract.json"):
        raise ValueError("run contract differs from frozen contract")
    if read_json(args.run_dir / "resolved_config.json") != contract["resolved_config"]:
        raise ValueError("run configuration differs from contract")
    launch = read_json(args.launch_receipt)
    if launch["freeze_sha256"] != postflight["freeze_sha256"] or launch["mode"] != "run":
        raise ValueError("launch receipt differs from verified recipe")
    result, saves = check_run(args.run_dir, args.process_dir, contract, freeze["expected"])
    inventory = checkpoint_inventory(result, saves)
    final_path = Path(result["checkpoint_dir"]) / result["last_checkpoint"]
    inspection = inspect_checkpoint(final_path, result["contract_sha256"])
    if (inspection["status"] != "valid" or inspection["valid_tokens_seen"] != result["valid_tokens_seen"]
            or inspection["initialization_sha256"] != result["initialization_sha256"]):
        raise ValueError("final checkpoint did not pass independent inspection")
    reference = git_reference(contract)
    related = []
    for name in ("learning_curves_v1/curve_summary.json", "runtime_cost_v1.json"):
        path = args.experiment_dir / "results" / name
        report = read_json(path)
        if (report["source"]["metrics_sha256"] != sha256_file(args.run_dir / "metrics.jsonl")
                or report["source"]["result_sha256"] != sha256_file(args.run_dir / "result.json")):
            raise ValueError("analysis refers to other run evidence")
        script = ROOT / "scripts" / ("analyze_pretrain_curves.py" if name.startswith("learning") else "analyze_pretrain_cost.py")
        if report["source"]["analysis_script_sha256"] != sha256_file(script):
            raise ValueError("analysis script differs from recorded identity")
        related.append(describe(path, args.experiment_dir / "results"))
    plot = args.experiment_dir / "results/learning_curves_v1/learning_curves.png"
    if sha256_file(plot) != read_json(args.experiment_dir / "results/learning_curves_v1/curve_summary.json")["plot_sha256"]:
        raise ValueError("curve image identity mismatch")
    related.append(describe(plot, args.experiment_dir / "results"))
    copies = {f"run/{name}": args.run_dir / name for name in (
        "contract.json", "resolved_config.json", "run_manifest.json", "metrics.jsonl", "result.json",
    )}
    copies.update({f"process/{name}": args.process_dir / name for name in (
        "process.json", "stdout.log", "stderr.log",
    )})
    copies["frozen_launch.json"] = args.launch_receipt
    copies[final_path.name + ".receipt.json"] = Path(str(final_path) + ".receipt.json")
    for path in copies.values():
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"credential pattern requires review: {path.name}")
    args.output_dir.mkdir(parents=True)
    for name, path in copies.items():
        target = args.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256_file(path)
        shutil.copyfile(path, target)
        if before != sha256_file(target) or before != sha256_file(path):
            raise ValueError("source changed or archived copy differs")
    write_json(args.output_dir / "postflight_verification.json", {
        "context": "after_completed_training_without_parameter_updates", **postflight,
    })
    write_json(args.output_dir / "final_checkpoint_inspection.json", inspection)
    write_json(args.output_dir / "checkpoint_inventory.json", inventory)
    manifest = {
        "schema_version": 1, "status": "evidence_archived_final_checkpoint_checked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_dir.name,
        "contract_sha256": result["contract_sha256"], "freeze_sha256": postflight["freeze_sha256"],
        "archive_script_sha256": sha256_file(Path(__file__)), "code_reference": reference,
        "files": [describe(path, args.output_dir) for path in sorted(args.output_dir.rglob("*")) if path.is_file()],
        "related_results_relative_to_results_directory": related,
        "acceptance": {"engineering_budget_and_evidence": "passed",
                       "validation_nll_improved": result["final_validation"]["nll"] < result["initial_or_resume_validation"]["nll"],
                       "independent_test_and_downstream_ability": "not_evaluated"},
        "backup_and_release": {
            "checkpoint_location": result["checkpoint_dir"],
            "checkpoint_tensors_copied_into_git_archive": False,
            "independent_checkpoint_backup_verified": False,
            "datasets_copied_into_git_archive": False,
            "dataset_identity_and_rebuild": "See data_recipe and the fixed-source preparation scripts.",
            "hf_model_export_and_release": "pending",
            "git_commit_containing_this_archive": "pending_at_generation",
        },
        "limitations": ["Hashes and Git provenance are not cryptographic signatures.",
                        "Archive copies retain original local paths; prepare a separate reviewed public release if needed.",
                        "A failed archive may leave a partial directory; require a complete, hash-verified manifest.",
                        "Only the final checkpoint was read and hashed; other points received inventory/receipt checks."],
    }
    write_json(args.output_dir / "archive_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "archive": str(args.output_dir),
                      "files": len(manifest["files"]), "bytes": sum(r["bytes"] for r in manifest["files"]),
                      "engineering": manifest["acceptance"], "backup_and_release": manifest["backup_and_release"]}, indent=2))


if __name__ == "__main__":
    main()
