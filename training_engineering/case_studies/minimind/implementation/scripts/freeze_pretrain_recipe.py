"""Bind an accepted recipe to its data contract, runtime identity and probe evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from minimind_lab.training.experiment import prepare_contract, sha256_file
from train_frozen_pretrain import environment_identity, initialization_identity


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def freeze_recipe(recipe_dir, data_freeze_path, comparison_path, stability_path, checkpoint_dir):
    recipe_dir = Path(recipe_dir).resolve()
    targets = [recipe_dir / "contract.json", recipe_dir / "training_freeze.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("use a new recipe version instead of overwriting frozen artifacts")
    config_path = recipe_dir / "training_config.json"
    contract, paths, split = prepare_contract(config_path, ROOT)
    cfg = contract["resolved_config"]
    if cfg["purpose"] != "pretrain" or cfg["training"]["max_updates"] is not None:
        raise ValueError("this freeze requires a complete one-epoch pretraining plan")
    initialization_hash, parameter_count = initialization_identity(cfg)
    data_freeze_path = Path(data_freeze_path).resolve()
    data_freeze = read_json(data_freeze_path)
    if data_freeze["status"] != "frozen":
        raise ValueError("data contract has not been frozen")
    data_paths = {}
    for name in ("contract", "split_manifest", "encoding_audit"):
        item = data_freeze[name]
        path = data_freeze_path.parent / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"data freeze artifact mismatch: {name}")
        data_paths[name] = path
    if sha256_file(paths["split_manifest"]) != data_freeze["split_manifest"]["sha256"]:
        raise ValueError("training configuration uses a different text split")
    if sha256_file(paths["encoding_report"]) != data_freeze["encoding_audit"]["sha256"]:
        raise ValueError("training configuration uses a different encoding audit")
    audit = read_json(paths["encoding_report"])
    if (audit["split_files_sha256"] != data_freeze["split_files_sha256"]
            or any(audit["tokenizer_files_sha256"].get(name) != digest
                   for name, digest in data_freeze["tokenizer_files_sha256"].items())):
        raise ValueError("data freeze differs from the encoding audit")

    comparison_path, stability_path = Path(comparison_path).resolve(), Path(stability_path).resolve()
    comparison, stability = read_json(comparison_path), read_json(stability_path)
    if comparison["status"] != "passed" or comparison["contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("resume gate did not pass for this runtime contract")
    if (stability["status"] != "paused" or stability["global_step"] < 20
            or stability["contract_sha256"] != contract["contract_sha256"]
            or stability["initialization_sha256"] != initialization_hash):
        raise ValueError("stability evidence does not match this initialization and contract")

    artifacts = {
        "training_config": config_path,
        "training_contract": recipe_dir / "TRAINING_CONTRACT.md",
        "resolved_contract": targets[0],
        "data_freeze": data_freeze_path,
        "data_contract": data_paths["contract"],
        "split_manifest": data_paths["split_manifest"],
        "encoding_audit": data_paths["encoding_audit"],
        "resume_comparison": comparison_path,
        "stability_result": stability_path,
        "launcher": Path(__file__).with_name("train_frozen_pretrain.py"),
        "freeze_builder": Path(__file__),
        "dependency_lock": ROOT / "uv.lock",
        "dependency_specification": ROOT / "pyproject.toml",
    }
    # Check all external inputs before publishing the final freeze marker.
    file_entries = {
        name: {"file": Path(os.path.relpath(path, recipe_dir)).as_posix(), "sha256": sha256_file(path)}
        for name, path in artifacts.items() if name != "resolved_contract"
    }
    write_new_json(targets[0], contract)
    file_entries["resolved_contract"] = {"file": "contract.json", "sha256": sha256_file(targets[0])}
    freeze = {
        "schema_version": 1,
        "freeze_id": "pretrain_mini_one_epoch_v1",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_entries,
        "expected": {
            "contract_sha256": contract["contract_sha256"],
            "initialization_sha256": initialization_hash,
            "model_parameters": parameter_count,
            "train_records": split["counts"]["train"],
            "validation_records": split["counts"]["validation"],
            "train_valid_targets": audit["statistics"]["train"]["valid_causal_targets"],
            "validation_valid_targets": audit["statistics"]["validation"]["valid_causal_targets"],
            "max_updates": contract["identity"]["max_updates_resolved"],
            "initial_validation_nll": comparison["initial_validation"]["nll"],
        },
        "storage": {
            "initial_run_dir": "runs/pretrain_mini_one_epoch_v1",
            "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        },
        "environment": environment_identity(),
        "checks": {"resume_gate": "passed", "early_stability_through_step": stability["global_step"]},
    }
    write_new_json(targets[1], freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--data-freeze", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--stability-result", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze = freeze_recipe(args.recipe_dir, args.data_freeze, args.comparison, args.stability_result, args.checkpoint_dir)
    print(json.dumps({"status": freeze["status"], "freeze_id": freeze["freeze_id"], **freeze["expected"]}, indent=2))


if __name__ == "__main__":
    main()
