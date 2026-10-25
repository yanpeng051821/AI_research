"""Verify a frozen recipe before delegating to the tested pretraining entry."""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.experiment import (
    canonical_hash,
    model_state_hash,
    prepare_contract,
    run_experiment,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def environment_identity():
    driver = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        driver = result.stdout.strip().splitlines()
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "cuda_build": torch.version.cuda,
        "gpu_name": gpu,
        "gpu_driver_versions": driver,
    }


def initialization_identity(config):
    # Only inspect a CPU initialization; leave the caller's random stream intact.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["training"]["seed"])
        model = MinimindForCausalLM(ModelConfig(**config["model"]))
        return model_state_hash(model), sum(p.numel() for p in model.parameters())


def verify_frozen_recipe(freeze_path, root=ROOT):
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["schema_version"] != 1 or freeze["status"] != "frozen":
        raise ValueError("recipe must be a version-1 frozen recipe")
    paths = {}
    for name, artifact in freeze["files"].items():
        path = (freeze_path.parent / artifact["file"]).resolve()
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact identity mismatch: {name}")
        paths[name] = path
    if paths["launcher"] != Path(__file__).resolve():
        raise ValueError("frozen launcher is not the currently executing launcher")
    saved = json.loads(paths["resolved_contract"].read_text(encoding="utf-8"))
    if canonical_hash(saved["identity"]) != saved["contract_sha256"]:
        raise ValueError("saved contract identity is internally inconsistent")
    current, _, _ = prepare_contract(paths["training_config"], Path(root))
    expected = freeze["expected"]
    if current != saved or current["contract_sha256"] != expected["contract_sha256"]:
        raise ValueError("current runtime/configuration differs from frozen contract")
    if environment_identity() != freeze["environment"]:
        raise ValueError("device/driver/Python environment differs from frozen recipe")
    initialization_hash, parameter_count = initialization_identity(current["resolved_config"])
    if initialization_hash != expected["initialization_sha256"]:
        raise ValueError("initialization differs from frozen recipe")
    if parameter_count != expected["model_parameters"]:
        raise ValueError("parameter count differs from frozen recipe")
    return {
        "status": "frozen_recipe_verified_not_trained",
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": sha256_file(freeze_path),
        "contract_sha256": current["contract_sha256"],
        "initialization_sha256": initialization_hash,
        "model_parameters": parameter_count,
        "max_updates": current["identity"]["max_updates_resolved"],
        "files_verified": len(paths),
        "training_config": str(paths["training_config"]),
    }, freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.check_only and (args.output_dir is not None or args.resume is not None):
        parser.error("output/resume controls require --run")
    report, freeze = verify_frozen_recipe(args.freeze)
    report["mode"] = "run" if args.run else "check_only"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        return
    output = args.output_dir or (ROOT / freeze["storage"]["initial_run_dir"])
    if args.resume is not None and args.output_dir is None:
        parser.error("resume requires a new --output-dir")
    result = run_experiment(
        report["training_config"], output, ROOT,
        checkpoint_dir=freeze["storage"]["checkpoint_dir"], resume=args.resume,
    )
    print(json.dumps({"status": result["status"], "global_step": result["global_step"]}, indent=2))


if __name__ == "__main__":
    main()
