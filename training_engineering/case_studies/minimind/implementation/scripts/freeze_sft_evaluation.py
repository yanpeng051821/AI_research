"""Freeze the accepted SFT evaluation inputs before establishing B0."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from freeze_sft_data_recipe import descriptor, write_new
from inspect_sft_eval_draft import inspect, sha
from load_exported_pretrain import verify_bundle


ROOT = Path(__file__).resolve().parents[1]


def check(path):
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker["status"] != "frozen":
        raise ValueError("evaluation is not frozen")
    for entry in marker["files"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"frozen evaluation file changed: {entry['path']}")
    for name, version in marker["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"evaluation package version changed: {name}")
    return {"status": "frozen_evaluation_verified_not_run", "freeze_sha256": sha(path)}


def freeze(directory, bundle, authorized):
    if not authorized:
        raise ValueError("explicit evaluation authorization required")
    names = ["evaluation_config.json", "probes.jsonl", "inspection.frozen.json",
             "evaluation_acceptance.json", "evaluation_freeze.json"]
    if any((directory / name).exists() for name in names):
        raise FileExistsError("do not overwrite a frozen evaluation version")
    draft = directory / "evaluation_config.draft.json"
    probes = directory / "probes.draft.jsonl"
    before = {str(p.resolve()): sha(p) for p in (draft, probes)}
    manifest = verify_bundle(bundle)
    config = json.loads(draft.read_text(encoding="utf-8"))
    config["status"] = "frozen"
    config["generation"]["device_and_precision"] = "cuda_fp32"
    config["scoring"]["task_failure"] = "any check fails"
    config["scoring"]["task_uncertain"] = "no failed checks but at least one uncertain check"
    config["acceptance"]["evaluation_run_authorized"] = True
    config["validation"] = {"records": 2048, "batch_size": 4, "precision": "fp32",
                            "autocast": False, "expected_valid_tokens": 833052,
                            "aggregation": "total_loss_sum / total_valid_tokens"}
    config["runtime"] = {"deterministic_algorithms": True, "tf32": False, "cpu_threads": 1,
                         "cublas_workspace_config": ":4096:8"}
    write_new(directory / names[0], config)
    with (directory / names[1]).open("xb") as stream:
        stream.write(probes.read_bytes())
    inspect(directory / names[1], directory / names[0], directory / names[2])
    acceptance = {"status": "user_authorized_evaluation_only", "authorization": "User instructed: 开始吧",
                  "scope": "Freeze the discussed evaluation settings and run B0; no training authorization.",
                  "draft_inputs_sha256": before, "training_authorized": False,
                  "capability_success_thresholds": "Set after B0 calibration and before SFT; not after seeing S1."}
    write_new(directory / names[3], acceptance)
    paths = [directory / n for n in names[:-1]] + [directory / "EVALUATION_CONTRACT.md", Path(__file__),
             ROOT / "scripts/evaluate_sft.py", ROOT / "scripts/inspect_sft_eval_draft.py",
             ROOT / "scripts/load_exported_pretrain.py", ROOT / "scripts/prepare_sft_recipe.py",
             ROOT / "pyproject.toml", ROOT / "uv.lock", directory.parent / "data_recipe/data_freeze.json",
             bundle / "export_manifest.json", bundle / "config.json", bundle / "model.safetensors"]
    paths += list((ROOT / "src/minimind_lab/models").rglob("*.py"))
    paths += [ROOT / "src/minimind_lab" / name for name in (
        "data/sft.py", "data/sft_protocol.py", "data/validator.py", "training/evaluate.py")]
    files = {str(p.resolve()): descriptor(p) for p in paths}
    if any(sha(Path(p)) != value for p, value in before.items()):
        raise ValueError("draft changed during freeze")
    write_new(directory / names[4], {"status": "frozen", "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "runtime_versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers", "safetensors", "numpy")},
        "baseline": {"model_state_sha256": manifest["model_state_sha256"], "source": manifest["source"]},
        "scope": "Paired evaluation inputs, scoring and execution; capability thresholds pending pre-training calibration.",
        "training_authorized": False})
    return check(directory / names[4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--recipe-dir", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--authorize-evaluation", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
    else:
        if not args.recipe_dir or not args.bundle:
            parser.error("freeze requires recipe-dir and bundle")
        result = freeze(args.recipe_dir.resolve(), args.bundle.resolve(), args.authorize_evaluation)
    print(json.dumps(result, indent=2))
