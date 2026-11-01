"""Freeze/check the approved SFT recipe after final-runtime engineering gates pass."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from sft_formal_runtime import ROOT, RUNTIME_SCRIPTS, build_contract
from train_sft import sha256_file, write_json
from freeze_sft_evaluation import check as check_evaluation


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor(path, recipe):
    return {"file": Path(os.path.relpath(path, recipe)).as_posix(), "sha256": sha256_file(path)}


def check(marker_path):
    marker_path = Path(marker_path).resolve()
    marker = read(marker_path)
    if marker["status"] != "frozen" or marker["formal_training_started"] is not False:
        raise ValueError("invalid SFT freeze marker")
    for info in marker["files"].values():
        if sha256_file((marker_path.parent / info["file"]).resolve()) != info["sha256"]:
            raise ValueError(f"frozen input changed: {info['file']}")
    config = (marker_path.parent / marker["files"]["training_config"]["file"]).resolve()
    current, data = build_contract(config)
    if current["contract_sha256"] != marker["contract_sha256"]:
        raise ValueError("training runtime/recipe identity mismatch")
    patch_path = (ROOT / current["identity"]["core"]["candidate"]["scoring_patch"]).resolve()
    evaluation_path = patch_path.parent / "evaluation_freeze.json"
    if sha256_file(evaluation_path) != read(patch_path)["base_evaluation_freeze_sha256"]:
        raise ValueError("V2 patch base evaluation mismatch")
    check_evaluation(evaluation_path)
    return current, data


def freeze(recipe, comparison_path, stability_path, tests_path):
    recipe = recipe.resolve()
    marker_path = recipe / "training_freeze.json"
    contract_path = recipe / "contract.json"
    if marker_path.exists() or contract_path.exists():
        raise FileExistsError("do not overwrite a frozen training version")
    accepted = read(recipe / "training_acceptance.json")
    current, _ = build_contract(recipe / "training_config.json")
    if accepted["status"] != "user_approved_recipe_freeze_not_start" or accepted["operations"] != current["identity"]["operations"]:
        raise ValueError("user acceptance does not match operations")
    comparison, stability = read(comparison_path), read(stability_path)
    if (comparison["status"] != "passed" or comparison["contract_sha256"] != current["contract_sha256"]
            or comparison["checkpoint_state_exact_equal"] is not True
            or comparison["update_metrics_exact_equal"] is not True
            or comparison["final_step"] < 20):
        raise ValueError("final-runtime resume gate not passed")
    if (stability["status"] != "paused" or stability["run_kind"] != "engineering_gate"
            or stability["global_step"] < 20 or stability["contract_sha256"] != current["contract_sha256"]):
        raise ValueError("final-runtime stability gate mismatch")
    for filename, digest in comparison["inputs_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError("comparison evidence changed")
    suites = ET.parse(tests_path).getroot()
    cases = list(suites.iter("testcase"))
    if not cases or any(list(suites.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("require passing, nonempty final-runtime regression report")
    if not any("test_sft_formal_runtime" in case.attrib.get("classname", "") for case in cases):
        raise ValueError("final-runtime tests absent from regression evidence")
    files = {"training_config": recipe / "training_config.json",
             "acceptance": recipe / "training_acceptance.json", "human_contract": recipe / "TRAINING_CONTRACT.md",
             "candidate": (ROOT / current["config"]["base_candidate"]).resolve(),
             "resume_comparison": comparison_path, "stability": stability_path, "tests": tests_path,
             "dependency_lock": ROOT / "uv.lock", "dependency_specification": ROOT / "pyproject.toml"}
    for name in RUNTIME_SCRIPTS:
        files[name] = ROOT / "scripts" / name
    files["objectives"] = recipe.parent / "EXPERIMENT_OBJECTIVES.md"
    files["evaluation_freeze"] = recipe.parent / "evaluation_recipe/evaluation_freeze.json"
    files["scoring_patch"] = recipe.parent / "evaluation_recipe/scoring_v2_patch.json"
    files["operations_measurement"] = recipe.parent / "results/operations_measurement_v1.json"
    files["comparer"] = ROOT / "scripts/compare_sft_resume.py"
    for name in ("test_sft_formal_runtime.py", "test_sft_training_entry.py", "test_sft_resume_comparison.py"):
        files[name] = ROOT / "tests" / name
    entries = {name: descriptor(path, recipe) for name, path in files.items()}
    write_json(contract_path, current)
    entries["resolved_contract"] = descriptor(contract_path, recipe)
    for info in entries.values():
        if sha256_file((recipe / info["file"]).resolve()) != info["sha256"]:
            raise ValueError("freeze inputs changed before commit")
    marker = {"schema_version": 1, "status": "frozen", "contract_sha256": current["contract_sha256"],
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "files": entries,
              "formal_training_started": False, "launch_requires_explicit_run": True,
              "gates": {"real_cuda_resume": "passed", "stability_step": stability["global_step"],
                        "regression_cases": len(cases)}, "expected_updates": current["identity"]["core"]["planned_updates"]}
    write_json(marker_path, marker)
    check(marker_path)
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", type=Path)
    mode.add_argument("--recipe-dir", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--stability", type=Path)
    parser.add_argument("--tests", type=Path)
    args = parser.parse_args()
    if args.check:
        contract, _ = check(args.check)
        print(json.dumps({"status": "passed", "contract_sha256": contract["contract_sha256"], "training_started": False}))
    else:
        if not all((args.comparison, args.stability, args.tests)):
            parser.error("freeze requires comparison, stability and tests")
        result = freeze(args.recipe_dir, args.comparison.resolve(), args.stability.resolve(), args.tests.resolve())
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
