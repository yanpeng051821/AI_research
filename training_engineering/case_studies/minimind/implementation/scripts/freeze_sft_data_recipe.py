"""Freeze a verified SFT data version, or check its hashes without starting training."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from inspect_sft_encoding import sha256_file
from verify_sft_recipe import run as verify


ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def descriptor(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def check_hashes(contract):
    references = [contract["policy"], contract["preparation_report"],
                  contract["input_candidate"], *contract["artifacts"].values()]
    if "acceptance" in contract:
        references.append(contract["acceptance"])
    for entry in references:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"data or metadata changed: {entry['path']}")
    for name, expected in contract["pipeline_identity"]["runtime_files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"runtime changed: {name}")
    tokenizer = ROOT / "assets/minimind_tokenizer"
    actual = {p.relative_to(tokenizer).as_posix(): sha256_file(p)
              for p in tokenizer.rglob("*") if p.is_file()}
    if actual != contract["pipeline_identity"]["tokenizer_files_sha256"]:
        raise ValueError("tokenizer asset inventory changed")
    for name, expected in contract["runtime"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"runtime package changed: {name}")


def check(marker_path):
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["status"] != "frozen":
        raise ValueError("data freeze is not committed")
    for entry in marker["files"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"frozen file changed: {entry['path']}")
    contract = json.loads(Path(marker["files"]["contract"]["path"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(marker["files"]["verification"]["path"]).read_text(encoding="utf-8"))
    if contract["status"] != "frozen" or receipt["status"] != "passed":
        raise ValueError("invalid contract or verification status")
    if receipt["contract_sha256"] != marker["files"]["contract"]["sha256"]:
        raise ValueError("verification does not bind the final contract")
    check_hashes(contract)
    return {"status": "passed", "data_contract_sha256": receipt["contract_sha256"],
            "train_records": contract["split_statistics"]["train"]["records"],
            "validation_records": contract["split_statistics"]["validation"]["records"],
            "training_started": False}


def freeze(candidate_path, acceptance_path, recipe_dir, verification_path):
    recipe_dir.mkdir(parents=True, exist_ok=True)
    final_path = recipe_dir / "DATA_CONTRACT.json"
    pending_path = recipe_dir / "DATA_CONTRACT.pending.json"
    marker_path = recipe_dir / "data_freeze.json"
    if any(p.exists() for p in (final_path, pending_path, marker_path, verification_path)):
        raise FileExistsError("freeze outputs already exist; do not overwrite a data version")
    candidate_identity = descriptor(candidate_path)
    acceptance_identity = descriptor(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    contract = json.loads(candidate_path.read_text(encoding="utf-8"))
    if contract["status"] != "candidate_pending_review_not_frozen":
        raise ValueError("expected an unfrozen candidate")
    if (acceptance["status"] != "user_authorized_bounded_data_freeze"
            or acceptance["candidate_sha256"] != candidate_identity["sha256"]
            or acceptance["policy_sha256"] != contract["policy"]["sha256"]
            or acceptance["accepted_limits"] != contract["limits"]
            or acceptance["training_execution_authorized"] is not False):
        raise ValueError("acceptance does not match the bounded candidate")
    check_hashes(contract)
    contract.update(contract_id="minimind_mini_sft_data_768_v1", status="frozen",
                    ready_for_training=True, training_execution_authorized=False,
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(), acceptance=acceptance_identity)
    contract["approval"].update(new_processing_and_split_choices_reviewed=True,
                                accepted_under_user_delegated_finalization=True,
                                independent_output_verification_completed=True)
    # Verify the final bytes before publishing them; the marker is the last commit step.
    write_new(pending_path, contract)
    verify(pending_path, verification_path, load_datasets=True)
    receipt = json.loads(verification_path.read_text(encoding="utf-8"))
    if (sha256_file(pending_path) != receipt["contract_sha256"]
            or descriptor(candidate_path) != candidate_identity
            or descriptor(acceptance_path) != acceptance_identity):
        raise ValueError("freeze inputs changed during verification")
    check_hashes(contract)
    pending_path.rename(final_path)
    files = {"contract": descriptor(final_path), "candidate": candidate_identity,
             "acceptance": acceptance_identity, "policy": contract["policy"],
             "preparation_report": contract["preparation_report"],
             "verification": descriptor(verification_path),
             "freezer": descriptor(Path(__file__)),
             "verifier": descriptor(ROOT / "scripts/verify_sft_recipe.py")}
    write_new(marker_path, {"schema_version": 1, "status": "frozen", "files": files,
                            "frozen_at_utc": contract["frozen_at_utc"],
                            "data_ready_not_training_authorization": True,
                            "training_started": False})
    return check(marker_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--recipe-dir", type=Path)
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
    else:
        if not all((args.candidate, args.acceptance, args.recipe_dir, args.verification_output)):
            parser.error("freeze requires candidate, acceptance, recipe-dir and verification-output")
        result = freeze(args.candidate, args.acceptance, args.recipe_dir, args.verification_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
