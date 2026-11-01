"""Compare a real continuous SFT probe with a paused and resumed probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal_tree(left, right, location="checkpoint"):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"tensor mismatch: {location}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"mapping mismatch: {location}")
        for key in left:
            equal_tree(left[key], right[key], f"{location}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(right) is not type(left) or len(left) != len(right):
            raise ValueError(f"sequence mismatch: {location}")
        for index, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, f"{location}[{index}]")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"value mismatch: {location}")


def compare(continuous, paused, resumed, output):
    if output.exists():
        raise FileExistsError(output)
    results = []
    inputs = {str(Path(__file__).resolve()): sha(Path(__file__))}
    for directory in (continuous, paused, resumed):
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoint"])
        receipt_path = Path(str(checkpoint) + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("checkpoint receipt mismatch")
        for p in (result_path, checkpoint, receipt_path, directory / "metrics.jsonl", directory / "candidate_contract.json"):
            inputs[str(p.resolve())] = sha(p)
        results.append(result)
    a, b, c = results
    if not (a["status"] == b["status"] == c["status"] == "paused"
            and a["global_step"] == c["global_step"] > b["global_step"]
            and a["contract_sha256"] == b["contract_sha256"] == c["contract_sha256"]
            and Path(c["resume_from"]).resolve() == Path(b["checkpoint"]).resolve()
            and a["resume_from"] is None and b["resume_from"] is None):
        raise ValueError("not a same-contract continuous/pause/resume comparison")
    left = torch.load(a["checkpoint"], map_location="cpu", weights_only=True)
    right = torch.load(c["checkpoint"], map_location="cpu", weights_only=True)
    equal_tree(left, right)
    equal_tree(a["final_validation"], c["final_validation"], "final_validation")

    def updates(directory):
        return [r for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if (r := json.loads(line))["event"] == "update"]

    rows_a, rows_b, rows_c = updates(continuous), updates(paused), updates(resumed)
    for rows, start, end in ((rows_a, 1, a["global_step"]), (rows_b, 1, b["global_step"]),
                             (rows_c, b["global_step"] + 1, c["global_step"])):
        if [r["global_step"] for r in rows] != list(range(start, end + 1)):
            raise ValueError("update coverage mismatch")
    keys = ("global_step", "loss", "valid_tokens", "micro_batches", "grad_norm", "lr_used", "next_lr", "committed_samples")
    equal_tree([{k: r[k] for k in keys} for r in rows_a],
               [{k: r[k] for k in keys} for r in rows_b + rows_c], "update_metrics")
    if any(sha(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("comparison inputs changed")
    report = {"status": "passed", "contract_sha256": a["contract_sha256"],
              "pause_step": b["global_step"], "final_step": a["global_step"],
              "comparison": "continuous20_vs_10_then_resume_to20" if a["global_step"] == 20 else "continuous_vs_resumed",
              "checkpoint_state_exact_equal": True,
              "checked_states": list(left), "update_metrics_exact_equal": True,
              "final_validation_exact_equal": True,
              "checkpoint_file_sha256_equal": sha(Path(a["checkpoint"])) == sha(Path(c["checkpoint"])),
              "inputs_sha256": inputs, "formal_training_started": False,
              "limitations": "Same-device short-run determinism; not proof of cross-device determinism or long-run stability."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("continuous", "paused", "resumed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare(args.continuous.resolve(), args.paused.resolve(), args.resumed.resolve(), args.output.resolve())
