"""Exact resume comparison for final-runtime gate artifacts (contract.json)."""

import argparse
import json
from pathlib import Path

import torch

from compare_sft_resume import equal_tree, sha
import compare_sft_resume


def compare(continuous, paused, resumed, output):
    if output.exists():
        raise FileExistsError(output)
    results, states, metrics = [], [], []
    inputs = {str(p.resolve()): sha(p) for p in (Path(__file__), Path(compare_sft_resume.__file__))}
    for directory in (continuous, paused, resumed):
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoint"])
        receipt_path = Path(str(checkpoint) + ".receipt.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("checkpoint receipt mismatch")
        for path in (checkpoint, receipt_path, directory / "result.json", directory / "metrics.jsonl", directory / "contract.json"):
            inputs[str(path.resolve())] = sha(path)
        results.append(result)
        states.append(torch.load(checkpoint, map_location="cpu", weights_only=True))
        metrics.append([r for s in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                        if (r := json.loads(s))["event"] == "update"])
    a, b, c = results
    if (not a["global_step"] == c["global_step"] > b["global_step"]
            or len({r["contract_sha256"] for r in results}) != 1
            or any(r["run_kind"] != "engineering_gate" or r["status"] != "paused" for r in results)
            or a["resume_from"] is not None or b["resume_from"] is not None
            or Path(c["resume_from"]).resolve() != Path(b["checkpoint"]).resolve()):
        raise ValueError("gate relationship mismatch")
    equal_tree(states[0], states[2])
    equal_tree(a["final_validation"], c["final_validation"], "validation")
    keys = ("global_step", "loss", "valid_tokens", "micro_batches", "grad_norm", "lr_used", "next_lr", "committed_samples")
    for rows, start, end in ((metrics[0], 1, a["global_step"]), (metrics[1], 1, b["global_step"]),
                             (metrics[2], b["global_step"] + 1, c["global_step"])):
        if [r["global_step"] for r in rows] != list(range(start, end + 1)):
            raise ValueError("gate update coverage mismatch")
    equal_tree([{k: r[k] for k in keys} for r in metrics[0]],
               [{k: r[k] for k in keys} for r in metrics[1] + metrics[2]], "update_metrics")
    if any(sha(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("inputs changed")
    report = {"status": "passed", "contract_sha256": a["contract_sha256"],
              "final_step": a["global_step"], "pause_step": b["global_step"],
              "checkpoint_state_exact_equal": True, "update_metrics_exact_equal": True,
              "final_validation_exact_equal": True, "checked_states": list(states[0]),
              "checkpoint_file_sha256_equal": sha(Path(a["checkpoint"])) == sha(Path(c["checkpoint"])),
              "inputs_sha256": inputs, "formal_training_started": False}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("continuous", "paused", "resumed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    compare(args.continuous.resolve(), args.paused.resolve(), args.resumed.resolve(), args.output.resolve())
