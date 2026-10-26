"""Plot completed-run learning curves without importing the training runtime."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weighted_windows(updates, size):
    windows = []
    for start in range(0, len(updates), size):
        chunk = updates[start:start + size]
        targets = sum(row["valid_tokens"] for row in chunk)
        windows.append({
            "first_step": chunk[0]["global_step"],
            "last_step": chunk[-1]["global_step"],
            "updates": len(chunk),
            "valid_targets": targets,
            "token_weighted_training_loss": sum(
                row["loss"] * row["valid_tokens"] for row in chunk
            ) / targets,
        })
    return windows


def summarize(rows, result, window_size):
    if result["status"] != "completed":
        raise ValueError("analyze only a completed run")
    if window_size < 1:
        raise ValueError("window size must be positive")
    updates = [row for row in rows if row["event"] == "update"]
    validations = [row for row in rows if row["event"] in {
        "initial_validation", "periodic_validation", "final_validation",
    }]
    expected_steps = list(range(1, result["global_step"] + 1))
    if [row["global_step"] for row in updates] != expected_steps:
        raise ValueError("update steps must be complete and ordered")
    for row in updates:
        if not math.isfinite(row["loss"]) or row["valid_tokens"] <= 0:
            raise ValueError("invalid training loss or target count")
    if sum(row["valid_tokens"] for row in updates) != result["valid_tokens_seen"]:
        raise ValueError("update target total differs from result")
    if not validations or validations[0]["event"] != "initial_validation":
        raise ValueError("missing initialization validation")
    if validations[-1]["event"] != "final_validation":
        raise ValueError("missing final validation")
    if validations[-1]["global_step"] != result["global_step"]:
        raise ValueError("final validation is not at the completed budget")
    if validations[0]["nll"] != result["initial_or_resume_validation"]["nll"]:
        raise ValueError("initial validation differs from result")
    if validations[-1]["nll"] != result["final_validation"]["nll"]:
        raise ValueError("final validation differs from result")
    for row in validations:
        if (not math.isfinite(row["nll"])
                or row["valid_tokens"] != result["final_validation"]["valid_tokens"]
                or row["records"] != result["validation_records"]
                or row["scope"] != result["validation_scope"]):
            raise ValueError("validation values or scope differ")
    if any(a["global_step"] >= b["global_step"]
           for a, b in zip(validations, validations[1:])):
        raise ValueError("validation steps are not strictly increasing")

    initial = validations[0]["nll"]
    total_drop = initial - validations[-1]["nll"]
    points = []
    for i, row in enumerate(validations):
        previous = validations[i - 1] if i else None
        drop = previous["nll"] - row["nll"] if previous else None
        step_gap = row["global_step"] - previous["global_step"] if previous else None
        points.append({
            "step": row["global_step"],
            "nll": row["nll"],
            "drop_since_previous": drop,
            "step_gap": step_gap,
            "drop_per_100_updates": drop / step_gap * 100 if previous else None,
            "share_of_observed_total_drop": (
                (initial - row["nll"]) / total_drop if total_drop > 0 else None
            ),
        })
    return {
        "contract_sha256": result["contract_sha256"],
        "updates": len(updates),
        "validation_records": result["validation_records"],
        "validation_targets": result["final_validation"]["valid_tokens"],
        "validation_points": points,
        "validation_nll_strictly_decreases_at_logged_points": all(
            a["nll"] > b["nll"] for a, b in zip(points, points[1:])
        ),
        "training_window_updates": window_size,
        "training_windows": weighted_windows(updates, window_size),
        "interpretation_limits": [
            "Training loss uses changing batches and parameters; validation uses fixed data.",
            "Window averages reconstruct loss sums from rounded logged loss values.",
            "No fixed-checkpoint training-set NLL was measured for a generalization gap.",
            "Declining validation NLL does not prove convergence or downstream ability.",
            "Curve shape alone does not identify the causal effect of the LR schedule.",
        ],
    }


def plot_curves(updates, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    windows = summary["training_windows"]
    points = summary["validation_points"]
    for ax, lower_bound, title in zip(
        axes, (0, 500), ("Full training trajectory", "After update 500 (rescaled y-axis)"),
    ):
        raw = [row for row in updates if row["global_step"] >= lower_bound]
        averaged = [row for row in windows if row["first_step"] > lower_bound]
        fixed = [row for row in points if row["step"] >= lower_bound]
        ax.plot([r["global_step"] for r in raw], [r["loss"] for r in raw],
                color="#147d92", alpha=0.18, linewidth=0.55, label="Training: each update")
        ax.plot([r["last_step"] for r in averaged],
                [r["token_weighted_training_loss"] for r in averaged],
                color="#096477", linewidth=2,
                label=f"Training: {summary['training_window_updates']}-update weighted blocks")
        ax.plot([r["step"] for r in fixed], [r["nll"] for r in fixed],
                color="#b34b38", marker="o", markersize=4.5, linewidth=1.8,
                label="Validation: fixed 5,000 records")
        ax.set(title=title, xlabel="Optimizer update", ylabel="Mean negative log-likelihood")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8.5)
        last = fixed[-1]
        ax.annotate(f"final val = {last['nll']:.4f}",
                    xy=(last["step"], last["nll"]), xytext=(-8, 18),
                    textcoords="offset points", ha="right", color="#943b2c")
    fig.suptitle("MiniMind mini pretraining: one epoch", fontsize=15)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-updates", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("use a new output directory; do not overwrite evidence")
    metrics = args.run_dir / "metrics.jsonl"
    result_path = args.run_dir / "result.json"
    before = {"metrics_sha256": sha256_file(metrics), "result_sha256": sha256_file(result_path)}
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summary = summarize(rows, result, args.window_updates)
    summary["source"] = {"run_id": args.run_dir.name, **before,
                         "analysis_script_sha256": sha256_file(Path(__file__))}
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True)
    updates = [row for row in rows if row["event"] == "update"]
    plot_curves(updates, summary, args.output_dir / "learning_curves.png")
    if before != {"metrics_sha256": sha256_file(metrics), "result_sha256": sha256_file(result_path)}:
        raise ValueError("source files changed during analysis")
    summary["plot_sha256"] = sha256_file(args.output_dir / "learning_curves.png")
    report = args.output_dir / "curve_summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(report), "validation_points": summary["validation_points"]}, indent=2))


if __name__ == "__main__":
    main()
