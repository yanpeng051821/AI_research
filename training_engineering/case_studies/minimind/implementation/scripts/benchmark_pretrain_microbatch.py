import argparse
import json
from pathlib import Path

from minimind_lab.training.microbatch_benchmark import run_microbatch_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniMind pretraining micro-batch candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_microbatch_benchmark(args.config, args.output_dir, root)
    summary = [
        {
            "micro_batch_size": row["micro_batch_size"],
            "gradient_accumulation_steps": row["gradient_accumulation_steps"],
            "status": row["status"],
            "records_per_second": row.get("measured_records_per_second"),
            "minimum_device_free_ratio": row.get("minimum_device_free_ratio"),
        }
        for row in report["results"]
    ]
    print(json.dumps({"results": summary, "recommended_micro_batch_size": report["recommended_micro_batch_size"]}, indent=2))


if __name__ == "__main__":
    main()
