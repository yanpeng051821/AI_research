import argparse
import json
import sys

from collections.abc import Sequence
from pathlib import Path

from minimind_lab.data.pipeline import audit_jsonl


def build_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimind-audit",
        description="Audit pretrain or sft jsonl datasets."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input jsonl file."
    )

    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["pretrain", "sft"],
        help="Type of dataset to audit."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for accepted.jsonl, rejected.jsonl, and report.json."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = audit_jsonl(
            args.input_path,
            args.dataset_type,
            args.output_dir
        )
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    if report["rejected_lines"] > 0:
        return 1
    
    return 0


def entrypoint()-> None:
    raise SystemExit(main())



