"""Inspect a pretraining checkpoint directory and identify the latest valid resume point."""

import argparse
import json
from pathlib import Path

from minimind_lab.training.recovery import inspect_checkpoint_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--contract", type=Path)
    identity.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.contract is not None:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        expected_contract_sha256 = contract["contract_sha256"]
    else:
        expected_contract_sha256 = args.contract_sha256
    report = inspect_checkpoint_directory(args.checkpoint_dir, expected_contract_sha256)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
