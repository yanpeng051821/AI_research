"""Check/execute a frozen SFT recipe, or run a bounded pre-freeze engineering gate."""

import argparse
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sft_formal_runtime import build_contract, run
from freeze_sft_training import check
from train_sft import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--freeze", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--engineering-stop-after", type=int)
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.run and args.freeze is None:
        parser.error("formal execution requires --freeze, not a candidate config")
    if args.engineering_stop_after is not None and args.config is None:
        parser.error("engineering gate requires a config, not a frozen launch")
    if args.check_only and args.resume is not None:
        parser.error("check-only does not restore or train")
    if not args.check_only and args.output_dir is None:
        parser.error("execution requires a new output directory")
    output = args.output_dir.resolve() if args.output_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", {"status": "preparing", "training_started": False})
    try:
        contract, data = check(args.freeze) if args.freeze else build_contract(args.config.resolve())
        if output:
            write_json(output / "contract.json", contract)
        if args.check_only:
            result = {"status": "checked_not_trained", "contract_sha256": contract["contract_sha256"], "training_started": False}
        else:
            result = run(contract, data, output, resume=args.resume,
                         gate_stop=args.engineering_stop_after)
        if output:
            write_json(output / "result.json", result)
            write_json(output / "run_manifest.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        if output:
            progress = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            write_json(output / "run_manifest.json", {**progress, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
