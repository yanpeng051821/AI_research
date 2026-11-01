"""Inspect fixed SFT samples before training without filtering or truncating them."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from jinja2 import TemplateError
import numpy as np
from transformers import AutoTokenizer

from minimind_lab.data.sft import encode_sft_conversation, prepare_sft_messages
from minimind_lab.data.validator import validate_sft_record


ROOT = Path(__file__).resolve().parents[1]
STRATA = ("plain", "reasoning_only", "tool_only", "reasoning_and_tool")


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def boundary_paths(value, markers, path):
    if isinstance(value, str):
        return [path] if any(marker in value for marker in markers) else []
    found = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(boundary_paths(item, markers, f"{path}[{index}]"))
    elif isinstance(value, dict):
        for index, (key, item) in enumerate(value.items()):
            found.extend(boundary_paths(key, markers, f"{path}.key[{index}]"))
            # Numeric field locations avoid copying arbitrary text keys into reports.
            found.extend(boundary_paths(item, markers, f"{path}.value[{index}]"))
    return found


def distribution(values):
    if not values:
        return None
    return {
        "min": min(values), "max": max(values), "mean": float(np.mean(values)),
        "percentiles": {f"p{p}": float(np.percentile(values, p, method="linear"))
                        for p in (50, 90, 95, 99)},
    }


def summarize(rows):
    encoded = [row for row in rows if row["input_tokens"] is not None]
    input_positions = sum(row["input_tokens"] for row in encoded)
    shifted_positions = sum(max(row["input_tokens"] - 1, 0) for row in encoded)
    targets = sum(row["targets_after_shift"] for row in encoded)
    return {
        "records": len(rows), "encoded_records": len(encoded),
        "records_without_encoding_flags": sum(not row["issues"] for row in rows),
        "flagged_records": sum(bool(row["issues"]) for row in rows),
        "issues": dict(sorted(Counter(issue for row in rows for issue in row["issues"]).items())),
        "input_tokens": distribution([row["input_tokens"] for row in encoded]),
        "targets_after_shift": distribution([row["targets_after_shift"] for row in encoded]),
        "total_input_positions": input_positions,
        "total_shifted_positions": shifted_positions,
        "total_targets_after_shift": targets,
        "supervised_fraction_of_shifted_positions": targets / shifted_positions if shifted_positions else None,
        "statistics_scope": "encoded_records_including_flagged_records_not_accepted_training_data",
    }


def inspect_samples(samples, expected_sha256, tokenizer):
    raw = samples.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("sample file identity mismatch")
    markers = tuple(marker for marker in (tokenizer.bos_token, tokenizer.eos_token) if marker)
    if len(markers) != 2:
        raise ValueError("tokenizer must define BOS and EOS text")

    rows = []
    previous_line = 0
    for sample_row, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        packet = json.loads(line)
        if not isinstance(packet, dict) or "record" not in packet:
            raise ValueError(f"invalid sample envelope at row {sample_row}")
        source_line = packet.get("source_line")
        digest = packet.get("source_line_sha256")
        if type(source_line) is not int or source_line <= previous_line:
            raise ValueError("source_line must be positive and strictly increasing")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"invalid source-line hash at row {sample_row}")
        if packet.get("stratum") not in STRATA:
            raise ValueError(f"invalid stratum at row {sample_row}")
        previous_line = source_line
        row = {key: packet[key] for key in ("source_line", "source_line_sha256", "stratum")}
        row.update({"sample_row": sample_row, "issues": [], "input_tokens": None,
                    "targets_after_shift": None})
        rows.append(row)
        issues = validate_sft_record(packet["record"])
        if issues:
            row["issues"].append("invalid_structure")
            row["structure_issues"] = issues
            continue

        messages = packet["record"]["conversations"]
        try:
            prepared, tools = prepare_sft_messages(messages)
        except ValueError as exc:
            row["issues"].append("field_conversion_error")
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
            continue
        paths = boundary_paths(prepared, markers, "messages") + boundary_paths(tools, markers, "tools")
        if paths:
            row["issues"].append("potential_message_boundary_literal")
            row["boundary_paths"] = paths[:10]
            row["boundary_path_count"] = len(paths)
        try:
            feature = encode_sft_conversation(messages, tokenizer)
        except (ValueError, TypeError, TemplateError) as exc:
            row["issues"].append("encoding_error")
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
            continue
        ids, labels = feature["input_ids"], feature["labels"]
        if len(ids) != len(labels):
            raise RuntimeError("encoder returned misaligned input_ids and labels")
        row["input_tokens"] = len(ids)
        row["targets_after_shift"] = sum(label != -100 for label in labels[1:])
        if row["targets_after_shift"] == 0:
            row["issues"].append("zero_targets_after_shift")

    if not rows:
        raise ValueError("sample file must not be empty")
    if sha256_file(samples) != expected_sha256:
        raise ValueError("sample file changed during inspection")
    result = {
        "status": "encoding_flags_found" if any(row["issues"] for row in rows)
                  else "sample_encoding_diagnostics_clear",
        "overall": summarize(rows),
        "by_stratum": {name: summarize([row for row in rows if row["stratum"] == name])
                       for name in STRATA},
        "longest_examples": sorted(
            (row for row in rows if row["input_tokens"] is not None),
            key=lambda row: (-row["input_tokens"], row["source_line"]),
        )[:10],
    }
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--sample-key", choices=("random", *STRATA), default="random")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.report_dir.exists():
        raise FileExistsError("use a fresh report directory")
    source_report_sha = sha256_file(args.source_report)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    expected = source_report["sample_files"][args.sample_key]["sha256"]
    if sha256_file(args.samples) != expected:
        raise ValueError("sample file identity mismatch")
    tokenizer_hashes = {path.relative_to(args.tokenizer_dir).as_posix(): sha256_file(path)
                        for path in sorted(args.tokenizer_dir.rglob("*.json"))}
    if not tokenizer_hashes:
        raise ValueError("tokenizer JSON assets are missing")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    result, rows = inspect_samples(args.samples, expected, tokenizer)
    after = {path.relative_to(args.tokenizer_dir).as_posix(): sha256_file(path)
             for path in sorted(args.tokenizer_dir.rglob("*.json"))}
    if tokenizer_hashes != after or sha256_file(args.source_report) != source_report_sha:
        raise ValueError("tokenizer or source report changed during inspection")
    result.update({
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"sample_key": args.sample_key, "sample_path": str(args.samples),
                  "sample_sha256": expected, "source_report_sha256": source_report_sha,
                  "source_identity_reference": source_report["source"],
                  "full_source_rechecked_by_this_entry": False},
        "encoding": {"entry": "encode_sft_conversation", "truncation": False, "padding": False,
                     "generation_prompt": False, "extra_special_tokens": False,
                     "retained_thinking_supervised": True,
                     "ending": "EOS plus trailing newline", "percentile_method": "numpy linear"},
        "tokenizer_files_sha256": tokenizer_hashes,
        "code_sha256": {name: sha256_file(path) for name, path in {
            "inspection": Path(__file__), "encoder": ROOT / "src/minimind_lab/data/sft.py",
            "validator": ROOT / "src/minimind_lab/data/validator.py",
        }.items()},
        "runtime": {name: importlib.metadata.version(name)
                    for name in ("transformers", "tokenizers", "numpy")},
        "scope": {"sample_only": True, "full_source_encoding_verified": False,
                  "quality_acceptance_claimed": False, "tool_protocol_verified": False,
                  "boundary_detection_is_conservative_not_complete": True,
                  "training_runtime_gates_changed": False, "filtering_executed": False,
                  "training_recipe_frozen": False, "training_started": False},
    })
    args.report_dir.mkdir(parents=True, exist_ok=False)
    records_path = args.report_dir / "records.jsonl"
    with records_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result["records_file"] = {"name": records_path.name, "sha256": sha256_file(records_path)}
    with (args.report_dir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report_dir": str(args.report_dir), "status": result["status"],
                      "overall": result["overall"]}, indent=2), flush=True)
    if result["status"] == "encoding_flags_found":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
