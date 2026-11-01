"""Keep the first exact conversation in a length-selected SFT candidate."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

from inspect_sft_duplicates import conversation_sha256
from inspect_sft_encoding import sha256_file
from inspect_sft_full_encoding import pipeline_identity
from minimind_lab.data.validator import validate_sft_record


ROOT = Path(__file__).resolve().parents[1]


def deduplicate(source, details, output_dir, length_report, encoding_report):
    limit = length_report["policy"]["max_length"]
    expected = length_report["outputs"]["length_eligible"]
    detail_digest = hashlib.sha256()
    detail_count = 0

    def eligible_rows():
        nonlocal detail_count
        with details.open("rb") as stream:
            for detail_count, raw in enumerate(stream, 1):
                detail_digest.update(raw)
                row = json.loads(raw)
                if row["source_line"] != detail_count or row["issues"]:
                    raise ValueError("invalid or unordered encoding detail")
                if row["input_tokens"] <= limit:
                    yield row
        if (detail_count != encoding_report["overall"]["records"]
                or detail_digest.hexdigest() != length_report["encoding_details_sha256"]):
            raise ValueError("encoding detail identity/count mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {"unique": output_dir / "unique_candidate.jsonl",
             "duplicates": output_dir / "duplicate_records.jsonl",
             "lineage": output_dir / "lineage.jsonl"}
    temporary = {key: path.with_suffix(".jsonl.partial") for key, path in paths.items()}
    hashes = {key: hashlib.sha256() for key in paths}
    counts = {key: {"records": 0, "bytes": 0, "targets_after_shift": 0}
              for key in ("unique", "duplicates")}
    first_seen = {}
    source_digest = hashlib.sha256()
    source_bytes = total = targets = lineage_bytes = 0
    with (source.open("rb") as raw_stream,
          temporary["unique"].open("xb") as unique_stream,
          temporary["duplicates"].open("xb") as duplicate_stream,
          temporary["lineage"].open("xb") as lineage_stream):
        streams = {"unique": unique_stream, "duplicates": duplicate_stream}
        for total, (raw, row) in enumerate(zip_longest(raw_stream, eligible_rows()), 1):
            if raw is None or row is None:
                raise ValueError("length candidate and encoding details have different counts")
            raw_hash = hashlib.sha256(raw).hexdigest()
            if (raw_hash != row["source_line_sha256"] or row["targets_after_shift"] < 1):
                raise ValueError(f"candidate/detail mismatch at input line {total}")
            source_digest.update(raw)
            source_bytes += len(raw)
            targets += row["targets_after_shift"]
            record = json.loads(raw.decode("utf-8"))
            issues = validate_sft_record(record)
            if issues:
                raise ValueError(f"invalid candidate record at line {total}: {issues}")
            key = conversation_sha256(record["conversations"])
            if key not in first_seen:
                destination = "unique"
                first_seen[key] = [counts["unique"]["records"] + 1, row["source_line"], 1]
            else:
                destination = "duplicates"
                first_seen[key][2] += 1
            retained_output_line, retained_source_line, _ = first_seen[key]
            streams[destination].write(raw)
            hashes[destination].update(raw)
            counts[destination]["records"] += 1
            counts[destination]["bytes"] += len(raw)
            counts[destination]["targets_after_shift"] += row["targets_after_shift"]
            trace = {"input_line": total, "source_line": row["source_line"],
                     "source_line_sha256": raw_hash, "conversation_sha256": key,
                     "action": "keep" if destination == "unique" else "exclude_exact_duplicate",
                     "retained_output_line": retained_output_line,
                     "retained_source_line": retained_source_line}
            trace_bytes = (json.dumps(trace, separators=(",", ":")) + "\n").encode("utf-8")
            lineage_stream.write(trace_bytes)
            hashes["lineage"].update(trace_bytes)
            lineage_bytes += len(trace_bytes)
            if total % 100000 == 0:
                print(f"Processed {total:,} length-eligible records", flush=True)
    if (total != expected["records"] or source_bytes != expected["bytes"]
            or source_digest.hexdigest() != expected["sha256"]
            or targets != expected["targets_after_shift"]):
        raise ValueError("length candidate identity/count/target mismatch")
    for key, path in paths.items():
        temporary[key].rename(path)
    outputs = {key: {**counts[key], "path": str(paths[key].resolve()),
                     "sha256": hashes[key].hexdigest()} for key in counts}
    outputs["lineage"] = {"path": str(paths["lineage"].resolve()), "records": total,
                          "bytes": lineage_bytes, "sha256": hashes["lineage"].hexdigest()}
    return {
        "input_records": total, "unique_conversations": len(first_seen),
        "duplicate_groups": sum(item[2] > 1 for item in first_seen.values()),
        "records_in_duplicate_groups": sum(item[2] for item in first_seen.values() if item[2] > 1),
        "max_group_size": max((item[2] for item in first_seen.values()), default=0),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-report", type=Path, required=True)
    parser.add_argument("--encoding-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    args = parser.parse_args()
    if args.report_dir.exists() or args.output_dir.exists():
        raise FileExistsError("use fresh data and report directories")
    start = time.perf_counter()
    length_bytes, encoding_bytes = args.length_report.read_bytes(), args.encoding_report.read_bytes()
    length_report, encoding_report = json.loads(length_bytes), json.loads(encoding_bytes)
    dependencies = {name: sha256_file(path) for name, path in {
        "deduplication_entry": Path(__file__),
        "conversation_key": ROOT / "scripts/inspect_sft_duplicates.py",
        "validator": ROOT / "src/minimind_lab/data/validator.py",
    }.items()}
    if (hashlib.sha256(encoding_bytes).hexdigest() != length_report["encoding_report_sha256"]
            or length_report["encoding_details_sha256"] != encoding_report["files"]["records"]["sha256"]
            or length_report["source"]["sha256"] != encoding_report["source"]["sha256"]
            or length_report["policy"]["overlength_policy"] != "exclude_whole_record"
            or length_report["policy"]["truncation"]):
        raise ValueError("incompatible input reports or length policy")
    before_pipeline = pipeline_identity(args.tokenizer_dir)
    if before_pipeline != length_report["pipeline"] or before_pipeline != encoding_report["pipeline"]:
        raise ValueError("encoding pipeline changed; rerun preflight and length selection")
    result = deduplicate(Path(length_report["outputs"]["length_eligible"]["path"]),
                         Path(encoding_report["files"]["records"]["path"]),
                         args.output_dir, length_report, encoding_report)
    if (args.length_report.read_bytes() != length_bytes or args.encoding_report.read_bytes() != encoding_bytes
            or pipeline_identity(args.tokenizer_dir) != before_pipeline
            or dependencies["deduplication_entry"] != sha256_file(Path(__file__))
            or dependencies["conversation_key"] != sha256_file(ROOT / "scripts/inspect_sft_duplicates.py")):
        raise ValueError("input reports or processing code changed during run")
    for output in result["outputs"].values():
        if sha256_file(Path(output["path"])) != output["sha256"]:
            raise ValueError("written output hash mismatch")
    report = {
        "schema_version": 1, "status": "exact_deduplication_completed_not_final_training_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {"max_length": length_report["policy"]["max_length"],
                   "overlength_policy": "exclude_whole_record",
                   "deduplication": "exact_conversations_keep_first_in_source_order",
                   "key": "canonical conversations SHA256; preserves message order and every message field",
                   "ignored": "outer JSON formatting, object key order and top-level non-conversation metadata"},
        "input": length_report["outputs"]["length_eligible"],
        "original_source": length_report["source"],
        "length_report_sha256": hashlib.sha256(length_bytes).hexdigest(),
        "encoding_report_sha256": hashlib.sha256(encoding_bytes).hexdigest(),
        "encoding_details_sha256": length_report["encoding_details_sha256"],
        "code_sha256": dependencies, "pipeline": before_pipeline,
        **result,
        "scope": {"input_modified": False, "raw_record_bytes_preserved": True,
                  "relative_source_order_preserved": True, "exact_deduplication_applied": True,
                  "content_quality_accepted": False, "identity_rewrite_applied": False,
                  "near_duplicates_checked": False, "shared_prompt_isolation_checked": False,
                  "split_created": False, "data_recipe_frozen": False},
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
