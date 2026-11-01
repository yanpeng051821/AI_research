"""Select complete SFT records by audited encoded length; not final data acceptance."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

from inspect_sft_encoding import sha256_file
from inspect_sft_full_encoding import pipeline_identity


ROOT = Path(__file__).resolve().parents[1]


def select_records(source, details, output_dir, limit, expected_source,
                   expected_details_sha256, expected_records):
    if limit < 2:
        raise ValueError("max length must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {"length_eligible": output_dir / "length_eligible.jsonl",
             "overlength_excluded": output_dir / "overlength_excluded.jsonl"}
    temporary = {key: path.with_suffix(".jsonl.partial") for key, path in paths.items()}
    counts = {key: {"records": 0, "bytes": 0, "targets_after_shift": 0}
              for key in paths}
    output_digests = {key: hashlib.sha256() for key in paths}
    source_digest, details_digest = hashlib.sha256(), hashlib.sha256()
    total = source_bytes = 0
    with (source.open("rb") as raw_stream, details.open("rb") as detail_stream,
          temporary["length_eligible"].open("xb") as retained,
          temporary["overlength_excluded"].open("xb") as excluded):
        streams = {"length_eligible": retained, "overlength_excluded": excluded}
        for total, (raw, detail) in enumerate(zip_longest(raw_stream, detail_stream), 1):
            if raw is None or detail is None:
                raise ValueError("source and encoding details have different record counts")
            source_digest.update(raw)
            details_digest.update(detail)
            source_bytes += len(raw)
            row = json.loads(detail)
            if (row["source_line"] != total or row["issues"]
                    or row["source_line_sha256"] != hashlib.sha256(raw).hexdigest()):
                raise ValueError(f"source/detail mismatch at line {total}")
            if row["input_tokens"] < 2 or row["targets_after_shift"] < 1:
                raise ValueError(f"invalid audited length or targets at line {total}")
            destination = "length_eligible" if row["input_tokens"] <= limit else "overlength_excluded"
            streams[destination].write(raw)
            output_digests[destination].update(raw)
            counts[destination]["records"] += 1
            counts[destination]["bytes"] += len(raw)
            counts[destination]["targets_after_shift"] += row["targets_after_shift"]
    if (total != expected_records or source_bytes != expected_source["size"]
            or source_digest.hexdigest() != expected_source["sha256"]
            or details_digest.hexdigest() != expected_details_sha256):
        raise ValueError("source/detail final identity mismatch")
    # No final-named dataset is exposed until all input identities have passed.
    for key, path in paths.items():
        temporary[key].rename(path)
    return {key: {**counts[key], "path": str(paths[key].resolve()),
                  "sha256": output_digests[key].hexdigest()} for key in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--encoding-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    args = parser.parse_args()
    if args.report_dir.exists() or args.output_dir.exists():
        raise FileExistsError("use fresh data and report directories")
    reference_bytes = args.encoding_report.read_bytes()
    reference = json.loads(reference_bytes)
    overall = reference["overall"]
    if overall["flagged_records"] or overall["encoded_records"] != overall["records"]:
        raise ValueError("full encoding report must have no flags or missing encodings")
    if pipeline_identity(args.tokenizer_dir) != reference["pipeline"]:
        raise ValueError("encoding pipeline changed; rerun preflight first")
    outputs = select_records(args.source, Path(reference["files"]["records"]["path"]),
                             args.output_dir, args.max_length, reference["source"],
                             reference["files"]["records"]["sha256"], overall["records"])
    if (args.encoding_report.read_bytes() != reference_bytes
            or pipeline_identity(args.tokenizer_dir) != reference["pipeline"]):
        raise ValueError("reference or encoding pipeline changed during selection")
    for output in outputs.values():
        if sha256_file(Path(output["path"])) != output["sha256"]:
            raise ValueError("written output hash mismatch")
    report = {
        "schema_version": 1,
        "status": "length_selection_completed_not_final_training_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {"max_length": args.max_length,
                   "overlength_policy": "exclude_whole_record", "truncation": False},
        "source": {**reference["source"], "records": overall["records"]},
        "encoding_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "encoding_details_sha256": reference["files"]["records"]["sha256"],
        "pipeline": reference["pipeline"],
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": outputs,
        "scope": {"source_modified": False, "raw_record_bytes_preserved": True,
                  "relative_source_order_preserved": True, "content_filtering_applied": False,
                  "identity_rewrite_applied": False, "deduplication_applied": False,
                  "split_created": False, "data_recipe_frozen": False},
    }
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
