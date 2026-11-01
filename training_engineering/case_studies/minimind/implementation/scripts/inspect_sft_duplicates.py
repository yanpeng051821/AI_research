"""Audit exact conversation duplicates without deleting or splitting records."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from minimind_lab.data.validator import validate_sft_record


def conversation_sha256(messages):
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def scan_duplicates(source, expected_sha256, expected_size, expected_records):
    start = time.perf_counter()
    groups = {}
    examples = []
    digest = hashlib.sha256()
    byte_count = record_count = 0
    for_read = Path(source)
    with for_read.open("rb") as stream:
        for record_count, raw in enumerate(stream, 1):
            digest.update(raw)
            byte_count += len(raw)
            record = json.loads(raw.decode("utf-8"))
            issues = validate_sft_record(record)
            if issues:
                raise ValueError(f"invalid record at line {record_count}: {issues}")
            key = conversation_sha256(record["conversations"])
            if key in groups:
                group = groups[key]
                group[1] += 1
                if len(examples) < 10:
                    examples.append({"conversation_sha256": key,
                                     "first_source_line": group[0],
                                     "duplicate_source_line": record_count})
            else:
                groups[key] = [record_count, 1]
            if record_count % 100000 == 0:
                print(f"Checked {record_count:,} source records", flush=True)
    if (byte_count != expected_size or record_count != expected_records
            or digest.hexdigest() != expected_sha256):
        raise ValueError("source identity/count mismatch")
    repeated_sizes = [item[1] for item in groups.values() if item[1] > 1]
    return {
        "schema_version": 1,
        "status": "exact_conversation_duplicates_audited_no_filtering",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"size": byte_count, "sha256": digest.hexdigest(),
                   "records": record_count},
        "method": {
            "key": "SHA256 of conversations serialized with sorted object keys and compact JSON",
            "preserved": "message/list order, every message field, text whitespace and JSON strings",
            "ignored": "top-level fields other than conversations and outer JSON formatting",
            "not_checked": ["near duplicates", "shared prompts with different answers",
                            "shared source documents", "semantic equivalence of tool JSON strings"],
        },
        "counts": {
            "unique_conversations": len(groups),
            "duplicate_groups": len(repeated_sizes),
            "records_in_duplicate_groups": sum(repeated_sizes),
            "duplicates_beyond_first": sum(size - 1 for size in repeated_sizes),
            "max_group_size": max((item[1] for item in groups.values()), default=0),
        },
        "example_pairs": examples,
        "scope": {"source_modified": False, "filtering_executed": False,
                  "split_created": False, "content_quality_acceptance_claimed": False},
        "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.report_dir.exists():
        raise FileExistsError("use a fresh report directory")
    reference_bytes = args.source_report.read_bytes()
    reference = json.loads(reference_bytes)
    report = scan_duplicates(args.source, reference["source"]["sha256"],
                             reference["source"]["size"],
                             reference["counts"]["total_lines"])
    if args.source_report.read_bytes() != reference_bytes:
        raise ValueError("source report changed during scan")
    report["source_report_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
    report["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
