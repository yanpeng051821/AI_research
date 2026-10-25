"""Inspect the frozen full mini source for schema errors and exact text duplicates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/pretrain_mini_v1/source_inspection.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new output file")
    manifest = json.loads((ROOT / "data/processed/pretrain_candidate_v1/manifest.json").read_text(encoding="utf-8"))
    source_path = ROOT / manifest["source_path"]
    with source_path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_hash != manifest["source_sha256"]:
        raise ValueError("raw source file hash mismatch")

    errors = Counter()
    schemas = Counter()
    first_line_by_hash = {}
    duplicate_counts = Counter()
    duplicate_examples = []
    error_examples = []
    valid_count = 0
    line_count = 0
    char_count = 0
    for item in iter_jsonl(source_path):
        line_count += 1
        issues = ([item["error_code"]] if item["error_code"] is not None
                  else validate_pretrain_record(item["record"]))
        if issues:
            errors.update(issues)
            if len(error_examples) < 20:
                error_examples.append({"source_line": item["line_number"], "issues": issues})
            continue
        record = item["record"]
        schemas[tuple(sorted(record))] += 1
        text = record["text"]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest in first_line_by_hash:
            duplicate_counts[digest] += 1
            if len(duplicate_examples) < 20:
                duplicate_examples.append({
                    "source_line": item["line_number"],
                    "first_source_line": first_line_by_hash[digest],
                    "text_sha256": digest.hex(),
                })
        else:
            first_line_by_hash[digest] = item["line_number"]
        valid_count += 1
        char_count += len(text)
        if line_count % 250_000 == 0:
            print(f"inspected {line_count:,} records", flush=True)
    if line_count != manifest["source_total_lines"]:
        raise ValueError("source physical line count mismatch")
    report = {
        "scope": "full_source_structure_and_exact_text_identity_not_fact_check_or_split",
        "source_repo": manifest["source_repo"],
        "source_revision": manifest["source_revision"],
        "source_sha256": source_hash,
        "source_bytes": source_path.stat().st_size,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_lines": line_count,
        "valid_records": valid_count,
        "invalid_records": line_count - valid_count,
        "errors_by_code": dict(errors),
        "error_examples": error_examples,
        "schemas": [{"keys": list(keys), "records": count} for keys, count in sorted(schemas.items())],
        "total_text_characters": char_count,
        "unique_exact_texts": len(first_line_by_hash),
        "duplicate_extra_records": sum(duplicate_counts.values()),
        "duplicate_text_groups": len(duplicate_counts),
        "largest_exact_text_group": max(duplicate_counts.values(), default=0) + (1 if valid_count else 0),
        "duplicate_examples": duplicate_examples,
        "limitations": [
            "Exact UTF-8 text identity only; no normalization or near-duplicate analysis.",
            "Original document/conversation lineage is not inferred from text equality.",
            "Token length statistics and post-truncation token-prefix overlap require separate checks.",
            "No factual correctness or licensing review is performed by this structural scan.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
