"""Compare encoding lengths on a frozen sample without changing existing artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_lengths(candidate_dir, tokenizer_dir, lengths, verify_source):
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_path = candidate_dir / "candidates.jsonl"
    candidate_hash = sha256_file(candidate_path)
    if candidate_hash != manifest["candidate_sha256"]:
        raise ValueError("candidate file hash does not match its manifest")
    if not lengths or any(length < 3 for length in lengths):
        raise ValueError("lengths must allow BOS, content and EOS")
    source_path = ROOT / manifest["source_path"]
    if verify_source and sha256_file(source_path) != manifest["source_sha256"]:
        raise ValueError("raw source file hash does not match its manifest")

    records = []
    for item in iter_jsonl(candidate_path):
        if item["error_code"] is not None:
            raise ValueError(f"candidate line {item['line_number']}: {item['error_code']}")
        record = item["record"]
        if validate_pretrain_record(record):
            raise ValueError(f"invalid candidate line {item['line_number']}")
        records.append(record)
    if len(records) != manifest["record_count"] or not records:
        raise ValueError("candidate count mismatch")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    counts = []
    pad_in_body = 0
    for start in range(0, len(records), 256):
        ids = tokenizer(
            [record["text"] for record in records[start:start + 256]],
            add_special_tokens=False, truncation=False,
        )["input_ids"]
        counts.extend(len(sequence) for sequence in ids)
        pad_in_body += sum(tokenizer.pad_token_id in sequence for sequence in ids)
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        raise ValueError("candidate sample has no content tokens")
    comparisons = []
    for length in sorted(set(lengths)):
        budget = length - 2
        dropped = int(np.maximum(counts - budget, 0).sum())
        complete = int((counts <= budget).sum())
        comparisons.append({
            "max_length": length,
            "content_budget": budget,
            "complete_records": complete,
            "complete_record_fraction": complete / len(records),
            "truncated_records": len(records) - complete,
            "truncated_record_fraction": (len(records) - complete) / len(records),
            "dropped_tail_content_tokens": dropped,
            "dropped_tail_content_fraction": dropped / total,
            "retained_content_tokens_with_truncation": total - dropped,
            "fixed_length_positions_per_epoch": len(records) * length,
            "content_utilization_of_fixed_positions": (total - dropped) / (len(records) * length),
        })
    return {
        "scope": "frozen_10000_record_sample_not_full_dataset_or_split",
        "source_repo": manifest["source_repo"],
        "source_revision": manifest["source_revision"],
        "source_sha256": manifest["source_sha256"],
        "raw_source_hash_verified_this_run": verify_source,
        "source_total_lines_from_manifest": manifest["source_total_lines"],
        "candidate_sha256": candidate_hash,
        "tokenizer_files_sha256": {
            name: sha256_file(tokenizer_dir / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "sample_count": len(records),
        "content_tokens": total,
        "min_content_tokens": int(counts.min()),
        "max_content_tokens": int(counts.max()),
        "length_percentiles": {
            f"p{percentile}": float(np.percentile(counts, percentile, method="linear"))
            for percentile in (50, 90, 95, 99)
        },
        "sample_records_containing_pad_id_in_body": pad_in_body,
        "comparisons": comparisons,
        "limitations": [
            "Length statistics describe the frozen sample, not all source records.",
            "Tail truncation preserves every sampled record; whole-record exclusion is a different policy.",
            "Fixed sequence positions are a compute-size proxy, not measured GPU time.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "data/processed/pretrain_candidate_v1")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 768, 1024])
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/pretrain_candidate_v1/analysis/length_comparison.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new output file")
    report = compare_lengths(args.candidate_dir, args.tokenizer_dir, args.lengths, args.verify_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
