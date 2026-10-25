"""Audit full split token lengths and exact model-input overlap at a candidate length."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import encode_pretrain_text
from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).digest()


def checked_rows(path):
    previous = 0
    for item in iter_jsonl(path):
        record = item["record"]
        if item["error_code"] or validate_pretrain_record(record):
            raise ValueError(f"invalid split record {path}:{item['line_number']}")
        line = record.get("source_line")
        if type(line) is not int or line <= previous:
            raise ValueError("source_line must be positive and strictly increasing within each split")
        previous = line
        yield record


def batches(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def audit_encoding(split_dir, tokenizer, max_length, batch_size=256):
    if type(max_length) is not int or max_length < 3 or batch_size <= 0:
        raise ValueError("max_length must be at least 3 and batch_size must be positive")
    if any(getattr(tokenizer, name) is None for name in ("bos_token_id", "eos_token_id", "pad_token_id")):
        raise ValueError("tokenizer must define BOS, EOS and PAD")
    if tokenizer.truncation_side != "right":
        raise ValueError("audit requires the current dataset's right truncation policy")
    manifest_path = split_dir / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "text_split_complete_encoding_not_checked":
        raise ValueError("unsupported split manifest status")
    before = {}
    for name, info in manifest["files"].items():
        before[name] = sha256_file(split_dir / info["file"])
        if before[name] != info["sha256"]:
            raise ValueError(f"{name} file identity mismatch")

    validation_inputs = {}
    overlap_records = 0
    overlap_examples = []
    statistics = {}
    for split in ("validation", "train"):
        lengths = []
        totals = Counter()
        for rows in batches(checked_rows(split_dir / manifest["files"][split]["file"]), batch_size):
            sequences = tokenizer(
                [row["text"] for row in rows], add_special_tokens=False,
                truncation=False, padding=False,
            )["input_ids"]
            if len(sequences) != len(rows):
                raise ValueError("tokenizer batch count mismatch")
            for row, body in zip(rows, sequences):
                kept = body[:max_length - 2]
                ids = [tokenizer.bos_token_id] + kept + [tokenizer.eos_token_id]
                # Compare the vectorized audit recipe to the actual Dataset encoder.
                if totals["records"] < 4:
                    if ids != encode_pretrain_text(row["text"], tokenizer, max_length):
                        raise ValueError("audit encoding does not match Dataset encoder")
                key = input_digest(ids)
                if split == "validation":
                    validation_inputs.setdefault(key, row["source_line"])
                elif key in validation_inputs:
                    overlap_records += 1
                    if len(overlap_examples) < 20:
                        overlap_examples.append({
                            "train_source_line": row["source_line"],
                            "validation_source_line": validation_inputs[key],
                            "input_sha256": key.hex(), "input_length": len(ids),
                        })
                lengths.append(len(body))
                totals.update({
                    "records": 1, "body_tokens": len(body), "retained_body_tokens": len(kept),
                    "dropped_tail_tokens": len(body) - len(kept),
                    "truncated_records": int(len(body) > max_length - 2),
                    "active_input_positions": len(ids), "valid_causal_targets": len(ids) - 1,
                    "records_with_pad_id_in_retained_body": int(tokenizer.pad_token_id in kept),
                    "empty_encoded_bodies": int(not body),
                })
            if totals["records"] % 32768 == 0:
                print(f"{split}: encoded {totals['records']:,} records", flush=True)
        if totals["records"] != manifest["counts"][split] or not lengths:
            raise ValueError(f"{split} count mismatch or empty split")
        count = totals["records"]
        statistics[split] = {
            **dict(totals),
            "fixed_length_positions": count * max_length,
            "active_position_fraction": totals["active_input_positions"] / (count * max_length),
            "truncated_record_fraction": totals["truncated_records"] / count,
            "min_body_tokens": min(lengths), "max_body_tokens": max(lengths),
            "body_length_percentiles": {
                f"p{p}": float(np.percentile(lengths, p, method="linear")) for p in (50, 90, 95, 99)
            },
        }
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError("split manifest changed during audit")
    for name, info in manifest["files"].items():
        if sha256_file(split_dir / info["file"]) != before[name]:
            raise ValueError(f"{name} changed during audit")
    return {
        "schema_version": 1,
        "status": "passed" if overlap_records == 0 else "blocked_exact_input_overlap",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "full_train_and_validation_encoding_not_model_training_or_quality_evaluation",
        "split_manifest_sha256": manifest_hash,
        "split_files_sha256": before,
        "max_length": max_length, "content_budget": max_length - 2,
        "encoding_policy": "BOS + first (max_length - 2) body tokens + EOS; fixed right padding by collator",
        "special_token_ids": {name: getattr(tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")},
        "statistics": statistics,
        "unique_validation_input_sequences": len(validation_inputs),
        "train_records_with_exact_validation_input": overlap_records,
        "overlap_examples": overlap_examples,
        "limitations": [
            "SHA256 compares exact full unpadded input token IDs; no near-duplicate or document lineage guarantee.",
            "Candidate length only; GPU memory, throughput and final training recipe remain unverified.",
            "Raw text files are not modified by this audit; any overlap requires an explicit split-policy revision.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/processed/pretrain_mini_v1/split_seed43_v1")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.split_dir / f"encoding_audit_{args.max_length}.json"
    if output.exists():
        raise FileExistsError(f"use a new output file: {output}")
    tokenizer_files = sorted(args.tokenizer_dir.rglob("*.json"))
    if not tokenizer_files:
        raise ValueError("tokenizer JSON assets are missing")
    before = {path.relative_to(args.tokenizer_dir).as_posix(): sha256_file(path) for path in tokenizer_files}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    report = audit_encoding(args.split_dir, tokenizer, args.max_length, args.batch_size)
    after = {path.relative_to(args.tokenizer_dir).as_posix(): sha256_file(path) for path in args.tokenizer_dir.rglob("*.json")}
    if before != after:
        raise ValueError("tokenizer assets changed during audit")
    report["tokenizer_files_sha256"] = before
    report["runtime"] = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "numpy")}
    report["code_sha256"] = {
        "audit": sha256_file(Path(__file__)),
        "dataset_and_collator": sha256_file(ROOT / "src/minimind_lab/data/pretrain.py"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
