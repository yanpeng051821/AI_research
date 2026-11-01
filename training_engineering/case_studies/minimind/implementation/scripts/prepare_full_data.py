"""Audit, deduplicate, tokenize and index both pinned full JSONL sources."""

import argparse
from array import array
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer
from minimind_lab.data.sft import build_assistant_labels, prepare_sft_messages
from minimind_lab.data.sft_protocol import check_sft_protocol
from minimind_lab.data.validator import validate_pretrain_record, validate_sft_record


REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
FILES = {
    "pretrain": ("pretrain_t2t.jsonl", 8275074893,
                 "31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d", 340),
    "sft": ("sft_t2t.jsonl", 14096018369,
            "b5dab3d590b3b4ae3bfeec891ba0c76b0de5acbec4f4b078dcd3912eb1735c79", 768),
}
SPLIT_MODULUS = 100
VALIDATION_BUCKETS = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_digest(stage, record):
    if stage == "pretrain":
        value = record["text"].encode("utf-8")
    else:
        value = json.dumps(record["conversations"], sort_keys=True,
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).digest()


def split_for_digest(digest):
    bucket = int.from_bytes(digest[:8], "big") % SPLIT_MODULUS
    return "validation" if bucket < VALIDATION_BUCKETS else "train"


def encode_sft_for_audit(record, tokenizer, max_length):
    messages, tools = prepare_sft_messages(record["conversations"])
    markers = (tokenizer.bos_token, tokenizer.eos_token)

    def reject_boundary_literals(value):
        if isinstance(value, str):
            if any(marker and marker in value for marker in markers):
                raise ValueError("message_boundary_literal_in_source")
        elif isinstance(value, dict):
            for key, item in value.items():
                reject_boundary_literals(key)
                reject_boundary_literals(item)
        elif isinstance(value, list):
            for item in value:
                reject_boundary_literals(item)

    reject_boundary_literals(messages)
    reject_boundary_literals(tools)
    issues, notices = check_sft_protocol(messages, tools)
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assistant_prefix = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    )["input_ids"]
    message_end = tokenizer(
        f"{tokenizer.eos_token}\n", add_special_tokens=False
    )["input_ids"]
    labels = build_assistant_labels(ids, assistant_prefix, message_end)
    full_length = len(ids)
    ids = ids[:max_length]
    labels = labels[:max_length]
    valid_targets = sum(label != -100 for label in labels[1:])
    return full_length, valid_targets, issues, notices


def prepare_one(stage, source, output, tokenizer):
    filename, expected_size, expected_sha, max_length = FILES[stage]
    source = Path(source)
    if source.stat().st_size != expected_size or sha256_file(source) != expected_sha:
        raise ValueError(f"pinned source size/SHA256 mismatch: {source}")
    out = Path(output) / stage
    if out.exists():
        raise FileExistsError(f"refusing to overwrite derived data: {out}")
    out.mkdir(parents=True)
    db_path = out / "dedup.sqlite3"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
    counts = Counter()
    protocol_codes = Counter()
    length_histogram = Counter()
    examples = []
    indices = {name: array("Q") for name in ("train", "validation")}
    targets = Counter()
    last_report = time.monotonic()
    offset = 0
    exclusions = (out / "excluded.jsonl").open("w", encoding="utf-8", newline="\n")

    def exclude(line_number, code, digest=None, details=None):
        item = {"source_line": line_number, "reason": code}
        if digest is not None:
            item["record_sha256"] = digest.hex()
        if details:
            item["details"] = details
        exclusions.write(json.dumps(item, ensure_ascii=False) + "\n")

    with source.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            line_offset = offset
            offset += len(raw)
            counts["source_lines"] += 1
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["invalid_json_or_utf8"] += 1
                exclude(line_number, "invalid_json_or_utf8")
                continue

            issues = validate_pretrain_record(record) if stage == "pretrain" else validate_sft_record(record)
            if issues:
                counts["structural_excluded"] += 1
                for issue in issues:
                    counts[f"structural:{issue}"] += 1
                exclude(line_number, "structural", details=issues)
                if len(examples) < 50:
                    examples.append({"line": line_number, "category": "structural", "issues": issues})
                continue

            digest = record_digest(stage, record)
            try:
                db.execute("INSERT INTO seen VALUES (?)", (digest,))
            except sqlite3.IntegrityError:
                counts["duplicate_excluded"] += 1
                exclude(line_number, "duplicate", digest)
                continue

            if stage == "pretrain":
                text = record["text"]
                if not text.strip():
                    counts["empty_text_excluded"] += 1
                    exclude(line_number, "empty_text", digest)
                    continue
                ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                full_length = len(ids) + 2
                if full_length > max_length:
                    counts["truncated_records"] += 1
                encoded_length = min(full_length, max_length)
                valid_targets = encoded_length - 1
                issues = []
                notices = []
            else:
                try:
                    encoded_length, valid_targets, issues, notices = encode_sft_for_audit(
                        record, tokenizer, max_length
                    )
                except Exception as exc:
                    category = f"adapter_or_template:{type(exc).__name__}:{str(exc)[:180]}"
                    counts[category] += 1
                    if len(examples) < 50:
                        examples.append({"line": line_number, "category": "adapter_or_template",
                                         "error_type": type(exc).__name__, "message": str(exc)[:300]})
                    exclude(line_number, "adapter_or_template", digest,
                            {"error_type": type(exc).__name__, "message": str(exc)[:300]})
                    continue
                if encoded_length > max_length:
                    counts["truncated_records"] += 1
                for issue in issues:
                    protocol_codes[issue["code"]] += 1
                if issues:
                    counts["protocol_issue_records_encoded"] += 1
                if notices:
                    counts["protocol_notice_records_encoded"] += 1
                if valid_targets == 0:
                    counts["zero_target_excluded"] += 1
                    exclude(line_number, "zero_target_after_truncation", digest)
                    continue

            split = split_for_digest(digest)
            indices[split].append(line_offset)
            counts[f"{split}_records"] += 1
            targets[split] += valid_targets
            length_histogram[min(encoded_length, max_length)] += 1
            counts["unique_encoded_records"] += 1
            counts["valid_targets_total"] += valid_targets
            if line_number % 10000 == 0:
                db.commit()
            if time.monotonic() - last_report >= 10:
                db.commit()
                status = {
                    "status": "running", "stage": stage, "source_revision": REVISION,
                    "processed_lines": line_number, "source_lines": counts["source_lines"],
                    "unique_encoded_records": counts["unique_encoded_records"],
                    "train_records": counts["train_records"],
                    "validation_records": counts["validation_records"],
                    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                (out / "progress.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                last_report = time.monotonic()
    db.commit()
    db.close()
    exclusions.close()

    for split, values in indices.items():
        with (out / f"{split}.offsets.u64").open("wb") as handle:
            values.tofile(handle)
    actual_source_sha = sha256_file(source)
    if actual_source_sha != expected_sha or source.stat().st_size != expected_size:
        raise ValueError("source changed during audit")
    report = {
        "status": "completed",
        "stage": stage,
        "source": {"repo": "jingyaogong/minimind_dataset", "revision": REVISION,
                   "filename": filename, "size_bytes": expected_size, "sha256": expected_sha,
                   "actual_sha256": actual_source_sha},
        "tokenizer_dir": str(ROOT / "assets/minimind_tokenizer"),
        "max_length": max_length,
        "validation_fraction_rule": "SHA256 digest bucket 0 of 100 after exact/canonical deduplication",
        "deduplication": "pretrain exact UTF-8 text SHA256; SFT canonical conversations JSON SHA256",
        "counts": dict(counts),
        "valid_targets": dict(targets),
        "encoded_length_histogram_capped_at_max_length": dict(sorted(length_histogram.items())),
        "protocol_issue_codes_on_encoded_sft_records": dict(protocol_codes),
        "review_samples": examples,
        "offset_index_sha256": {
            split: sha256_file(out / f"{split}.offsets.u64")
            for split in ("train", "validation")
        },
        "exclusion_ledger": {"file": "excluded.jsonl", "sha256": sha256_file(out / "excluded.jsonl"),
                             "bytes": (out / "excluded.jsonl").stat().st_size},
        "limitations": [
            "This is full structural/encoding accounting, not a fact, safety, or licensing review.",
            "Tool protocol diagnostics do not automatically exclude records if the template encoded them.",
            "Unencodable adapter/template records are counted separately and require explicit interpretation.",
            "Dedup uses SHA256 identities; a cryptographic collision is treated as a duplicate.",
        ],
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({"status": "completed", "report": "audit_report.json"}, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(FILES), default=tuple(FILES))
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "assets/minimind_tokenizer")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define BOS, EOS and PAD")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        filename = FILES[stage][0]
        report = prepare_one(stage, args.source_dir / filename, args.output_dir, tokenizer)
        print(json.dumps({"stage": stage, "counts": report["counts"],
                          "valid_targets": report["valid_targets"]}), flush=True)


if __name__ == "__main__":
    main()
