import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

from transformers import AutoTokenizer

from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record


ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 512


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_records(path: Path) -> list[dict]:
    records = []
    for item in iter_jsonl(path):
        if item["error_code"] is not None:
            raise ValueError(f"{path}:{item['line_number']}: {item['error_code']}")
        if not isinstance(item["record"], dict):
            raise ValueError(f"{path}:{item['line_number']}: expected an object")
        records.append(item["record"])
    return records


def index_by_source(records: list[dict], label: str) -> dict[int, dict]:
    indexed = {}
    for record in records:
        source_line = record.get("source_line")
        if type(source_line) is not int or source_line <= 0:
            raise ValueError(f"{label}: invalid source_line {source_line!r}")
        if source_line in indexed:
            raise ValueError(f"{label}: duplicate source_line {source_line}")
        indexed[source_line] = record
    return indexed


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_splits(
    base_dir: Path,
    output_dir: Path,
    tokenizer_dir: Path,
    validation_size: int = 500,
    seed: int = 43,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory: {output_dir}")
    if type(validation_size) is not int or validation_size <= 0:
        raise ValueError("validation_size must be a positive integer")

    paths = {
        "candidates": base_dir / "candidates.jsonl",
        "source_manifest": base_dir / "manifest.json",
        "analysis_report": base_dir / "analysis/report.json",
        "retained_rows": base_dir / "analysis/retained_rows.jsonl",
        "length_filtered_rows": base_dir / "analysis/length_filtered_rows.jsonl",
        "review_decisions": base_dir / "analysis/review_decisions.jsonl",
    }
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_manifest = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    report = json.loads(paths["analysis_report"].read_text(encoding="utf-8"))
    candidate_hash = input_hashes["candidates"]
    if candidate_hash != source_manifest["candidate_sha256"]:
        raise ValueError("candidate SHA256 does not match source manifest")
    if report["max_length"] != MAX_LENGTH or report["content_budget"] != MAX_LENGTH - 2:
        raise ValueError("analysis report must use max_length=512 and content_budget=510")

    candidates = index_by_source(read_records(paths["candidates"]), "candidates")
    retained = index_by_source(read_records(paths["retained_rows"]), "retained rows")
    filtered = index_by_source(read_records(paths["length_filtered_rows"]), "filtered rows")
    reviews = index_by_source(read_records(paths["review_decisions"]), "reviews")
    if len(candidates) != source_manifest["record_count"] or len(candidates) != report["sample_count"]:
        raise ValueError("candidate count does not match input reports")
    if set(retained) & set(filtered) or set(retained) | set(filtered) != set(candidates):
        raise ValueError("length lists must partition all candidates")
    if len(retained) != report["retained_count"] or len(filtered) != report["length_filtered_count"]:
        raise ValueError("length counts do not match analysis report")
    if not set(reviews) <= set(candidates):
        raise ValueError("review contains an unknown source_line")

    # Re-encode original text to reject stale length lists before selecting data.
    tokenizer_files = {
        path.relative_to(tokenizer_dir).as_posix(): sha256_file(path)
        for path in sorted(tokenizer_dir.rglob("*.json"))
        if path.is_file()
    }
    if not tokenizer_files:
        raise ValueError("tokenizer directory has no JSON assets")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define BOS and EOS")

    lengths = {}
    seen_texts = set()
    for source_line, record in candidates.items():
        issues = validate_pretrain_record(record)
        if issues:
            raise ValueError(f"candidate source_line={source_line}: {issues}")
        if record["text"] in seen_texts:
            raise ValueError("duplicate candidate text; group duplicates before splitting")
        seen_texts.add(record["text"])
        count = len(tokenizer(
            record["text"], add_special_tokens=False, truncation=False
        )["input_ids"])
        lengths[source_line] = count
        statistics = retained.get(source_line, filtered.get(source_line))
        if count != statistics["token_count"]:
            raise ValueError(f"stale token count for source_line={source_line}")
        if (count <= MAX_LENGTH - 2) != (source_line in retained):
            raise ValueError(f"length partition is wrong for source_line={source_line}")
    if sum(lengths.values()) != report["total_tokens"]:
        raise ValueError("total tokens do not match analysis report")

    excluded_lines = set()
    train_only_lines = set()
    for source_line, review in reviews.items():
        if review["candidate_sha256"] != candidate_hash:
            raise ValueError("review candidate SHA256 does not match candidates")
        if review["text_sha256"] != text_sha256(candidates[source_line]["text"]):
            raise ValueError(f"review text SHA256 mismatch for source_line={source_line}")
        for key in ("source_repo", "source_revision"):
            if review[key] != source_manifest[key]:
                raise ValueError(f"review {key} does not match source manifest")
        if review["source_file"] != Path(source_manifest["source_path"]).name:
            raise ValueError("review source_file does not match source manifest")
        if review["decision"] == "exclude" and review["allowed_split"] == "exclude_all":
            excluded_lines.add(source_line)
        elif review["decision"] == "keep" and review["allowed_split"] == "train_only":
            train_only_lines.add(source_line)
        else:
            raise ValueError(f"unsupported review decision for source_line={source_line}")

    eligible_lines = set(retained) - excluded_lines
    if not train_only_lines <= eligible_lines:
        raise ValueError("a train-only reviewed record is not eligible")
    validation_pool = sorted(eligible_lines - train_only_lines)
    if validation_size > len(validation_pool) or validation_size >= len(eligible_lines):
        raise ValueError("validation_size leaves no training data or exceeds the pool")
    validation_lines = set(random.Random(seed).sample(validation_pool, validation_size))
    train_lines = eligible_lines - validation_lines

    # Files retain source order; the training sampler controls consumption order.
    def selected_records(source_lines: set[int]) -> list[dict]:
        return [
            {"source_line": n, "text": candidates[n]["text"]}
            for n in sorted(source_lines)
        ]

    train = selected_records(train_lines)
    validation = selected_records(validation_lines)
    if {r["text"] for r in train} & {r["text"] for r in validation}:
        raise ValueError("exact text overlap between training and validation")

    exclusions = []
    for source_line in sorted(set(candidates) - eligible_lines):
        reasons = []
        if source_line in filtered:
            reasons.append("content_tokens_exceed_510")
        if source_line in excluded_lines:
            reasons.append("manual_review:" + reviews[source_line]["reason_code"])
        exclusions.append({
            "source_line": source_line,
            "text_sha256": text_sha256(candidates[source_line]["text"]),
            "token_count": lengths[source_line],
            "reasons": reasons,
        })

    # Input checks finish before any output is created.
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "exclusions.jsonl", exclusions)
    output_hashes = {
        name: sha256_file(output_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "exclusions.jsonl")
    }

    def token_summary(source_lines: set[int]) -> dict:
        count = len(source_lines)
        content_tokens = sum(lengths[n] for n in source_lines)
        return {
            "record_count": count,
            "content_tokens": content_tokens,
            "valid_tokens_after_shift": content_tokens + count,
            "min_content_tokens": min(lengths[n] for n in source_lines),
            "max_content_tokens": max(lengths[n] for n in source_lines),
        }

    manifest = {
        "schema_version": 1,
        "snapshot_scope": "dataset_split_only_not_training_contract",
        "source": {
            name: source_manifest[name]
            for name in ("source_repo", "source_revision", "source_path", "source_sha256")
        },
        "inputs_sha256": input_hashes,
        "tokenizer_files_sha256": tokenizer_files,
        "builder_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        "max_length": MAX_LENGTH,
        "content_budget": MAX_LENGTH - 2,
        "long_record_policy": "exclude_whole_record_without_modifying_text",
        "split_unit": "candidate_record_identified_by_source_line",
        "split_seed": seed,
        "validation_size": validation_size,
        "sampling_method": "random.sample_from_sorted_eligible_unreviewed_source_lines",
        "file_order": "source_line_ascending",
        "train_only_reviewed_source_lines": sorted(train_only_lines),
        "counts": {
            "candidates": len(candidates),
            "length_eligible": len(retained),
            "length_excluded": len(filtered),
            "review_excluded": len(excluded_lines),
            "eligible": len(eligible_lines),
            "validation_pool": len(validation_pool),
            "excluded_unique": len(exclusions),
        },
        "train": token_summary(train_lines),
        "validation": token_summary(validation_lines),
        "outputs_sha256": output_hashes,
        "limitations": [
            "Content has not been fact-checked in full; validation may contain errors.",
            "Exact text overlap is checked; near duplicates and original document groups are not guaranteed isolated.",
            "Reviewed kept records are training-only; no independent final test set is created.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the week-7 pretraining data split.")
    parser.add_argument("--base-dir", type=Path, default=ROOT / "data/processed/pretrain_candidate_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/pretrain_512_v1")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    manifest = build_splits(
        args.base_dir, args.output_dir, args.tokenizer_dir,
        validation_size=args.validation_size, seed=args.seed,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "counts": manifest["counts"],
        "train": manifest["train"],
        "validation": manifest["validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()






