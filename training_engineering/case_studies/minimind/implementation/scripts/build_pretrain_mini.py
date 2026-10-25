
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record

import hashlib, random, json


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_text_groups(source_path: Path) -> dict[str, list[int]]:
    text_groups = {}
    
    for item in iter_jsonl(source_path):
        if item["error_code"] is not None:
            raise ValueError(
                f"line {item['line_number']}: {item['error_code']}"
                )

        record = item["record"]
        errors_list = validate_pretrain_record(record)
        if errors_list:
            raise ValueError(
                f"line: {item['line_number']}, errors: {errors_list}"
                )

        text = record["text"]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if text_groups.get(text_hash) is None:
            text_groups[text_hash] = []
        
        text_groups[text_hash].append(item["line_number"])

    return text_groups


def load_review_constraints(
    review_path: Path,
    text_groups: dict[str, list[int]],
) -> tuple[set[str], set[str]]:
    
    excluded_hashes = set()
    train_only_hashes = set()

    for item in iter_jsonl(review_path):
        if item["error_code"] is not None:
            raise ValueError(
                f"line {item['line_number']}: {item['error_code']}"
                )
        record = item["record"]
        text_sha256 = record["text_sha256"]
        if text_groups.get(text_sha256) is None:
            raise ValueError(f"text_sha256: {text_sha256} not found in source")
        
        if record['source_line'] not in text_groups[text_sha256]:
            raise ValueError(f"source_line: {record['source_line']} not found in source")
        
        if (record["decision"] == "exclude" and record["allowed_split"] == "exclude_all"):
            excluded_hashes.add(text_sha256)
        elif (record["decision"] == "keep" and record["allowed_split"] == "train_only"):
            train_only_hashes.add(text_sha256)
        else:
            raise ValueError("unsupported review decision")

    if excluded_hashes & train_only_hashes :
        raise ValueError("excluded and train-only groups overlap")
    
    return excluded_hashes, train_only_hashes


def select_validation_groups(
    text_groups: dict[str, list[int]],
    excluded_hashes: set[str],
    train_only_hashes: set[str],
    validation_size: int,
    seed: int,
) -> set[str]:
    all_hashes = set(text_groups)
    if type(validation_size) is not int or type(seed) is not int:
        raise ValueError("validation_size and seed must be integers")
    if not excluded_hashes <= all_hashes:
        raise ValueError("excluded groups contain unknown hashes")

    if not train_only_hashes <= all_hashes:
        raise ValueError("train-only groups contain unknown hashes")

    if excluded_hashes & train_only_hashes:
        raise ValueError("excluded and train-only groups overlap")

    pool = sorted(all_hashes - excluded_hashes - train_only_hashes)

    if validation_size <= 0 or validation_size > len(pool):
        raise ValueError("validation_size is outside the available pool")

    eligible_hashes = all_hashes - excluded_hashes
    if validation_size >= len(eligible_hashes):
        raise ValueError("validation selection leaves no training group")

    rng = random.Random(seed)
    selected = rng.sample(pool, validation_size)
    return set(selected)

def write_split_records(
    source_path: Path,
    output_dir: Path,
    excluded_hashes: set[str],
    validation_hashes: set[str],
) -> dict[str, int]:
    if excluded_hashes & validation_hashes:
        raise ValueError("excluded and validation groups overlap")
    
    output_dir.mkdir(parents=True, exist_ok=False)
    
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    excluded_path = output_dir / "exclusions.jsonl"

    with (  
        train_path.open("w", encoding="utf-8", newline="\n") as train_file,
        validation_path.open("w", encoding="utf-8", newline="\n") as validation_file,
        excluded_path.open("w", encoding="utf-8", newline="\n") as excluded_file,
    ):
        train_count = 0
        validation_count = 0
        excluded_count = 0
        for item in iter_jsonl(source_path):
            if item["error_code"] is not None:
                raise ValueError(
                    f"line {item['line_number']}: {item['error_code']}"
                )

            record = item["record"]
            errors = validate_pretrain_record(record)
            if errors:
                raise ValueError(
                    f"line {item['line_number']}: {errors}"
                )

            text = record["text"]
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            row = {
                "source_line": item["line_number"],
                "text": text,
            }
            serialized = json.dumps(row, ensure_ascii=False) + "\n"
            if text_sha256 in excluded_hashes:
                excluded_file.write(serialized)
                excluded_count += 1
            elif text_sha256 in validation_hashes:
                validation_file.write(serialized)
                validation_count += 1
            else:
                train_file.write(serialized)
                train_count += 1

    return {
        "train": train_count,
        "validation": validation_count,
        "excluded": excluded_count,
    }

def build_mini_dataset(
    source_path: Path,
    review_path: Path,
    output_dir: Path,
    validation_size: int = 5000,
    seed: int = 43,
) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory: {output_dir}")

    text_groups = collect_text_groups(source_path)
    excluded_hashes, train_only_hashes = load_review_constraints(review_path, text_groups)
    validation_hashes = select_validation_groups(text_groups, excluded_hashes, train_only_hashes, validation_size, seed)
    
    train_hashes = set(text_groups) - excluded_hashes - validation_hashes

    expected_train = sum(len(text_groups[h]) for h in train_hashes)
    expected_validation = sum(len(text_groups[h]) for h in validation_hashes)
    expected_excluded = sum(len(text_groups[h]) for h in excluded_hashes)

    expected_counts = {
        "train": expected_train,
        "validation": expected_validation,
        "excluded": expected_excluded,
    }

    actual_counts = write_split_records(source_path, output_dir, excluded_hashes, validation_hashes)

    if expected_counts != actual_counts:
        raise ValueError("mismatch in expected and actual counts")
    
    return actual_counts


def verify_input_identity(source_path, review_path, source_manifest_path, candidate_path):
    paths = {
        "source": source_path, "reviews": review_path,
        "source_manifest": source_manifest_path, "review_candidates": candidate_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    anchor = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if hashes["source"] != anchor["source_sha256"]:
        raise ValueError("source SHA256 does not match source manifest")
    if hashes["review_candidates"] != anchor["candidate_sha256"]:
        raise ValueError("review candidate SHA256 does not match source manifest")
    if source_path.name != Path(anchor["source_path"]).name:
        raise ValueError("source filename does not match source manifest")

    candidates = {}
    for item in iter_jsonl(candidate_path):
        record = item["record"]
        if item["error_code"] or validate_pretrain_record(record):
            raise ValueError(f"invalid review candidate at line {item['line_number']}")
        line = record.get("source_line")
        if type(line) is not int or line <= 0 or line in candidates:
            raise ValueError("invalid or duplicate candidate source_line")
        candidates[line] = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    if len(candidates) != anchor["record_count"]:
        raise ValueError("review candidate count mismatch")

    reviewed_lines = set()
    for item in iter_jsonl(review_path):
        record = item["record"]
        if item["error_code"] or not isinstance(record, dict):
            raise ValueError(f"invalid review at line {item['line_number']}")
        expected = {
            "source_repo": anchor["source_repo"],
            "source_revision": anchor["source_revision"],
            "source_file": source_path.name,
            "candidate_sha256": hashes["review_candidates"],
        }
        for name, value in expected.items():
            if record.get(name) != value:
                raise ValueError(f"review {name} does not match source identity")
        line = record.get("source_line")
        if type(line) is not int or line in reviewed_lines:
            raise ValueError("invalid or duplicate review source_line")
        if line not in candidates or record.get("text_sha256") != candidates[line]:
            raise ValueError("review does not identify its original candidate")
        reviewed_lines.add(line)
    return anchor, hashes


def publish_mini_dataset(
    source_path, review_path, output_dir, source_manifest_path, candidate_path,
    validation_size=5000, seed=43,
):
    if output_dir.exists():
        raise FileExistsError(f"use a new output directory: {output_dir}")
    anchor, before = verify_input_identity(
        source_path, review_path, source_manifest_path, candidate_path,
    )
    print("Input identities verified; grouping and writing records...", flush=True)
    counts = build_mini_dataset(source_path, review_path, output_dir, validation_size, seed)
    if sum(counts.values()) != anchor["source_total_lines"]:
        raise ValueError("split counts do not cover the frozen source")
    _, after = verify_input_identity(source_path, review_path, source_manifest_path, candidate_path)
    if before != after:
        raise ValueError("input files changed during construction")

    filenames = {"train": "train.jsonl", "validation": "validation.jsonl", "excluded": "exclusions.jsonl"}
    manifest = {
        "schema_version": 1,
        "status": "text_split_complete_encoding_not_checked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": anchor["source_repo"],
        "source_revision": anchor["source_revision"],
        "source_file": source_path.name,
        "source_total_lines": anchor["source_total_lines"],
        "inputs": {name: {"path": str(path.resolve()), "sha256": before[name]} for name, path in {
            "source": source_path, "reviews": review_path,
            "source_manifest": source_manifest_path, "review_candidates": candidate_path,
        }.items()},
        "split_policy": {
            "group_identity": "SHA256 of exact original text encoded as UTF-8",
            "duplicate_policy": "retain all copies within the same split",
            "validation_method": "random.Random(seed).sample(sorted(eligible_text_hashes), size)",
            "validation_groups": validation_size,
            "seed": seed,
            "record_order": "original source_line ascending",
            "manual_constraints": "exclude_all or train_only applied to entire text group",
        },
        "counts": counts,
        "files": {name: {
            "file": filename, "sha256": sha256_file(output_dir / filename),
            "bytes": (output_dir / filename).stat().st_size,
        } for name, filename in filenames.items()},
        "runtime": {"python": platform.python_version(), "code_sha256": {
            "builder": sha256_file(Path(__file__)),
            "reader": sha256_file(ROOT / "src/minimind_lab/data/reader.py"),
            "validator": sha256_file(ROOT / "src/minimind_lab/data/validator.py"),
        }},
        "limitations": [
            "Exact original text isolation only; no original-document or near-duplicate guarantee.",
            "Candidate file is the historical manual-review anchor, not the full training population.",
            "Tokenization, truncation overlap, factual quality and training readiness are not certified here.",
        ],
    }
    temporary = output_dir / "manifest.pending.json"
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.rename(output_dir / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build a versioned full mini text split; do not start training.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/minimind/pretrain_t2t_mini.jsonl")
    parser.add_argument("--reviews", type=Path, default=ROOT.parent / "experiments/pretrain_512_v1/review_decisions.jsonl")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/processed/pretrain_candidate_v1/manifest.json")
    parser.add_argument("--candidates", type=Path, default=ROOT / "data/processed/pretrain_candidate_v1/candidates.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/pretrain_mini_v1/split_seed43_v1")
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    manifest = publish_mini_dataset(
        args.source, args.reviews, args.output_dir, args.source_manifest, args.candidates,
        args.validation_size, args.seed,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()


