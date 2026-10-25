
import random
import hashlib
import json
from pathlib import Path

from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record

source_path = Path("data/raw/minimind/pretrain_t2t_mini.jsonl")
sample_size = 10_000
seed = 42

total_lines = 0
with source_path.open("r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1

if sample_size < 1 or sample_size > total_lines:
        raise ValueError("sample_size between 1 and total_lines")

# 创建独立的随机数状态
rng = random.Random(seed)
select_lines = rng.sample(
    range(1, total_lines + 1),
    k=sample_size
)

select_lines.sort()

print("total_lines:", total_lines)
print("selected_count:", len(select_lines))
print("first_10_selected_lines:", select_lines[:10])


selected_line_set = set(select_lines)
candidates = []

for line in iter_jsonl(source_path):
    if not line["line_number"] in selected_line_set:
        continue

    if line["error_code"] is not None:
        raise ValueError(
            f"line {line['line_number']}: {line['error_code']}"
        )

    record = line["record"]

    issues = validate_pretrain_record(record)
    if issues:
        raise ValueError(
            f"line {line['line_number']}: {issues}"
        )

    candidates.append(
        {
            "source_line": line["line_number"],
            "text": record["text"]
        }
    )

if len(candidates) != sample_size:
    raise ValueError(
        f"expected {sample_size} records, got {len(candidates)}"
    )

print("extracted_count:", len(candidates))
print("first_source_line:", candidates[0]["source_line"])
    
output_dir = Path("data/processed/pretrain_candidate_v1")
output_dir.mkdir(parents=True, exist_ok=True)

candidate_path = output_dir / "candidates.jsonl"
manifest_path = output_dir / "manifest.json"

with candidate_path.open("w", encoding="utf-8", newline="\n") as f:
    for candidate in candidates:
        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

with source_path.open("rb") as f:
    source_sha256 = hashlib.file_digest(f, "sha256").hexdigest()

with candidate_path.open("rb") as f:
    candidate_path_sha256 = hashlib.file_digest(f, "sha256").hexdigest()

manifest = {
    "source_repo": "jingyaogong/minimind_dataset",
    "source_revision": "312afb4f76391145c6902f765bb51691c09a12f5",
    "source_path": source_path.as_posix(),
    "source_sha256": source_sha256,
    "source_total_lines": total_lines,
    "sampling_method": "random.sample_without_replacement",
    "seed": seed,
    "record_count": len(candidates),
    "record_order": "source_line_ascending",
    "candidate_file": candidate_path.name,
    "candidate_sha256": candidate_path_sha256
}


manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")