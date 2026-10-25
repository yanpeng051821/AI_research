from transformers import AutoTokenizer
from minimind_lab.data.reader import iter_jsonl
from pathlib import Path

import json
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    "assets/minimind_tokenizer",
    local_files_only=True,
)

source_path = "data/processed/pretrain_candidate_v1/candidates.jsonl"
report_path = Path("data/processed/pretrain_candidate_v1/analysis/report.json")
retained_rows_path = Path("data/processed/pretrain_candidate_v1/analysis/retained_rows.jsonl")
length_filtered_rows_path = Path("data/processed/pretrain_candidate_v1/analysis/length_filtered_rows.jsonl")

max_length = 512
content_budget = max_length - 2

rows = []
seen_texts = set()
duplicate_count = 0

for row in iter_jsonl(source_path):
    if row["error_code"] is not None:
        raise ValueError(
            f"line {row['line_number']}: {row['error_code']}"
        )

    content = row["record"]["text"]
    if content in seen_texts:
        duplicate_count += 1
    else:
        seen_texts.add(content)
    encoding = tokenizer(
        content,
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = encoding["input_ids"]
    token_count = len(input_ids)

    rows.append(
        {
            "line_number": row["line_number"],
            "source_line": row["record"]["source_line"],
            "char_count": len(content),
            "token_count": token_count,
            "drop_tokens": max(0, token_count - content_budget),
        }
    )

if not rows:
    raise ValueError("no records to inspect")


token_lengths = [row["token_count"] for row in rows]
sample_count = len(rows)
total_tokens = sum(row["token_count"] for row in rows)
truncated_count = sum(row["drop_tokens"] > 0 for row in rows)
total_dropped = sum(row["drop_tokens"] for row in rows)

percentiles = np.percentile(
    token_lengths,
    [50, 90, 95, 99],
    method="linear",
)

length_percentiles = {
    f"p{q}": float(value)
    for q, value in zip([50, 90, 95, 99], percentiles)
}



print(f"'sample_count': {sample_count}, 'total_tokens':{total_tokens}, 'truncated_count': {truncated_count}, 'total_dropped':{total_dropped}")
print("min_tokens:", min(token_lengths))
print("max_tokens:", max(token_lengths))
print("mean_tokens:", total_tokens / sample_count)
print("truncation_rate:", truncated_count / sample_count)
print("unique_text_count:", len(seen_texts))
print("duplicate_record_count:", duplicate_count)


retained_rows = [
    row for row in rows
    if row["token_count"] <= content_budget
]

length_filtered_rows = [
    row for row in rows
    if row["token_count"] > content_budget
]

report = {
    "source_path" : source_path,
    "max_length": max_length,
    "content_budget": content_budget,
    "sample_count": sample_count,
    "total_tokens": total_tokens,
    "min_tokens": min(token_lengths),
    "max_tokens": max(token_lengths),
    "mean_tokens": total_tokens / sample_count,
    "truncated_count": truncated_count,
    "total_dropped": total_dropped,
    "unique_text_count": len(seen_texts),
    "duplicate_record_count": duplicate_count,
    "retained_count": len(retained_rows),
    "length_filtered_count": len(length_filtered_rows),
    "length_percentiles": length_percentiles,
    "percentile_method": "linear",
}

report_path.parent.mkdir(parents=True, exist_ok=True)

report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

with retained_rows_path.open("w", encoding="utf-8", newline="\n") as f:
    for row in retained_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

with length_filtered_rows_path.open("w", encoding="utf-8", newline="\n") as f:
    for row in length_filtered_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
