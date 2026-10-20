from transformers import AutoTokenizer
from minimind_lab.data.reader import iter_jsonl

tokenizer = AutoTokenizer.from_pretrained(
    "assets/minimind_tokenizer",
    local_files_only=True,
)

source_path = "data/source_samples/pretrain_sample.jsonl"
max_length = 512
content_budget = max_length - 2

rows = []

for row in iter_jsonl(source_path):
    if row["error_code"] is not None:
        raise ValueError(
            f"line {row['line_number']}: {row['error_code']}"
        )

    content = row["record"]["text"]
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
            "char_count": len(content),
            "token_count": token_count,
            "drop_tokens": max(0, token_count - content_budget),
        }
    )

for row in rows:
    print(row)


sample_count = len(rows)
total_tokens = sum(row["token_count"] for row in rows)
truncated_count = sum(row["drop_tokens"] > 0 for row in rows)
total_dropped = sum(row["drop_tokens"] for row in rows)
print(f"'sample_count': {sample_count}, 'total_tokens':{total_tokens}, 'truncated_count': {truncated_count}, 'total_dropped':{total_dropped}")