
import random
from pathlib import Path

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


