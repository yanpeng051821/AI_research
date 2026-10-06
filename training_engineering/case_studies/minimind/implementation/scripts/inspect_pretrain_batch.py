import torch


from pathlib import Path
from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

tokenizer= AutoTokenizer.from_pretrained(
    ROOT / "assets/minimind_tokenizer",
    local_files_only=True,
)

dataset= PretrainDataset(
    ROOT / "data/audit/pretrain/accepted.jsonl",
    tokenizer,
    max_length=512,
)

# 1. 样本的数量
print(f"Dataset size: {len(dataset)}")

collator = PretrainCollator(tokenizer, max_length=512)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
)

batch = next(iter(dataloader))
attention_mask = batch["attention_mask"]
labels = batch["labels"]
input_ids = batch["input_ids"]
# 2. 三个张量的shape 和 dtype
print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
print(f"attention_mask shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")
print(f"labels shape: {labels.shape}, dtype: {labels.dtype} ")

# 3. 每条样本的有效token数
valid_tokens = attention_mask.sum(dim=1)
print(f"Valid tokens: {valid_tokens}")

# 4. causal shift 后的loss token数
shift_labels = labels[:, 1:]
loss_tokens = (shift_labels != -100).sum(dim=1)
print(f"loss tokens: {loss_tokens}")

# 5. padding均被设置为-100
padding_labels = labels[attention_mask == 0]
if not torch.all(padding_labels == -100).item():
    raise ValueError(f"Padding labels are not -100: {padding_labels}")
print("padding check passed")

# 6.真实位置的labels 都等于inputids 
loss_positions = labels != -100
if not torch.all(labels[loss_positions] == input_ids[loss_positions]).item():
    raise ValueError(f"Real labels are not equal to input_ids")
print("label check passed")

