import torch

from pathlib import Path
from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.generation import greedy_generate

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


# 创建配置
config = ModelConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32)
torch.manual_seed(42)
model = MinimindForCausalLM(config)
model.backbone.apply(initialize_weights)
logits = model(batch["input_ids"], causal=True, attention_mask=batch["attention_mask"])

loss = causal_lm_loss(logits, batch["labels"])
loss.backward()

print(f"logits.shape:{logits.shape},loss.item:{loss.item()}")
print(f"embed_grad_nrom:{model.lm_head.weight.grad.norm().item()}")
print("embedding std:", model.backbone.embedding.weight.std().item())
print("shared weight:", model.lm_head.weight is model.backbone.embedding.weight)
print("final norm weight:", model.backbone.final_norm.weight)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)

before_weight = model.backbone.embedding.weight.detach().clone()

optimizer.step()
after_weight = model.backbone.embedding.weight.detach().clone()

max_change = (after_weight - before_weight).abs().max().item()

print(f"max_change:{max_change}")
print(f"shared after step:{model.lm_head.weight is model.backbone.embedding.weight}")

with torch.no_grad():
    after_logits = model(batch["input_ids"], causal=True, attention_mask=batch["attention_mask"])
    after_loss = causal_lm_loss(after_logits, batch["labels"])

print(f"before_loss:{loss.item()}")
print(f"after_loss:{after_loss.item()}")


with torch.no_grad():
    prefix_ids = batch["input_ids"][:1, :8]
    prefix_mask = torch.ones_like(prefix_ids)
    prefix_logits = model(prefix_ids, causal=True, attention_mask=prefix_mask)

next_token = prefix_logits[:, -1:].argmax(dim=-1)
print(f"next_token.shape:{next_token.shape}")

extended_ids = torch.cat([prefix_ids, next_token], dim=-1)

print("prefix:", tokenizer.decode(prefix_ids[0].tolist()))
print("extended:", tokenizer.decode(extended_ids[0].tolist()))


generated_ids = prefix_ids.clone()
max_new_token = 5
model.eval()

generated_ids = greedy_generate(
    model,
    prefix_ids,
    max_new_tokens=5,
    eos_token_id=tokenizer.eos_token_id,
)

print(f"generated:{tokenizer.decode(generated_ids[0].tolist())}")
print("new token count:", generated_ids.shape[1] - prefix_ids.shape[1])

