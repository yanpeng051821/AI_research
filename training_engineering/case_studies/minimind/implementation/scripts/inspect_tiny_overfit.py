import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.generation import greedy_generate



torch.manual_seed(42)

config = ModelConfig(
    vocab_size=20,
    hidden_size=16,
    intermediate_size=32,
)
input_ids = torch.tensor([[1, 3, 4, 2]])
labels = input_ids.clone()

model = MinimindForCausalLM(config)
model.backbone.apply(initialize_weights)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2, weight_decay=0.0)

for i in range(100):
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"loss:{loss.item()}")

final_logits = model(input_ids)
fianl_loss = causal_lm_loss(final_logits, labels)
print(f"final_loss:{fianl_loss.item()}")

model.eval()
generated_ids = greedy_generate(
    model,
    torch.tensor([[1]]),
    max_new_tokens=9,
    eos_token_id=2,
)

print(f"generated_ids:{generated_ids}")




