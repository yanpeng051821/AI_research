
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss_sum_and_count

torch.manual_seed(42)

config = ModelConfig(
    vocab_size=8,
    hidden_size=16,
    intermediate_size=32,
)

model = MinimindForCausalLM(config).double()

input_ids = torch.tensor([
    [1, 3, 2, 0, 0, 0, 0],
    [1, 4, 5, 6, 7, 3, 2],
])

attention_mask = input_ids != 0

labels = input_ids.clone()
labels[~attention_mask] = -100

logits = model(input_ids=input_ids,attention_mask=attention_mask)

loss_sum, token_count = causal_lm_loss_sum_and_count(logits, labels)

mean_loss = loss_sum / token_count
print(f"loss_sum:{loss_sum}, token_count:{token_count}, mean_loss:{mean_loss}")

mean_loss.backward()

reference_grads = {
    name: param.grad.detach().clone()
    for name, param in model.named_parameters()
}

# 实际路径
model.zero_grad(set_to_none=True)

total_tokens = 0
total_loss_sum = 0.0

for i in range(input_ids.shape[0]):
    micro_input_ids = input_ids[i:i+1]
    micro_labels = labels[i:i+1]
    micro_attention_mask = attention_mask[i:i+1]

    logits = model(input_ids=micro_input_ids, attention_mask=micro_attention_mask)
    loss, count = causal_lm_loss_sum_and_count(logits, micro_labels)
    total_loss_sum += loss.item()
    total_tokens += count.item()

    loss.backward()

for param in model.parameters():
    param.grad.div_(total_tokens)


for name, parameter in model.named_parameters():
    torch.testing.assert_close(
        parameter.grad,
        reference_grads[name],
        rtol=1e-5,
        atol=1e-7,
        msg=lambda message: f"{name}: {message}",
    )

model.zero_grad(set_to_none=True)
wrong_mean_loss = 0.0

for i in range(input_ids.shape[0]):
    micro_input_ids = input_ids[i:i+1]
    micro_labels = labels[i:i+1]
    micro_attention_mask = attention_mask[i:i+1]

    logits = model(input_ids=micro_input_ids, attention_mask=micro_attention_mask)
    loss, count = causal_lm_loss_sum_and_count(logits, micro_labels)
    wrong_contribution = loss / count / input_ids.shape[0]
    wrong_mean_loss += wrong_contribution.item()
    wrong_contribution.backward()


max_grad_error = max(
    (parameter.grad - reference_grads[name]).abs().max().item()
    for name, parameter in model.named_parameters()
)

print(f"reference mean loss: {mean_loss.item()}")
print(f"wrong mean loss: {wrong_mean_loss}")
print(f"wrong max gradient error: {max_grad_error}")






