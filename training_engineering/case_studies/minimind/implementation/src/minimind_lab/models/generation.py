import torch

from torch import nn

def greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1  or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, T] with T > 0")
    
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    
    generated_ids = input_ids.detach().clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token],dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated_ids




