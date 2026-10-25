
import torch
from contextlib import nullcontext
import math

from minimind_lab.models.loss import causal_lm_loss_sum_and_count


def evaluate_nll(model, dataloader, *, device=None, autocast_dtype=None):
    if autocast_dtype not in (None, torch.bfloat16):
        raise ValueError("evaluation supports FP32 or BF16 autocast")
    was_training = model.training
    total_loss_sum = 0.0
    total_tokens = 0

    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                if device is not None:
                    batch = {key: value.to(device) for key, value in batch.items()}
                context = (
                    torch.autocast(device_type=batch["input_ids"].device.type, dtype=autocast_dtype)
                    if autocast_dtype is not None else nullcontext()
                )
                with context:
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss_logits = logits.float() if autocast_dtype is not None else logits
                loss_sum, token_count = causal_lm_loss_sum_and_count(loss_logits, batch["labels"])
                if not math.isfinite(loss_sum.item()):
                    raise FloatingPointError("non-finite validation loss")

                total_loss_sum += loss_sum.item()
                total_tokens += token_count.item()
                del logits, loss_logits, loss_sum, token_count

            
            if total_tokens == 0:
                raise ValueError("total_tokens must be greater than 0")
            
            return {"nll":total_loss_sum / total_tokens, "valid_tokens": total_tokens}

    finally:
        model.train(was_training)
    






