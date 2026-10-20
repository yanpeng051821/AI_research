
import torch

from minimind_lab.models.loss import causal_lm_loss_sum_and_count


def evaluate_nll(model, dataloader):
    was_training = model.training
    total_loss_sum = 0.0
    total_tokens = 0

    model.eval()

    try:
        with torch.no_grad():
            for batch in dataloader:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss_sum, token_count = causal_lm_loss_sum_and_count(logits, batch["labels"])

                total_loss_sum += loss_sum.item()
                total_tokens += token_count.item()

            
            if total_tokens == 0:
                raise ValueError("total_tokens must be greater than 0")
            
            return {"nll":total_loss_sum / total_tokens, "valid_tokens": total_tokens}

    finally:
        model.train(was_training)
    






