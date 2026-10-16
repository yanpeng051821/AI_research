
import torch

from torch import nn

from minimind_lab.models.loss import causal_lm_loss_sum_and_count



def train_update(
    model,
    optimizer,
    micro_batches,
    scheduler,
    max_grad_norm=1.0,
):
    if not micro_batches:
        raise ValueError("micro batches must be not empty")

    # 准备累积， 初始化
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_tokens=0
    total_loss_sum=0.0

    # 准备遍历
    for micro_batch in micro_batches:
        logits = model(input_ids=micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"])

        loss_sum, token_count = causal_lm_loss_sum_and_count(logits, micro_batch["labels"])
        total_loss_sum += loss_sum.item()
        total_tokens += token_count.item()

        loss_sum.backward()
    
    parameters = []
    for param in model.parameters():
        if param.grad is None:
            continue
        param.grad.div_(total_tokens)
        parameters.append(param)

    grad_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    lr_used = optimizer.param_groups[0]["lr"]

    optimizer.step()
    scheduler.step()

    lr_next = optimizer.param_groups[0]["lr"]

    return {
        "loss": total_loss_sum / total_tokens,
        "valid_tokens": total_tokens,
        "micro_batches": len(micro_batches),
        "grad_norm": grad_norm.item(),
        "lr_used": lr_used,
        "next_lr": lr_next,
    }








