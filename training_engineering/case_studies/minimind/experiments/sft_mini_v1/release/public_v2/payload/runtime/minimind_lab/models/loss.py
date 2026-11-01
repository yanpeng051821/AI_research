import torch

from torch.nn import functional as F

def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    loss_sum, loss_count = causal_lm_loss_sum_and_count(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    loss = loss_sum / loss_count

    return loss


def causal_lm_loss_sum_and_count(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 长度检查、维度检查、shift
    if logits.shape[-2] < 2:
        raise ValueError("sequence length must be at least 2")

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    ignored = shift_labels == ignore_index
    
    if ignored.all():
        raise ValueError("labels must have at least 1 non-ignore_index token")

    loss_sum = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), 
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum"
    )

    valid = shift_labels != ignore_index
    loss_count = valid.sum()
    return loss_sum, loss_count









