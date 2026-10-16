
from minimind_lab.training.step import train_update


def train_epoch(
    model,
    optimizer,
    dataloader,
    scheduler,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
):
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    pending_batches = []
    history = []

    for batch in dataloader:
        pending_batches.append(batch)
        if len(pending_batches) == gradient_accumulation_steps:
            result = train_update(
                model,
                optimizer,
                pending_batches,
                scheduler,
                max_grad_norm=max_grad_norm,
            )
            history.append(result)
            pending_batches = []
    
    if pending_batches:
        result = train_update(
            model,
            optimizer,
            pending_batches,
            scheduler,
            max_grad_norm=max_grad_norm,
        )
        history.append(result)
    
    return history
