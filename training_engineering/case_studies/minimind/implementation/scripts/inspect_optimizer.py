
import torch

from torch import nn


w_sgd = nn.Parameter(torch.tensor(2.0))
w_adam = nn.Parameter(torch.tensor(2.0))
w_adam_decay = nn.Parameter(torch.tensor(2.0))

sgd = torch.optim.SGD([w_sgd], lr=0.1)
adam = torch.optim.AdamW([w_adam], lr=0.1, weight_decay=0.0)
adam_decay = torch.optim.AdamW([w_adam_decay], lr=0.1, weight_decay=0.2)

for name, parameter, optimizer in [
    ("SGD", w_sgd, sgd),
    ("AdamW", w_adam, adam),
    ("AdamW-decay", w_adam_decay, adam_decay)

]:
    print(f"name:{name},before:{parameter.item()}")
    befored = parameter.item()
    optimizer.zero_grad(set_to_none=True)
    loss = parameter.square()
    loss.backward()
    print(f"grad:{parameter.grad.item()}")
    optimizer.step()
    print(f"name:{name},after:{parameter.item()}")
    print(f"changed: after - before:{(parameter.item() - befored)}")

    if name.startswith("AdamW"):
        state = optimizer.state[parameter]
        print(f"step:{state['step'].item()}")
        print(f"m:{state['exp_avg'].item()}")
        print(f"v:{state['exp_avg_sq'].item()}")

        group = optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        weight_decay = group["weight_decay"]
        lr = group["lr"]
        eps = group["eps"]
        print(f"lr: {lr}")
        print(f"eps: {eps}")

        t = state["step"].item()
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()

        m_hat = m / ( 1 - beta1**t)
        v_hat = v / ( 1 - beta2**t)
        update =  lr * m_hat / ( v_hat**0.5 + eps)
        predicted_after = befored - update

        print(f"m_hat: {m_hat}")
        print(f"v_hat: {v_hat}")
        print(f"update : {update}")
        print(f"predicted_after: {predicted_after}")

        adaptive_update = lr * m_hat / (v_hat**0.5 + eps)
        decay_update = lr * weight_decay * befored
        predicted_after = befored - decay_update - adaptive_update

        print(f"weight_decay: {weight_decay}")
        print(f"decay_update: {decay_update}")
        print(f"adaptive_update: {adaptive_update}")
        print(f"predicted_after: {predicted_after}")
        print(f"actual_param: {parameter.item()}")





