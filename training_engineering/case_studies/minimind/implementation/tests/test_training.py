import copy

import pytest
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.training.loop import train_epoch
from minimind_lab.training.step import train_update
from minimind_lab.training.evaluate import evaluate_nll

def test_train_update_matches_full_batch():
    # 随机数和模型准备
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32)
    reference_model = MinimindForCausalLM(config).double()
    actual_model = copy.deepcopy(reference_model)

    ref_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-3, weight_decay=0.0)
    actual_optimizer = torch.optim.AdamW(actual_model.parameters(), lr=1e-3, weight_decay=0.0)

    ref_scheduler = torch.optim.lr_scheduler.StepLR(ref_optimizer, step_size=1, gamma=0.5)
    actual_scheduler = torch.optim.lr_scheduler.StepLR(actual_optimizer, step_size=1, gamma=0.5)

    # 数据准备
    input_ids = torch.tensor([
        [1, 3, 2, 0, 0, 0, 0],
        [1, 4, 5, 6, 7, 3, 2],
    ])
    attention_mask = input_ids != 0
    labels = input_ids.clone()
    labels[~attention_mask] = -100

    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels

    micro_batches = [{key: value[i:i+1] for key, value in batch.items()} for i in range(2)]

    #  参考路径
    reference_model.train()
    ref_logits = reference_model(input_ids=input_ids, attention_mask=attention_mask)
    ref_loss = causal_lm_loss(ref_logits, labels)
    ref_loss.backward()
    ref_grad_norm = torch.nn.utils.clip_grad_norm_(
        reference_model.parameters(), 
        max_norm=1.0, 
        error_if_nonfinite=True,
    )
    ref_lr_used = ref_optimizer.param_groups[0]["lr"]
    ref_optimizer.step()
    ref_scheduler.step()
    ref_lr_next = ref_optimizer.param_groups[0]["lr"]

    # 实际路径
    actual_model.train()
    actual_history = train_update(actual_model,actual_optimizer ,micro_batches, actual_scheduler, max_grad_norm=1.0)

    for ref_param, actual_param in zip(reference_model.parameters(), actual_model.parameters()):
        torch.testing.assert_close(ref_param, actual_param, rtol=1e-5, atol=1e-7)

    assert actual_history["loss"] == pytest.approx(ref_loss.item(), rel=1e-5)
    assert actual_history["valid_tokens"] == 8
    assert actual_history["micro_batches"] == 2
    assert actual_history["lr_used"] == ref_lr_used
    assert actual_history["next_lr"] == ref_lr_next
    assert actual_history["grad_norm"] == pytest.approx(
        ref_grad_norm.item(), 
        rel=1e-5,
        abs=1e-7,
    )
    assert actual_scheduler.last_epoch == 1


@pytest.fixture
def micro_batches():
    input_ids = torch.tensor([
        [1, 3, 2, 0, 0, 0, 0],
        [1, 4, 5, 6, 7, 3, 2],
    ])
    attention_mask = input_ids != 0
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return [
        {key: value[i:i + 1] for key, value in batch.items()}
        for i in range(2)
    ]


def _make_model():
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32)
    return MinimindForCausalLM(config).double()


def _make_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.02
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    return optimizer, scheduler


@pytest.mark.parametrize("batches", [None, []], ids=["none", "empty"])
def test_train_update_rejects_empty_input_without_updating(batches):
    model = _make_model()
    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    scheduler_before = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(ValueError, match="micro batches"):
        train_update(model, optimizer, batches, scheduler)

    assert not optimizer.state
    assert scheduler.state_dict() == scheduler_before
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


def test_train_update_preserves_frozen_weights(micro_batches):
    model = _make_model()
    frozen = model.backbone.embedding.weight
    frozen.requires_grad_(False)
    frozen_before = frozen.detach().clone()
    trainable = model.backbone.final_norm.weight
    trainable_before = trainable.detach().clone()
    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    model.eval()

    metrics = train_update(model, optimizer, micro_batches, scheduler)

    assert model.training
    assert model.lm_head.weight is frozen
    assert not frozen.requires_grad
    assert frozen.grad is None
    assert frozen not in optimizer.state
    torch.testing.assert_close(frozen, frozen_before, rtol=0, atol=0)
    assert not torch.equal(trainable, trainable_before)
    assert metrics["valid_tokens"] == 8
    assert scheduler.last_epoch == 1


def test_train_epoch_updates_tail(micro_batches):
    actual_model = _make_model()
    reference_model = copy.deepcopy(actual_model)
    actual_optimizer, actual_scheduler = _make_optimizer_and_scheduler(actual_model)
    ref_optimizer, ref_scheduler = _make_optimizer_and_scheduler(reference_model)
    short_batch, long_batch = micro_batches
    batches = [short_batch, long_batch, short_batch, long_batch, short_batch]

    history = train_epoch(
        actual_model,
        actual_optimizer,
        iter(batches),
        actual_scheduler,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,
    )

    for window in (batches[:2], batches[2:4]):
        train_update(
            reference_model, ref_optimizer, window, ref_scheduler,
            max_grad_norm=0.5,
        )
    before_tail = {
        name: param.detach().clone()
        for name, param in reference_model.named_parameters()
    }
    train_update(
        reference_model, ref_optimizer, batches[4:], ref_scheduler,
        max_grad_norm=0.5,
    )

    assert len(history) == 3
    assert [item["micro_batches"] for item in history] == [2, 2, 1]
    assert [item["valid_tokens"] for item in history] == [8, 8, 2]
    assert [item["lr_used"] for item in history] == pytest.approx(
        [1e-3, 5e-4, 2.5e-4]
    )
    assert [item["next_lr"] for item in history] == pytest.approx(
        [5e-4, 2.5e-4, 1.25e-4]
    )
    assert actual_scheduler.last_epoch == ref_scheduler.last_epoch == 3
    reference_parameters = dict(reference_model.named_parameters())
    for name, param in actual_model.named_parameters():
        torch.testing.assert_close(
            param, reference_parameters[name], rtol=1e-5, atol=1e-7
        )
    assert any(
        not torch.equal(param, before_tail[name])
        for name, param in actual_model.named_parameters()
    )
    assert all(state["step"].item() == 3 for state in actual_optimizer.state.values())


def test_train_epoch_empty_input_does_not_update():
    model = _make_model()
    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    scheduler_before = copy.deepcopy(scheduler.state_dict())

    assert train_epoch(model, optimizer, iter(()), scheduler) == []

    assert not optimizer.state
    assert scheduler.state_dict() == scheduler_before
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("accumulation_steps", [0, -1])
def test_train_epoch_rejects_nonpositive_accumulation(accumulation_steps):
    model = _make_model()
    optimizer, scheduler = _make_optimizer_and_scheduler(model)
    scheduler_before = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_epoch(
            model, optimizer, iter(()), scheduler,
            gradient_accumulation_steps=accumulation_steps,
        )

    assert not optimizer.state
    assert scheduler.state_dict() == scheduler_before


def test_evaluate_nll_matches_full_batch(micro_batches):
    model = _make_model()
    model.train()

    full_batch = {
        key: torch.cat([batch[key] for batch in micro_batches], dim=0)
        for key in micro_batches[0]
    }

    with torch.no_grad():
        logits = model(
            input_ids=full_batch["input_ids"],
            attention_mask=full_batch["attention_mask"],
        )
        expected_nll = causal_lm_loss(
            logits, full_batch["labels"]
        ).item()

    result = evaluate_nll(model, micro_batches)

    assert result["nll"] == pytest.approx(expected_nll, rel=1e-5, abs=1e-7)
    assert result["valid_tokens"] == 8
    assert model.training




