import math
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from minimind_lab.data.pretrain import PretrainCollator
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.model import MinimindForCausalLM


def test_uniform_logits_have_log_vocab_loss():
    logits = torch.zeros((1, 5, 20), dtype=torch.float64)
    labels = torch.tensor([[1, 10, 11, 2, -100]])

    loss = causal_lm_loss(logits, labels)

    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.tensor(math.log(20), dtype=logits.dtype))


@pytest.mark.parametrize("ignore_index", [-100, -1])
def test_loss_and_gradients_match_per_token_nll(ignore_index):
    torch.manual_seed(42)
    logits = torch.randn((2, 5, 4), dtype=torch.float64, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([
        [0, 1, 2, 3, ignore_index],
        [0, 2, ignore_index, ignore_index, ignore_index],
    ])

    actual = causal_lm_loss(logits, labels, ignore_index=ignore_index)
    token_losses = []
    for sample in range(2):
        for position in range(4):
            target = labels[sample, position + 1].item()
            if target != ignore_index:
                log_probs = F.log_softmax(reference[sample, position], dim=-1)
                token_losses.append(-log_probs[target])
    assert len(token_losses) == 4
    expected = torch.stack(token_losses).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_dropped_and_ignored_positions_do_not_affect_loss_or_receive_gradient():
    torch.manual_seed(42)
    logits = torch.randn((2, 5, 4), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, -100], [0, 2, -100, -100, -100]])
    ignored = labels[:, 1:] == -100

    loss = causal_lm_loss(logits, labels)
    changed = logits.detach().clone()
    changed[:, -1] = torch.randn_like(changed[:, -1]) * 100
    shifted_changed = changed[:, :-1]
    shifted_changed[ignored] = torch.randn_like(shifted_changed[ignored]) * 100
    changed_labels = labels.clone()
    # Position zero is dropped, so even an out-of-vocabulary value is unused here.
    changed_labels[:, 0] = 999
    torch.testing.assert_close(loss, causal_lm_loss(changed, changed_labels))

    loss.backward()
    torch.testing.assert_close(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))
    ignored_grad = logits.grad[:, :-1][ignored]
    torch.testing.assert_close(ignored_grad, torch.zeros_like(ignored_grad))
    assert logits.grad[:, :-1][~ignored].abs().sum() > 0


def test_batch_can_include_a_sample_with_no_valid_targets():
    torch.manual_seed(42)
    logits = torch.randn((2, 4, 5), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, -100], [0, 1, 2, -100]])

    actual = causal_lm_loss(logits, labels)
    expected = causal_lm_loss(logits[1:2], labels[1:2])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad[0], torch.zeros_like(logits.grad[0]))


@pytest.mark.parametrize("ignore_index", [-100, -1])
@pytest.mark.parametrize("first_label_valid", [True, False])
def test_loss_rejects_no_valid_targets_after_shift(ignore_index, first_label_valid):
    logits = torch.zeros((2, 3, 5))
    labels = torch.full((2, 3), ignore_index)
    if first_label_valid:
        labels[:, 0] = 1

    with pytest.raises(ValueError, match="non-ignore_index token"):
        causal_lm_loss(logits, labels, ignore_index=ignore_index)


@pytest.mark.parametrize("length", [0, 1])
def test_loss_rejects_sequence_too_short_for_next_token_prediction(length):
    logits = torch.zeros((2, length, 5))
    labels = torch.zeros((2, length), dtype=torch.long)

    with pytest.raises(ValueError, match="sequence length must be at least 2"):
        causal_lm_loss(logits, labels)


def test_pretrain_batch_model_loss_and_backward_are_connected():
    torch.manual_seed(42)
    collator = PretrainCollator(SimpleNamespace(pad_token_id=0), max_length=6)
    batch = collator([{"input_ids": [1, 3, 4, 2]}, {"input_ids": [1, 5, 2]}])
    config = ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32)
    model = MinimindForCausalLM(config)

    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = causal_lm_loss(logits, batch["labels"])
    assert logits.shape == (2, 6, 8)
    assert (batch["labels"][:, 1:] != -100).sum().item() == 5
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.lm_head.weight.grad.abs().sum() > 0
