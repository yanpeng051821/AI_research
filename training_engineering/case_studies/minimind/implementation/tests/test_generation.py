import pytest
import torch
from torch import nn

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.generation import greedy_generate
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.model import MinimindForCausalLM


class ScriptedLM(nn.Module):
    """Choose known tokens so stopping does not depend on random model weights."""

    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, input_ids, attention_mask):
        call_index = len(self.calls)
        self.calls.append(
            (input_ids.clone(), attention_mask.clone(), torch.is_grad_enabled())
        )
        scores = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        # Earlier positions deliberately prefer a different token.
        scores[:, :-1, 7] = 100.0
        scores[:, -1, self.tokens[call_index]] = 10.0
        return scores + self.anchor


def test_generation_feeds_back_full_prefix_and_rebuilds_mask():
    model = ScriptedLM([4, 5, 6])
    prefix = torch.tensor([[1, 3]])
    original = prefix.clone()

    result = greedy_generate(model, prefix, max_new_tokens=3)

    torch.testing.assert_close(result, torch.tensor([[1, 3, 4, 5, 6]]))
    torch.testing.assert_close(prefix, original)
    assert result.dtype == prefix.dtype
    assert result.device == prefix.device
    assert len(model.calls) == 3
    for index, (received, mask, grad_enabled) in enumerate(model.calls):
        torch.testing.assert_close(received, result[:, :2 + index])
        torch.testing.assert_close(mask, torch.ones_like(received))
        assert not grad_enabled


@pytest.mark.parametrize("tokens", [[2, 5, 6], [4, 2, 6]])
def test_generation_stops_after_appending_first_generated_eos(tokens):
    model = ScriptedLM(tokens)
    prefix = torch.tensor([[1, 3]])

    result = greedy_generate(model, prefix, max_new_tokens=3, eos_token_id=2)

    stop_count = tokens.index(2) + 1
    expected = torch.tensor([[1, 3] + tokens[:stop_count]])
    torch.testing.assert_close(result, expected)
    assert len(model.calls) == stop_count
    assert result[0, -1].item() == 2


def test_generation_without_eos_keeps_going_when_token_two_is_selected():
    model = ScriptedLM([2, 4, 5])

    result = greedy_generate(model, torch.tensor([[1]]), max_new_tokens=3)

    torch.testing.assert_close(result, torch.tensor([[1, 2, 4, 5]]))
    assert len(model.calls) == 3


def test_generation_zero_new_tokens_returns_independent_prefix_without_forward():
    model = ScriptedLM([])
    prefix = torch.tensor([[1, 3]])

    result = greedy_generate(model, prefix, max_new_tokens=0)

    torch.testing.assert_close(result, prefix)
    assert result.data_ptr() != prefix.data_ptr()
    assert model.calls == []


@pytest.mark.parametrize(
    "prefix",
    [
        None,
        torch.tensor(1),
        torch.tensor([1, 3]),
        torch.ones((2, 3), dtype=torch.long),
        torch.ones((1, 2, 3), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
    ],
    ids=["none", "scalar", "one-dimensional", "batch-two", "rank-three", "empty"],
)
def test_generation_rejects_invalid_prefix_before_forward(prefix):
    model = ScriptedLM([])

    with pytest.raises(ValueError, match="shape"):
        greedy_generate(model, prefix, max_new_tokens=1)
    assert model.calls == []


def test_generation_rejects_negative_budget_with_accurate_message():
    model = ScriptedLM([])

    with pytest.raises(ValueError, match="non-negative|at least 0"):
        greedy_generate(model, torch.tensor([[1, 3]]), max_new_tokens=-1)
    assert model.calls == []


@pytest.mark.parametrize("training", [True, False])
def test_generation_preserves_model_mode_parameters_and_existing_gradients(training):
    model = ScriptedLM([4])
    model.train(training)
    model.anchor.grad = torch.tensor(0.25)
    before = model.anchor.detach().clone()

    greedy_generate(model, torch.tensor([[1, 3]]), max_new_tokens=1)

    assert model.training is training
    torch.testing.assert_close(model.anchor, before)
    torch.testing.assert_close(model.anchor.grad, torch.tensor(0.25))
    assert not model.calls[0][2]


def test_generation_does_not_print(capsys):
    greedy_generate(ScriptedLM([4]), torch.tensor([[1, 3]]), max_new_tokens=1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_generation_real_model_matches_manual_unpadded_greedy_steps():
    torch.manual_seed(42)
    model = MinimindForCausalLM(
        ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    )
    model.backbone.apply(initialize_weights)
    model.eval()
    prefix = torch.tensor([[1, 3, 4]])

    expected = prefix.clone()
    with torch.no_grad():
        for _ in range(3):
            last_scores = model(expected)[:, -1, :]
            selected = last_scores.argmax(dim=-1, keepdim=True)
            expected = torch.cat((expected, selected), dim=1)

    actual = greedy_generate(model, prefix, max_new_tokens=3)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 6)
    assert all(parameter.grad is None for parameter in model.parameters())
