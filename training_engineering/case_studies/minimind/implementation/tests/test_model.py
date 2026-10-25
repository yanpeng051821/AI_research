import copy

import pytest
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindBackbone, MinimindForCausalLM
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.rmsnorm import RMSNorm


@pytest.mark.parametrize("qk_norm", [False, True])
def test_causal_lm_propagates_qk_norm_config_and_backpropagates(qk_norm):
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=4,
        kv_heads=2,
        rms_norm_eps=3e-4,
        qk_norm=qk_norm,
    )
    model = MinimindForCausalLM(config)

    for block in model.backbone.blocks:
        for norm in (block.attention.q_norm, block.attention.k_norm):
            if qk_norm:
                assert isinstance(norm, RMSNorm)
                assert norm.eps == config.rms_norm_eps
                assert norm.weight.shape == (4,)
            else:
                assert isinstance(norm, torch.nn.Identity)

    input_ids = torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = input_ids.masked_fill(mask == 0, -100)
    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (2, 4, 32)
    loss = causal_lm_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    if qk_norm:
        for block in model.backbone.blocks:
            assert block.attention.q_norm.weight.grad.abs().sum() > 0
            assert block.attention.k_norm.weight.grad.abs().sum() > 0


def test_backbone_constructor_propagates_config_and_registers_independent_layers():
    config = ModelConfig(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        num_layers=3,
        num_heads=4,
        kv_heads=2,
        rope_base=10000.0,
    )
    model = MinimindBackbone(config)

    assert model.config is config
    assert model.embedding.weight.shape == (20, 16)
    assert len(model.blocks) == 3
    assert model.final_norm.eps == config.rms_norm_eps

    layer_parameter_ids = set()
    for block in model.blocks:
        assert block.attention_norm.eps == config.rms_norm_eps
        assert block.feed_forward.rms_norm.eps == config.rms_norm_eps
        assert block.attention.rope_base == config.rope_base
        assert block.attention.num_heads == config.num_heads
        assert block.attention.kv_heads == config.kv_heads
        assert block.feed_forward.swiglu.intermediate_size == config.intermediate_size
        parameter_ids = {id(parameter) for parameter in block.parameters()}
        assert layer_parameter_ids.isdisjoint(parameter_ids)
        layer_parameter_ids.update(parameter_ids)

    assert {id(parameter) for parameter in model.blocks.parameters()} == layer_parameter_ids
    assert layer_parameter_ids.issubset({id(parameter) for parameter in model.parameters()})


@pytest.mark.parametrize("causal", [True, False])
def test_backbone_forward_matches_component_chain_and_has_finite_gradients(causal):
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32, num_layers=3)
    model = MinimindBackbone(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    output = model(input_ids, causal=causal, attention_mask=mask)
    with torch.no_grad():
        expected = model.embedding(input_ids)
        for block in model.blocks:
            expected, _ = block(expected, causal=causal, attention_mask=mask)
        expected = model.final_norm(expected)

    assert output.shape == (2, 4, 16)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)

    # A varied upstream gradient avoids a nearly constant post-normalization norm loss.
    upstream = torch.randn_like(output)
    (output[mask.bool()] * upstream[mask.bool()]).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    torch.testing.assert_close(
        model.embedding.weight.grad[0], torch.zeros_like(model.embedding.weight.grad[0]),
    )
    assert model.embedding.weight.grad[1:7].abs().sum() > 0


@pytest.mark.parametrize("causal", [True, False])
def test_backbone_valid_outputs_match_unpadded_samples(causal):
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    model = MinimindBackbone(config)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    with torch.no_grad():
        output = model(input_ids, causal=causal, attention_mask=mask)
        for sample, length in enumerate([4, 2]):
            single_output = model(input_ids[sample:sample + 1, :length], causal=causal)
            torch.testing.assert_close(output[sample:sample + 1, :length], single_output)

        changed = input_ids.clone()
        changed[1, 2:] = torch.tensor([18, 19])
        changed_output = model(changed, causal=causal, attention_mask=mask)
    torch.testing.assert_close(output[mask.bool()], changed_output[mask.bool()])


def test_backbone_causal_output_ignores_future_tokens():
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    model = MinimindBackbone(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 18, 19]])

    with torch.no_grad():
        output = model(input_ids)
        changed_output = model(changed)
    torch.testing.assert_close(output[:, :2], changed_output[:, :2])


def test_backbone_propagates_unreadable_key_error():
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    model = MinimindBackbone(config)
    input_ids = torch.tensor([[1, 2, 3]])
    mask = torch.zeros_like(input_ids)

    with pytest.raises(ValueError, match="at least one readable key"):
        model(input_ids, attention_mask=mask)


def test_causal_lm_shares_parameter_and_projects_features_to_vocab():
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    model = MinimindForCausalLM(config)
    shared = model.backbone.embedding.weight
    assert model.lm_head.weight is shared
    assert model.lm_head.bias is None
    assert sum(parameter is shared for parameter in model.parameters()) == 1

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        hidden = model.backbone(input_ids)
        logits = model(input_ids)
    assert logits.shape == (2, 3, 20)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, hidden @ shared.T)
    torch.testing.assert_close(logits[0, 1, 7], (hidden[0, 1] * shared[7]).sum())


def test_shared_weight_gradient_sums_input_and_output_paths():
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=20, hidden_size=16, intermediate_size=32)
    tied = MinimindForCausalLM(config)
    untied = copy.deepcopy(tied)
    # Preserve identical values but separate the output parameter in the reference.
    untied.lm_head.weight = torch.nn.Parameter(untied.backbone.embedding.weight.detach().clone())
    assert untied.lm_head.weight is not untied.backbone.embedding.weight
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    tied_logits = tied(input_ids)
    untied_logits = untied(input_ids)
    torch.testing.assert_close(tied_logits, untied_logits)
    upstream = torch.randn_like(tied_logits)
    (tied_logits * upstream).sum().backward()
    (untied_logits * upstream).sum().backward()

    input_grad = untied.backbone.embedding.weight.grad
    output_grad = untied.lm_head.weight.grad
    assert input_grad is not None
    assert output_grad is not None
    torch.testing.assert_close(tied.lm_head.weight.grad, input_grad + output_grad)
    for name, parameter in tied.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    # This row was not looked up, but the output scoring path still uses it.
    torch.testing.assert_close(input_grad[19], torch.zeros_like(input_grad[19]))
    assert output_grad[19].abs().sum() > 0
