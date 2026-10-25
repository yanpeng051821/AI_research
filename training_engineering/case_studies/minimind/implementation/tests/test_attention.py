

import torch, pytest

from minimind_lab.models.attention import scaled_dot_product_attention, SingleHeadSelfAttention, MultiHeadSelfAttention, repeat_kv, GroupQueryAttention
from minimind_lab.models.rope import apply_rope, build_rope_angles
from torch.nn import functional as F



def test_causal_attention_matches_known_values():
    q = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    k = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    v = torch.tensor([[
        [10., 0.],
        [0., 20.],
        [30., 30.],
    ]])

    output, weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (1, 3, 2)
    assert weights.shape == (1, 3, 3)

    expected = torch.tensor([[
        [10.0000,  0.0000],
        [ 3.3024, 13.3952],
        [17.5872, 20.0698],
    ]])

    torch.testing.assert_close(
        output, 
        expected, 
        atol=1e-4, 
        rtol=0
    )


def test_noncausal_attention_matches_known_values():
    q = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    k = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    v = torch.tensor([[
        [10., 0.],
        [0., 20.],
        [30., 30.],
    ]])

    output, weights = scaled_dot_product_attention(q, k, v, causal=False)

    expected = torch.tensor([[
        [16.0445, 15.9889],
        [14.0111, 20.0556],
        [17.5872, 20.0698],
    ]])

    torch.testing.assert_close(output, expected, atol=1e-4, rtol=0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert weights[0, 0, 2] > 0


def test_attention_batch_matches_individual_samples():
    torch.manual_seed(7)

    q = torch.randn(2, 3, 2)
    k = torch.randn(2, 3, 2)
    v = torch.randn(2, 3, 2)

    batch_output, batch_weights = scaled_dot_product_attention(q, k, v)

    for i in range(2):
        single_output, single_weights = scaled_dot_product_attention(q[i:i+1], k[i:i+1], v[i:i+1])
        torch.testing.assert_close(batch_output[i:i+1], single_output)
        torch.testing.assert_close(batch_weights[i:i+1], single_weights)


def test_causal_attention_ignores_future_changes():
    q = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    k = torch.tensor([[
        [1., 0.],
        [0., 1.],
        [1., 1.],
    ]])

    v = torch.tensor([[
        [10., 0.],
        [0., 20.],
        [30., 30.],
    ]])

    original_output, original_weights = scaled_dot_product_attention(q, k, v)

    k_changed = k.clone()
    v_changed = v.clone()

    k_changed[:, 2, :] = torch.tensor([100., -100.])
    v_changed[:, 2, :] = torch.tensor([-500., 700.])

    changed_output, changed_weights = scaled_dot_product_attention(q, k_changed, v_changed)

    torch.testing.assert_close(
        changed_output[:, :2, :], original_output[:, :2, :]
    )
    torch.testing.assert_close(
        changed_weights[:, :2, :], original_weights[:, :2, :]
    )
    assert not torch.allclose(
        changed_output[:, 2, :], original_output[:, 2, :]
    )


def test_attention_forward_and_gradients_match_torch():
    torch.manual_seed(42)

    q = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)

    actual, _ = scaled_dot_product_attention(q, k, v)
    reference = F.scaled_dot_product_attention(
        rq,
        rk,
        rv,
        dropout_p=0.0,
        is_causal=True,
    )

    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)

    actual.square().sum().backward()
    reference.square().sum().backward()

    torch.testing.assert_close(q.grad, rq.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(k.grad, rk.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        v.grad, rv.grad, atol=1e-10, rtol=1e-10
    )

def test_single_head_attention_shapes_and_gradients():
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, requires_grad=True)

    model = SingleHeadSelfAttention(4, 2)

    output, weights = model(x)

    assert output.shape == (2, 3, 4)
    assert weights.shape == (2, 3, 3)

    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    

def test_single_head_attention_matches_explicit_projections():
    torch.manual_seed(42)
    model = SingleHeadSelfAttention(4, 2).double()

    x = torch.randn(2, 3, 4, dtype=torch.float64)

    actual, _ = model(x)

    xq = x @ model.q_proj.weight.T
    xk = x @ model.k_proj.weight.T
    xv = x @ model.v_proj.weight.T
    
    context = F.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        dropout_p=0.0,
        is_causal=True,
    )

    expected = context @ model.o_proj.weight.T

    torch.testing.assert_close(
        actual, 
        expected, 
        atol=1e-10, 
        rtol=1e-10
        )


def test_single_head_attention_ignores_future_inputs():
    torch.manual_seed(42)

    model = SingleHeadSelfAttention(4, 2)
    x = torch.randn(2, 3, 4)
    original_output, original_weights = model(x)

    x_changed = x.clone()
    x_changed[:, 2, :] += 100.0
    changed_output, changed_weights = model(x_changed)

    torch.testing.assert_close(original_output[:, :2, :], changed_output[:, :2, :]) 
    torch.testing.assert_close(original_weights[:, :2, :], changed_weights[:, :2, :])
    assert not torch.allclose(
        original_output[:, 2, :],
        changed_output[:, 2, :],
    )


@pytest.mark.parametrize("causal", [True, False])
def test_multihead_attention_shapes_and_gradients(causal):
    torch.manual_seed(42)
    model = MultiHeadSelfAttention(4, 2)
    x = torch.randn(2, 3, 4, requires_grad=True)
    
    output, weights = model(x, causal)

    # 检查形状
    assert output.shape == (2, 3, 4)
    assert weights.shape == (2, 2, 3, 3)

    # 检查权重与模式
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
    )

    if causal:
        assert (weights.triu(diagonal=1) == 0).all()
    else:
        assert ((weights[..., 0, 2]) > 0).all()

    # 验证反向传播
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

@pytest.mark.parametrize(
    "hidden_size,num_heads",
    [
        (0, 2),
        (4, 0),
        (-4, 2),
        (4, -2),
        (5, 2),
    ],
)
def test_multihead_attention_rejects_invalid_config(
    hidden_size, num_heads
):
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(hidden_size, num_heads)


@pytest.mark.parametrize("causal", [True, False])
def test_multihead_attention_matches_per_head_reference(causal):
    torch.manual_seed(42)
    
    # 实际路径
    model = MultiHeadSelfAttention(4, 2).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64)

    actual, _ = model(x, causal=causal)

    # 参考路径:显式投影
    q = x @ model.q_proj.weight.T
    k = x @ model.k_proj.weight.T
    v = x @ model.v_proj.weight.T

    head_outputs = []
    for h in range(model.num_heads):
        start = h * model.head_dim
        end = start + model.head_dim

        head_output = F.scaled_dot_product_attention(
            q[...,start:end],
            k[...,start:end],
            v[...,start:end],
            dropout_p=0.0,
            is_causal=causal,
        )
        head_outputs.append(head_output)

    merged_output = torch.cat(head_outputs, dim=-1)
    expected = merged_output @ model.o_proj.weight.T

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "n_rep",
    [1, 2, 3]
)
def test_repeat_kv_preserves_head_groups(n_rep):
    x = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    actual = repeat_kv(x, n_rep)

    assert actual.shape == (1, 2, 2*n_rep, 2)
    assert actual.device == x.device
    assert actual.dtype == x.dtype

    source_head_num = x.shape[2]
    for source_head in range(source_head_num):
        for copy_index in range(n_rep):
            output_head = source_head*n_rep + copy_index
            torch.testing.assert_close(
                actual[:,:,output_head ,:],
                x[:,:,source_head,:]
            )

    if n_rep == 1:
        assert actual is x



@pytest.mark.parametrize("n_rep", [1, 2, 3])
def test_repeat_kv_accumulates_gradients(n_rep):
    x = torch.randn(1, 2, 2, 2, requires_grad=True)
    actual = repeat_kv(x, n_rep)
    actual.sum().backward()

    torch.testing.assert_close(
        x.grad,
        torch.full_like(x, float(n_rep)),
    )


@pytest.mark.parametrize(
    ("hidden_size", "num_heads", "kv_heads", "rope_base"),
    [
        (0, 4, 2, 1e6),
        (8, 0, 2, 1e6),
        (8, 4, 0, 1e6),
        (10, 4, 2, 1e6),
        (8, 4, 3, 1e6),
        (12, 4, 2, 1e6),
        (8, 4, 2, 0.0),
    ],
)
def test_gqa_rejects_invalid_configuration(
    hidden_size: int,
    num_heads: int,
    kv_heads: int,
    rope_base: float,
):
    with pytest.raises(ValueError):
        GroupQueryAttention(
            hidden_size,
            num_heads,
            kv_heads,
            rope_base=rope_base,
        )

def test_gqa_builds_fp32_rope_angles_for_bfloat16_inputs(monkeypatch):
    import minimind_lab.models.attention as attention_module

    torch.manual_seed(42)
    original_builder = attention_module.build_rope_angles
    observed = []

    def capture_angles(*args, **kwargs):
        angles = original_builder(*args, **kwargs)
        observed.append(angles)
        return angles

    monkeypatch.setattr(attention_module, "build_rope_angles", capture_angles)
    model = GroupQueryAttention(16, 4, 2, qk_norm=True).to(dtype=torch.bfloat16)
    x = torch.randn(1, 260, 16, dtype=torch.bfloat16, requires_grad=True)
    output, _ = model(x)

    assert len(observed) == 1
    assert observed[0].dtype == torch.float32
    assert observed[0][257, 0].item() == 257
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("causal", [True, False])
def test_gqa_shapes_and_gradients(causal):
    torch.manual_seed(42)

    model = GroupQueryAttention(8, 4, 2)
    x = torch.randn(2, 3, 8, requires_grad=True)
    output, weights = model(x, causal)

    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 4, 3, 3)

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))

    if causal:
        torch.testing.assert_close(weights.triu(diagonal=1), torch.zeros_like(weights))
    else:
        assert (weights[...,0,2] > 0).all() 
    
    output.square().sum().backward()

    assert x.grad is not None
    assert model.q_proj.weight.grad is not None
    assert model.k_proj.weight.grad is not None
    assert model.v_proj.weight.grad is not None
    assert model.o_proj.weight.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.q_proj.weight.grad).all()
    assert torch.isfinite(model.k_proj.weight.grad).all()
    assert torch.isfinite(model.v_proj.weight.grad).all()
    assert torch.isfinite(model.o_proj.weight.grad).all()


@pytest.mark.parametrize("causal", [True, False])
def test_gqa_matches_per_head_reference(causal):
    torch.manual_seed(42)
    model = GroupQueryAttention(8, 4, 2).double()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    # 实际输出
    actual, _ = model(x, causal)

    # 参考输出
    q = x @ model.q_proj.weight.T
    k = x @ model.k_proj.weight.T
    v = x @ model.v_proj.weight.T

    batch_size, seq_len = x.shape[:2]
    q = q.reshape(
        batch_size,
        seq_len,
        model.num_heads,
        model.head_dim,
    ).transpose(1, 2)
    k = k.reshape(
        batch_size,
        seq_len,
        model.kv_heads,
        model.head_dim,
    ).transpose(1, 2)
    v = v.reshape(
        batch_size,
        seq_len,
        model.kv_heads,
        model.head_dim,
    ).transpose(1, 2)

    angles = build_rope_angles(
        seq_len=seq_len,
        head_dim=model.head_dim,
        rope_base=model.rope_base,
        dtype=q.dtype,
        device=q.device,
    )
    q, k = apply_rope(q, k, angles)

    expected_output = []

    for h in range(model.num_heads):
        kv_head = h // model.n_rep

        head_output = F.scaled_dot_product_attention(
            q[:, h],
            k[:, kv_head],
            v[:, kv_head],
            dropout_p=0.0,
            is_causal=causal,
        )

        expected_output.append(head_output)

    merged_output = torch.cat(expected_output, dim=-1)
    expected = merged_output @ model.o_proj.weight.T

    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


