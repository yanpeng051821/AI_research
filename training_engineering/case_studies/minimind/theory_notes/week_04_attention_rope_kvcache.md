# 第 4 周：Attention、RoPE、KV Cache

## 1. 本周学习目标

第 4 周专门拆 `Attention` 内部。

本周重点不是背公式，而是把 MiniMind 源码里的 shape 和工程逻辑走通：

- q/k/v 如何从 hidden state 投影出来。
- 为什么 q heads 和 kv heads 可以不同。
- RoPE 在哪里作用。
- `repeat_kv` 为什么存在。
- causal mask 如何保证只能看过去。
- KV cache 如何加速自回归生成。

本周结束后，你应该能独立写出：

```text
x -> q/k/v -> q_norm/k_norm -> RoPE -> KV cache -> repeat_kv -> attention -> o_proj
```

## 2. Attention 源码主线

源码位置：

```text
/Users/yanpeng/Documents/PythonProjects/datawhale/minimind/model/model_minimind.py
```

核心 forward：

```python
bsz, seq_len, _ = x.shape
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
xq, xk = self.q_norm(xq), self.k_norm(xk)
cos, sin = position_embeddings
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
past_kv = (xk, xv) if use_cache else None
xq, xk, xv = (
    xq.transpose(1, 2),
    repeat_kv(xk, self.n_rep).transpose(1, 2),
    repeat_kv(xv, self.n_rep).transpose(1, 2)
)
output = attention(xq, xk, xv)
output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
output = self.o_proj(output)
return output, past_kv
```

一句话版本：

```text
Attention 把每个 token 的 hidden state 变成 q/k/v，通过 q 和 k 算关系权重，再用权重加权 v，最后投影回 hidden_size。
```

## 3. 默认参数

MiniMind 默认主线可以先按这些参数理解：

```text
hidden_size C = 768
num_attention_heads Hq = 8
num_key_value_heads Hkv = 4
head_dim D = 96
```

因为：

```text
D = C / Hq = 768 / 8 = 96
```

GQA 中：

```text
q 有 8 个 head
k/v 只有 4 个 head
每组 k/v 会服务多个 q head
```

这里：

```text
n_rep = Hq / Hkv = 8 / 4 = 2
```

## 4. q/k/v Shape 表

设输入：

```text
x: [B, T, C]
```

经过线性投影：

```text
q_proj(x): [B, T, Hq * D] = [B, T, 768]
k_proj(x): [B, T, Hkv * D] = [B, T, 384]
v_proj(x): [B, T, Hkv * D] = [B, T, 384]
```

再 reshape：

```text
xq: [B, T, Hq, D]  = [B, T, 8, 96]
xk: [B, T, Hkv, D] = [B, T, 4, 96]
xv: [B, T, Hkv, D] = [B, T, 4, 96]
```

然后 RoPE 作用在：

```text
xq 和 xk
```

不作用在：

```text
xv
```

原因是：

```text
q/k 用来计算 attention score，需要感知位置关系；
v 是被加权汇总的信息内容，不直接参与位置匹配。
```

## 5. repeat_kv 的作用

因为 q 有 8 个 head，k/v 只有 4 个 head，attention 计算前要把 k/v 扩展到 8 个 head。

```text
repeat_kv 前:
xk: [B, T, 4, 96]
xv: [B, T, 4, 96]

repeat_kv 后:
xk: [B, T, 8, 96]
xv: [B, T, 8, 96]
```

再转置成 attention 常用格式：

```text
xq: [B, 8, T, 96]
xk: [B, 8, T, 96]
xv: [B, 8, T, 96]
```

这就是 GQA 的核心：

```text
减少 k/v head 数，节省 KV cache 和计算/显存，同时让多个 q head 共享一组 k/v。
```

## 6. Attention Score Shape

attention score：

```python
scores = (xq @ xk.transpose(-2, -1)) / sqrt(head_dim)
```

shape 是：

```text
xq: [B, H, T, D]
xk.transpose(-2, -1): [B, H, D, T]
scores: [B, H, T, T]
```

含义：

```text
每个 batch、每个 head、每个 query token，对所有 key token 的相关性分数。
```

softmax 后：

```text
attention weights: [B, H, T, T]
```

再乘 v：

```text
weights @ xv
[B, H, T, T] @ [B, H, T, D] -> [B, H, T, D]
```

最后转回：

```text
[B, H, T, D]
-> transpose
-> [B, T, H, D]
-> reshape
-> [B, T, H*D] = [B, T, C]
-> o_proj
-> [B, T, C]
```

## 7. 本周 Checklist

- [ ] 能写出 q/k/v 投影后的 shape。
- [ ] 能解释为什么 MiniMind 是 GQA。
- [ ] 能解释 `repeat_kv` 的作用。
- [ ] 能解释 RoPE 作用在 q/k 而不是 v。
- [ ] 能写出 attention scores 的 shape。
- [ ] 能解释 causal mask。
- [ ] 能解释 KV cache 在 generate 中为什么能省计算。

## 8. 当前进度

状态：进行中

已完成：

- [x] 第 3 周模型主干。
- [x] 已能复述 `input_ids -> embedding -> blocks -> lm_head -> loss`。

待完成：

- [ ] 逐行拆解 `Attention.forward`。
- [ ] 深入 RoPE。
- [ ] 深入 causal mask。
- [ ] 深入 KV cache。

