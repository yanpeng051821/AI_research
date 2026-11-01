import torch

from torch import nn
from minimind_lab.models.rope import (build_rope_angles, apply_rope)
from minimind_lab.models.rmsnorm import RMSNorm

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal:bool = True,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_t = k.transpose(-2, -1)

    # 精度转换成32位
    scaled_scores = q @ k_t
    
    if scaled_scores.dtype in (torch.float16, torch.bfloat16):
        scaled_scores = scaled_scores.float()
    
    scaled_scores = scaled_scores/ (q.shape[-1] ** 0.5)

    # scores: [B, T, T]
    allowed = torch.ones_like(
        scaled_scores,
        dtype=torch.bool,
    )

    if causal:
        query_len = q.shape[-2]
        key_len = k.shape[-2]

        causal_allowed = torch.ones(
            (query_len, key_len),
            dtype=torch.bool,
            device=q.device,
        ).tril()

        allowed = allowed & causal_allowed

    if attention_mask is not None:
        expected_shape = (
            q.shape[0],
            k.shape[-2],
        )

        if attention_mask.shape != expected_shape:
            raise ValueError(
                "attention_mask must have shape "
                f"{expected_shape}, got {tuple(attention_mask.shape)}"
            )

        if attention_mask.device != q.device:
            raise ValueError(
                "attention_mask must be on the same device as q"
            )
    
        key_allowed = attention_mask.to(torch.bool)

        while key_allowed.ndim < scaled_scores.ndim:
            key_allowed = key_allowed.unsqueeze(1)
        
        allowed = allowed & key_allowed
    
    # 检查，每个query 至少能读取一个key，否则立即报错
    has_readable_key = allowed.any(dim=-1)
    if not has_readable_key.all():
        raise ValueError("Each query must have at least one readable key")

    masked_scores = scaled_scores.masked_fill(~allowed, float("-inf"))

    weights = torch.softmax(masked_scores, dim=-1)
    # weights：[B, T, T]

    output = weights.to(dtype=v.dtype) @ v
    # output:[B, T, d_v]

    return output, weights

def repeat_kv(
        x:torch.Tensor,
        n_rep:int,
    ) -> torch.Tensor:
    if x.ndim != 4 or n_rep <=0:
        raise ValueError("x must be 4D and n_rep must be positive")
    
    if n_rep == 1:
        return x
    
    expanded = x.repeat_interleave(repeats=n_rep, dim=2)
    return expanded


class SingleHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size:int, head_dim:int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, hidden_size, bias=False)

    def forward(self, x:torch.Tensor, causal:bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_size]
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        
        # xq: [B, T, head_dim]
        # xk: [B, T, head_dim]
        # xv: [B, T, head_dim]
        output, weights = scaled_dot_product_attention(xq, xk, xv, causal)

        # output: [B, T, head_dim]
        # weights: [B, T, T]
        final_output = self.o_proj(output)
        # final_output: [B, T, hidden_size]

        return final_output, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size and num_heads must be positive; "
                "hidden_size must be divisible by num_heads"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_size]
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        # xq: [B, T, hidden_size]
        # xk: [B, T, hidden_size]
        # xv: [B, T, hidden_size]

        # 拆头
        B, T, H, d = xq.shape[0], xq.shape[1], self.num_heads, self.head_dim
        xq_split = xq.reshape(B, T, H, d)
        xk_split = xk.reshape(B, T, H, d)
        xv_split = xv.reshape(B, T, H, d)

        # 换轴
        xq_pos = xq_split.transpose(1, 2)
        xk_pos = xk_split.transpose(1, 2)
        xv_pos = xv_split.transpose(1, 2)
        # shape: [B, H, T, d]

        # attention计算
        output, weights = scaled_dot_product_attention(xq_pos, xk_pos, xv_pos,causal)
        # output: [B, H, T, d]
        # weights: [B, H, T, T]

        # 输出换轴合并
        merged_output = torch.cat(
            [output[:,h] for h in range(H)],
            dim=-1
        )
        # merged_output: [B, T, hidden_size]

        # 适配层
        final_output = self.o_proj(merged_output)
        # final_output: [B, T, hidden_size]

        return final_output, weights


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        kv_heads,
        rope_base: float = 1e6,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or kv_heads <=0 or hidden_size % num_heads != 0 or num_heads % kv_heads != 0:
            raise ValueError(
                "hidden_size and num_heads must be positive; "
                "hidden_size must be divisible by num_heads; "
                "num_heads must be divisible by kv_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // kv_heads
        self.rope_base = rope_base

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()


        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even when using RoPE"
            )
        
        if rope_base <= 0:
            raise ValueError("rope_base must be positive")

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim*self.num_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim*self.kv_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim*self.kv_heads, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x:torch.Tensor, causal:bool=True, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, hidden_size]
        B, T = x.shape[0], x.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q: [B, T, head_dim*num_heads], k/v :[B, T, head_dim*kv_heads]

        # 拆分
        q_split = q.reshape(B, T, self.num_heads, self.head_dim)
        k_split = k.reshape(B, T, self.kv_heads, self.head_dim)
        v_split = v.reshape(B, T, self.kv_heads, self.head_dim)
        # q_split: [B, T, H, head_dim]; kv_split: [B, T, kv_head, head_dim]

        # 归一化
        q_norm = self.q_norm(q_split)
        k_norm = self.k_norm(k_split)

        # 重复k / v
        k_repeat = repeat_kv(k_norm, n_rep=self.n_rep)
        v_repeat = repeat_kv(v_split, n_rep=self.n_rep)
        # kv_repeat:[B, T, H, head_dim]

        # 换轴
        q_trans = q_norm.transpose(1, 2)
        k_trans = k_repeat.transpose(1, 2)
        v_trans = v_repeat.transpose(1, 2)
        # [B, H, T, head_dim]

        # 计算旋转角度，seq-len / head-dim
        angles = build_rope_angles(
            seq_len=T,
            head_dim=self.head_dim,
            rope_base=self.rope_base,
            dtype=(torch.float64
                if q_trans.dtype==torch.float64
                else torch.float32
                ),
            device=q_trans.device,
        )

        # 计算rope
        q_rotated, k_rotated = apply_rope(q_trans,k_trans ,angles)


        # 计算attention
        output, weights = scaled_dot_product_attention(q_rotated, k_rotated, v_trans, causal=causal, attention_mask=attention_mask)
        # output: [B, H, T, head_dim]
        # weights: [B, H, T, T]

        # 合并输出
        output = torch.cat(
            [output[:,h,:,:]
            for h in range(self.num_heads)],
            dim=-1
        )
        # B, T, num_heads*head_dim

        final_output = self.o_proj(output)

        return final_output, weights







