# 第 4 周：Attention 与完整 Decoder 实践教程

> 教程重写：2026-10-19。本周把逐位置组件组装成可训练、可生成的 tiny 语言模型；QA 在文末。

## 1. 本周任务与准备

第三周的 FFN 能加工每个位置的特征，却不会主动读取其他位置。本周先解决跨位置的信息获取，再加入位置编码与可见性规则，最后接上预测目标和生成。

~~~text
第二周 batch → Embedding
             → 多层 Decoder：归一化 → Attention → 残差
                            → 归一化 → SwiGLU → 残差
             → final norm → LM head → logits
                                      ├→ labels → loss → backward → 更新
                                      └→ 最后位置选 token → 追加前缀 → 再前向
~~~

### 1.1 环境与完成标准

命令均在 implementation 目录执行，PowerShell，先 uv sync --locked。默认 CPU 上的小数值实验即可；CUDA 是同一代码在相同设备的张量上运行，不另写一套算法。观察脚本使用 import torch，需要模块时增加 from torch import nn。

本周完成不以“Attention 文件写完”为准，而是核心数值和梯度可核对、完整数据能进入一次更新、生成能反馈输出并停止、tiny 序列能过拟合。它不证明通用语言能力，也不启动长时间预训练。

当前仓库保留最终实现与测试。练习时在副本逐步增加能力：先不带 padding/RoPE，随后再接入。不要在早期步骤运行所有后期测试；本文给出相应的测试选择和代码替换位置。


**分步测试约定**：下面的 pytest 命令用于已有完整仓库的复练核验。从空文件逐步实现时，先执行本步给出的手算或最小断言；现有测试文件可能在收集时就导入后续类，-k 不会跳过这些导入。相关模块完成后再运行对应测试文件，不为通过尚未实现的后期测试提前增加功能。

### 1.2 实践顺序

| 步骤 | 解决的问题 | 主要文件 |
| --- | --- | --- |
| 2–4. 聚合与因果性 | 从哪些位置读取什么内容 | inspect_attention.py、attention.py |
| 5–7. 投影与多头 | 读取特征怎样学习和并行 | attention.py、inspect_attention_heads.py |
| 8–10. GQA、RoPE、padding | 共享 KV、位置关系和有效来源 | rope.py、attention.py |
| 11–12. Block 与整机 | 基础模块怎样协同 | block.py、config.py、model.py |
| 13–14. Loss 与真实 batch | 预测什么、怎样更新 | loss.py、inspect_pretrain_batch.py |
| 15–16. 生成与 tiny 过拟合 | 输出怎样成为下一轮输入 | generation.py、inspect_tiny_overfit.py |
| 17. 验收 | 局部与整体是否一致 | 各模块测试 |

尺寸统一记为 B=batch、T=序列长度、D=hidden_size、H=Q 头数、Hkv=KV 头数、d=每头宽度、V=词表大小。基础 Attention 中的 V 张量与词表大小 V 不要混淆，正文会注明语境。

## 2. 缩放点积与内容聚合

### 2.1 匹配、权重与输出

在 scripts/inspect_attention.py 先只写下面这个实验，不加入模型参数：

~~~python
import torch

q = torch.tensor([[1.0, 0.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
v = torch.tensor([[10.0, 0.0], [0.0, 20.0], [30.0, 30.0]])

scores = q @ k.T / q.shape[-1] ** 0.5
weights = torch.softmax(scores, dim=-1)
output = weights @ v
print(scores, weights, output, sep="\n")
torch.testing.assert_close(weights.sum(-1), torch.ones(1))
~~~

~~~powershell
uv run --locked python scripts/inspect_attention.py
~~~

scores 约为 [0.7071,0,0.7071]，weights 约为 [0.4011,0.1978,0.4011]，output 约为 [16.0445,15.9889]。scores 是未归一化的匹配分数，weights 是来源位置的聚合比例，output 是新的内容特征，不把三者都叫“相似度”。

点积沿特征维求和；除以 sqrt(d) 就是“缩放”。在分量近似独立、单位方差等假设下，未缩放点积的方差随 d 增长，缩放控制其量级。它不是额外一种 Attention 算法，也不是保证所有模型分数方差恒定的定理。

### 2.2 多个读取位置

把 q 改为 [[1,0],[0,1],[1,1]]，保留 K/V，并重新执行 scores、weights、output 三行计算。现在 scores 与 weights 为 [3,3]，output 为 [3,2]。每个 query 对来源位置分配自己的比例，因此沿最后一维 Softmax，每行权重和为 1。

只修改第二个 value 的第二个特征，Q/K 不变，weights 也不变，但所有读取它的输出第二维会受影响。这验证“匹配决定读多少，V 提供读什么”；若改的是后续模型原始输入 X，Q/K/V 都可能变，不能照搬这个结论。

## 3. 因果可见性

### 3.1 掩码位置

在同一脚本中保留三行 Q，增加：

~~~python
allowed = torch.ones(3, 3, dtype=torch.bool, device=q.device).tril()
masked_scores = scores.masked_fill(~allowed, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)
output = weights @ v
torch.testing.assert_close(weights.sum(-1), torch.ones(3))
assert (weights[~allowed] == 0).all()
torch.testing.assert_close(output[0], v[0])
print(weights, output, sep="\n")
~~~

第一行只读自己，第二行读前两个位置，第三行读全部。预期 output 约为 [[10,0],[3.3024,13.3952],[17.5872,20.0698]]。

禁止位置设为负无穷，不是零。零分数的指数仍为 1，会分走权重；负无穷的指数才为零。必须在 Softmax 前屏蔽，再对剩余候选重新归一化。

### 3.2 未来扰动

复制 K/V，只改第三行，用修改后的 K 重新打分、修改后的 V 重新汇总。前两行输出与 weights 应不变，第三行在这个例子中改变。原始 scores 的第三列可以变化，只是在前两行被 mask 覆盖。

检查时不能写成 changed_weights @ 原来的 v，否则实际只验证了 K 的变化。因果性是数据依赖约束，既要有断言，也要确保被扰动的变量进入了计算。

## 4. 批量 Attention 函数

### 4.1 输入输出与实现

把计算提取到 src/minimind_lab/models/attention.py。先实现下面的基础版本；第 10 节再整体替换为带 padding 检查的版本。

~~~python
import torch

def scaled_dot_product_attention(q, k, v, causal=True):
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    if causal:
        allowed = torch.ones(
            (q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device
        ).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights
~~~

本周输入是同长度非空 self-attention：Q/K 为 [B,T,d]，V 为 [B,T,dv]。scores 为 [B,T,T]，output 为 [B,T,dv]。这里数学上 dv 可以不同；实际 MiniMind 每头通常用相同宽度 d，再通过输出投影返回 D。

这段末两维运算也自然支持 [B,H,T,d]，前面的 B/H 是独立计算维度。不要把这里的矩阵乘法叫叉积。

### 4.2 验证理由

~~~powershell
uv run --locked pytest tests/test_attention.py -q -k "causal_attention or attention_batch or attention_forward"
~~~

已知值验证公式；batch 与逐样本对照验证样本互不读取；未来扰动验证可见性；PyTorch SDPA 对照验证前向与 Q/K/V 梯度。参考调用设置 dropout_p=0.0，并与实际路径保持相同 causal 配置。

梯度测试要使用独立输入副本，分别 backward，比较 q.grad、k.grad、v.grad。只比较 backward 后输入数值，通常只是证明 backward 不更新输入，不能验证求导正确。

## 5. 可学习的单头投影

### 5.1 模块作用

现在 Q/K/V 不再手工指定。对每个位置的输入 X，用三组独立 Linear 学习匹配与内容表示，再调用已验证的聚合函数。

~~~text
X [B,T,D] → q_proj / k_proj / v_proj → [B,T,d]
         → Attention → [B,T,d] → o_proj → [B,T,D]
~~~

线性层对所有位置共享参数，不会在投影时混合位置；跨位置交流发生在 weights @ V。输出投影恢复主干宽度，方便残差相加。

### 5.2 核心实现

在 attention.py 顶部增加 from torch import nn，再追加：

~~~python
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
~~~

### 5.3 运行与检查

~~~powershell
uv run --locked pytest tests/test_attention.py -q -k single_head
~~~

形状与有限梯度只保证计算能运行。还要用 X @ weight.T 显式生成 Q/K/V，与参考 Attention 及输出投影比较，再只改未来位置检查过去输出不变。

nn.Linear(D,d) 的 weight 存储为 [d,D]，所以行向量计算使用转置。这个转置不是旋转数据，而是交换索引轴。

## 6. 拆头、换轴与合并

### 6.1 从特征分组到独立位置矩阵

在 scripts/inspect_attention_heads.py 建立可追踪的连续数字：

~~~python
import torch

B, T, H, d = 1, 3, 2, 2
q_flat = torch.arange(B*T*H*d).reshape(B, T, H*d)
q_split = q_flat.reshape(B, T, H, d)
q_heads = q_split.transpose(1, 2)
print(q_heads[0, 0])
print(q_heads[0, 1])
restored = q_heads.transpose(1, 2).reshape(B, T, H*d)
torch.testing.assert_close(restored, q_flat)
assert not torch.equal(q_heads.reshape(B, T, H*d), q_flat)
~~~

头 0 的三行是 [[0,1],[4,5],[8,9]]，头 1 是 [[2,3],[6,7],[10,11]]。拆头沿每个位置的特征分组；换轴使每个头自己的 [T,d] 落在末两维，后续矩阵乘法便在每个头内沿位置计算。

直接 reshape 回去会得到错误排列，即使最终 shape 正确。正确顺序是 [B,H,T,d] → transpose → [B,T,H,d] → reshape → [B,T,D]。也可把各头的 [B,T,d] 沿最后一维 cat。

### 6.2 逐头对照

把 q_heads 转为浮点，取 k=q.clone()、v=q+10，同时计算所有头与逐头调用。相同头的输出和权重应一致。多头并行等于逐头计算，不意味着多头模型等于一个单头模型；不同头仍有不同投影和权重。

~~~powershell
uv run --locked python scripts/inspect_attention_heads.py
~~~

现有脚本还包含逐头与合并断言，可以在写出以上排列后继续核对，避免仅看打印图猜测。

## 7. 多头自注意力

### 7.1 装配实现

在 attention.py 追加以下类。配置检查先拒绝零头数，再做取余，防止错误以 ZeroDivisionError 代替约定的 ValueError：

~~~python
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
~~~

### 7.2 参考路径

~~~powershell
uv run --locked pytest tests/test_attention.py -q -k multihead
~~~

参考路径不能在每个头循环里都传完整 [B,T,D]。第 h 个头必须切最后一维 [h*d:(h+1)*d]，输出各为 [B,T,d]，拼接后才是 D。这个测试同时保护每头特征选择和 sqrt(d) 的缩放。

causal=False 必须透传到底层，否则默认因果路径可以全绿，公开接口却失效。测试目标使用 output.square().sum()；只沿最后一维求和仍是 [B,T]，不能无参 backward。

## 8. GQA：共享 KV 头

### 8.1 共享的范围

同一个头里，各位置 Q 本来就读取同一份 K/V 序列矩阵；GQA 增加的是**不同 Q 头之间**共享 KV 头。四个 Q 头、两个 KV 头时：

~~~text
Q0、Q1 → KV0 的全部位置
Q2、Q3 → KV1 的全部位置
~~~

每个 KV 头都含全部 token 位置，KV0 不是“第零个 token”。Q 仍不同，因此同组 Q 头也能产生不同读取比例。

### 8.2 分组扩展

在 attention.py 追加 repeat_kv：

~~~python
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
~~~

用下面的值在 scripts/inspect_kv_groups.py 观察：

~~~python
import torch
from minimind_lab.models.attention import repeat_kv

kv = torch.arange(8).reshape(1, 2, 2, 2)
expanded = repeat_kv(kv, 2)
assert expanded.shape == (1, 2, 4, 2)
for h in range(4):
    torch.testing.assert_close(expanded[:, :, h], kv[:, :, h // 2])
print(expanded)
~~~

原来位置 0 的两头是 [0,1] 和 [2,3]；这些数字只是特征值。扩展后按 KV0、KV0、KV1、KV1 排列，不能用 repeat 导致交错顺序。

~~~powershell
uv run --locked pytest tests/test_attention.py -q -k repeat_kv
~~~

重复路径仍连着原张量，求和目标下梯度按重复次数累加。这里的展开会占用临时存储；减少 K/V 投影与未展开 cache，不等于任何实现都没有扩展成本。

### 8.3 模块尺寸

D=8、H=4、Hkv=2 时，每头 d=2，重复次数为 H/Hkv=2：

| 步骤 | Q | K/V |
| --- | --- | --- |
| 投影 | [B,T,8] | [B,T,4] |
| 拆头 | [B,T,4,2] | [B,T,2,2] |
| 扩展 KV | 不变 | [B,T,4,2] |
| 换轴 | [B,4,T,2] | [B,4,T,2] |

先按这个形状顺序写投影与聚合。下一节增加旋转，第 10 节给出最终 GroupQueryAttention 完整代码。当前仓库使用的准确名称是 GroupQueryAttention、参数名 kv_heads，不再使用早期讨论中的 GroupedQueryAttention/num_kv_heads。

## 9. RoPE：位置进入匹配

### 9.1 二维旋转

先在 models/rope.py 导入 torch，实现 rotate_2d：

~~~python
import torch

def rotate_2d(
    x: torch.Tensor,
    angle: torch.Tensor,
):
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    new_x0 = x[0]*cos_angle - x[1]*sin_angle
    new_x1 = x[0]*sin_angle + x[1]*cos_angle

    return torch.stack([new_x0, new_x1], dim=0)
~~~

在 scripts/inspect_rope.py 使用 float64 的 [3,4]，角度 0 得 [3,4]，pi/2 得 [-4,3]。两次结果都应与原输入有相同平方和 25。零角度验证恒等，范数验证长度；两者不能互相替代。

~~~python
import torch
from minimind_lab.models.rope import rotate_2d

x = torch.tensor([3.0, 4.0], dtype=torch.float64)
for angle, expected in [
    (0.0, [3.0, 4.0]),
    (torch.pi / 2, [-4.0, 3.0]),
]:
    out = rotate_2d(x, torch.tensor(angle, dtype=x.dtype))
    torch.testing.assert_close(out, torch.tensor(expected, dtype=x.dtype), atol=1e-10, rtol=0)
    torch.testing.assert_close(out.square().sum(), x.square().sum())
~~~

### 9.2 特征配对与直接公式

旋转的是**同一位置、同一头里的两维特征**，不是两个 token。采用前后半配对：[a,b,c,d] 对应 (a,c)、(b,d)。沿最后一维切成 first、second，直接使用二维公式，再按原布局 cat：

~~~python
def rotate_features(
    x: torch.Tensor,
    angles: torch.Tensor,
)-> torch.Tensor:
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]

    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    new_a = a*cos_angles - b*sin_angles
    new_b = a*sin_angles + b*cos_angles

    return torch.cat([new_a, new_b], dim=-1)
~~~

[1,2,3,4] 与人工角度 [0,pi/2] 应得到 [1,-4,3,2]。角度有两个数，因为有两对特征。这个人工角度用于检查布局，不是实际频率表。

另一种等价写法是 x*cos_full + rotate_half(x)*sin_full：

~~~python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")

    head_dim = x.shape[-1]

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head dimension must be a positive even integer")

    half = x.shape[-1] // 2
    a = x[...,: half]
    b = x[..., half:]

    return torch.cat([-b, a], dim=-1)

def rotate_features_with_half(
    x: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    cos_full = torch.cat([cos_angles, cos_angles], dim=-1)
    sin_full = torch.cat([sin_angles, sin_angles], dim=-1)

    return (x*cos_full + rotate_half(x)*sin_full)
~~~

rotate_half 产生每对坐标的 [-y,x] 项，是公式展开所需的辅助量；不是先额外旋转九十度，再按位置旋转一次。直接公式已经足够实现 RoPE，辅助形式用于核对与参考实现的关系。

### 9.3 频率与角度表

第 i 对特征的频率为 base^(-2i/d)，位置 m 的角度为 m×frequency_i。同一个位置有多个频率，所以得到一个角度向量，不是所有特征共用一个角度。接着在 rope.py 追加：

~~~python
def build_rope_angles(
    seq_len:int,
    head_dim: int,
    rope_base: float=1e6,
    *,
    dtype:torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")

    if rope_base <= 0:
        raise ValueError("rope_base must be positive")

    positions = torch.arange(seq_len, dtype=dtype, device=device)
    position_columns = positions[:, None] # 增加一个空维度

    frequencies_indices = torch.arange(0, head_dim, 2, dtype=dtype, device=device)
    frequencies = 1.0 / (rope_base ** (frequencies_indices / head_dim))
    frequency_row = frequencies[None, :] # 增加一个空维度

    sequence_angles = position_columns * frequency_row
    return sequence_angles
~~~

d=4、base=1e6 时频率是 [1,0.001]，位置 0、1、2 对应 [[0,0],[1,0.001],[2,0.002]]。arange 不含终点；用旧的 range 包含终点会多出一对频率。base=1e6 与 1e-6 不能混淆。

### 9.4 批量应用与相对关系

角度表 [T,d/2] 广播到 [B,H,T,d/2]，对 Q/K 旋转，V 不旋转。追加：

~~~python
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, H, T, d]")

    if q.shape[0] != k.shape[0]:
        raise ValueError("q and k must have the same batch size")

    if q.shape[-2] != k.shape[-2]:
        raise ValueError("q and k must have the same sequence length")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")

    head_dim = q.shape[-1]

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head dimension must be a positive even integer")

    expected_angles_shape = (
        q.shape[-2],
        head_dim // 2,
    )

    if angles.shape != expected_angles_shape:
        raise ValueError(
            "angles must have shape "
            f"{expected_angles_shape}, got {tuple(angles.shape)}"
        )

    q_rotated = rotate_features(q, angles)
    k_rotated = rotate_features(k, angles)

    return q_rotated, k_rotated
~~~

Q/K 可以有不同头数，但同一实验中 batch、序列长度、每头宽度必须相同。函数本身不要求先扩展 KV。

固定内容 q=[1,2]、k=[3,4]、频率 0.5，对比位置 (2,5) 与 (9,12)。位置共同平移 7，点积应一致：

~~~text
(R(mω)q)^T R(nω)k = q^T R((n-m)ω)k
~~~

点积仍取决于内容，只是旋转部分依赖相对位置差。RoPE 能计算更远位置，不代表未经验证就能无限外推上下文。

~~~powershell
uv run --locked python scripts/inspect_rope.py
uv run --locked pytest tests/test_rope.py -q
~~~

正式测试覆盖角度、两种公式的前向和梯度、不同头数及非法输入。核心已理解后无需手写重复断言来增加测试数量。

## 10. Padding 与最终 GQA 接线

### 10.1 两种 Mask 与 Loss 的分工

causal mask 限制读未来，attention_mask 排除补齐的 key，labels=-100 排除训练目标。屏蔽 key 列不保证 padded query 输出为零；它仍可能读有效来源，其目标由 labels 决定是否参与 loss。

scores 为 [B,H,Q,K] 时，causal [Q,K] 可自然广播，padding [B,K] 则需要插成 [B,1,1,K]。广播只从右对齐数字，不认识哪个维度是 batch。选 B≠T 的测试，才能避免错轴恰好不报错。

### 10.2 完整底层函数

用下面版本**替换**第 4 节的 scaled_dot_product_attention，保留已有类与 repeat_kv：

~~~python
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal:bool = True,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_t = k.transpose(-2, -1)
    scaled_scores = q @ k_t / (q.shape[-1] ** 0.5)
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

    output = weights @ v
    # output:[B, T, d_v]

    return output, weights
~~~

可见性合并后，逐 query 检查至少有一个 key 可读。否则一整行负无穷进入手写 Softmax 会产生非有限值。当前合同是非空、右侧 padding；左侧 padding 在 causal 下形成全屏蔽早期 query 时会拒绝，不宣称支持所有 mask 情况或 KV cache 对齐。

### 10.3 可手算的验证

Q/K 全零，V 的三个位置为 10、20、1000，padding mask=[1,1,0]。非因果时每行输出为 (10+20)/2=15；无 mask 时是 1030/3。因果与 padding 合并后输出为 [10,15,15]。

~~~powershell
uv run --locked python scripts/inspect_padding_mask.py
uv run --locked pytest tests/test_attention_padding.py -q
~~~

第三个 value 故意很大，让误读 PAD 清楚暴露。除了已知值，测试还对照 PyTorch 前向/梯度、去掉 PAD 的逐样本结果和 PAD 扰动；全屏蔽行的拒绝行为也需检查当前测试实际覆盖。

### 10.4 完整分组注意力

在 attention.py 顶部增加以下导入，然后写入或替换最终 GQA 类。已有 SingleHead 和 MultiHead 继续保留作对照，它们没有接入 RoPE。


~~~python
from minimind_lab.models.rope import build_rope_angles, apply_rope

class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        kv_heads,
        rope_base: float = 1e6
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

        # 重复k / v
        k_repeat = repeat_kv(k_split, n_rep=self.n_rep)
        v_repeat = repeat_kv(v_split, n_rep=self.n_rep)
        # kv_repeat:[B, T, H, head_dim]

        # 换轴
        q_trans = q_split.transpose(1, 2)
        k_trans = k_repeat.transpose(1, 2)
        v_trans = v_repeat.transpose(1, 2)
        # [B, H, T, head_dim]

        # 计算旋转角度，seq-len / head-dim
        angles = build_rope_angles(
            seq_len=T,
            head_dim=self.head_dim,
            rope_base=self.rope_base,
            dtype=q_trans.dtype,
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
~~~

这里先扩展 K 再旋转；参考路径可先旋转原始少量 K 头再选共享头，相同角度下两者等价。逐头参考对 Q 头 h 使用 KV 头 h//n_rep，不能调用待测 repeat_kv 后再声称分组得到独立验证。

~~~powershell
uv run --locked pytest tests/test_attention.py tests/test_rope.py tests/test_attention_padding.py -q
~~~

加入 RoPE 后，参考路径也必须包含同一位置变换。等头数的 GQA 仍带 RoPE，不能直接与这里无 RoPE 的 MHA 比较数值；“头数相同”只让共享结构退化，其他计算条件还必须相同。

## 11. Decoder Block：两条残差分支

### 11.1 结构与实现

Attention 汇入上下文，FFN 加工已经汇入的特征。两个分支各有独立归一化：

~~~text
h = x + Attention(RMSNorm_1(x))
y = h + SwiGLU(RMSNorm_2(h))
~~~

第三周的 ResidualFeedForward 已包含第二条归一化和残差。现在在 models/block.py 增加 GroupQueryAttention 导入与 DecoderBlock，外层不要再重复加 h：

~~~python
from minimind_lab.models.attention import GroupQueryAttention

class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        kv_heads: int,
        eps: float = 1e-5,
        rope_base : float = 1e6,
        ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps)
        self.attention = GroupQueryAttention(hidden_size, num_heads, kv_heads, rope_base)
        self.feed_forward = ResidualFeedForward(hidden_size, intermediate_size, eps)

    def forward(
        self,
        x:torch.Tensor,
        causal:bool=True,
        attention_mask: torch.Tensor | None = None,
    )-> tuple[torch.Tensor, torch.Tensor]:
        x_normed = self.attention_norm(x)
        x_atten, weights = self.attention(x_normed, causal, attention_mask)
        x_ff_in = x + x_atten
        output = self.feed_forward(x_ff_in)
        return output, weights
~~~

### 11.2 残差验证

将 Attention 的 o_proj.weight 和 FFN 的 down_proj.weight 置零，两条分支新增量都为零，输出应等于 x；输出求和后的 x.grad 应全为 1。只关一条分支时还要检查另一条正常计算，避免“所有计算都关掉”掩盖接线错误。

~~~powershell
uv run --locked python scripts/inspect_decoder_block.py
uv run --locked pytest tests/test_decoder_block.py -q
~~~

这个文件还检查因果性与 padding 接线。不要只验证所有分支的局部正确，却遗漏 mask 从 Block 向下传递。

## 12. 模型主干与词表输出

### 12.1 扩展配置

在第三周 ModelConfig 中新增 num_layers、num_heads、kv_heads、rope_base。完整当前配置如下：

~~~python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 64
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-5

    num_layers: int = 2
    num_heads: int = 4
    kv_heads: int = 2
    rope_base: float = 1e6

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
~~~

这是各字段的正值检查，整除和偶数头宽度仍由 GQA 构造检查。测试非默认 eps、base，防止配置存起来了但没有传给内部模块。

### 12.2 主干与输出头

新建 models/model.py：

~~~python
import torch

from torch import nn
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.embedding import TokenEmbedding
from minimind_lab.models.block import DecoderBlock
from minimind_lab.models.rmsnorm import RMSNorm

class MinimindBackbone(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                kv_heads=config.kv_heads,
                eps=config.rms_norm_eps,
                rope_base=config.rope_base,
                )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
        attention_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        for block in self.blocks:
            hidden_states, _ = block(hidden_states, causal, attention_mask)
        output = self.final_norm(hidden_states)
        return output


class MinimindForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.backbone = MinimindBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.backbone(input_ids, causal, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits
~~~

ModuleList 注册多个独立 Block，forward 的循环才定义执行顺序。不能用 [block]*num_layers 创建多层，那会重复引用同一实例。最后的 RMSNorm 在所有 Block 后面，不替代各分支入口的归一化。

lm_head 对每个位置的上下文向量 h 与词表中每行权重做点积，得到 V 个 logits：

~~~text
hidden [B,T,D] @ weight.T [D,V] → logits [B,T,V]
~~~

Embedding 用 ID 选行，LM head 用向量给所有行打分，因此不是“反过来查表”。输出还不是概率；此处不提前 Softmax。

### 12.3 权重共享与验证

lm_head.weight 直接引用 embedding.weight，两个模块共享同一个 Parameter。它同时收到输入查表和输出打分路径的梯度，未被输入查询的词表行也可能经输出路径更新。

~~~powershell
uv run --locked pytest tests/test_config.py tests/test_model.py -q
~~~

除了形状，测试共享对象身份、参数遍历去重，以及共享梯度等于独立双参数参考的两条梯度之和。初始化用 model.backbone.apply(initialize_weights)，当前无 bias 的输出头共享该权重，避免又给同一个表随机初始化一次。

## 13. Causal LM Loss：下一 Token 目标

### 13.1 对齐与有效目标

数据层保留 labels 与 input_ids 同位置。loss 使用前 T-1 个位置的 logits 预测后 T-1 个 labels：

~~~text
输入：   BOS  A  B  EOS  PAD
预测位： BOS  A  B  EOS
标签：    A  B EOS -100
~~~

causal mask 防止读取未来，shift 定义预测下一个 token，两者不能替代。EOS 参与监督，PAD 不参与。cross_entropy 接收原始 logits，不先 Softmax。

### 13.2 完整接口

在 models/loss.py 写入当前接口。第四周先理解均值；仓库随后为第五周梯度累积提取了 sum/count，下面直接给兼容当前测试的完整版本：

~~~python
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
~~~

先 shift 再数有效目标。整个 batch 没有目标时拒绝；单条样本无目标而其他样本有目标时可以计算。两个样本有效数不同时，目标是总损失除以总 token 数，不是样本均值再平均。

### 13.3 验证

~~~powershell
uv run --locked pytest tests/test_loss.py -q
~~~

全零 logits 对应 loss=log(V)，作为独立手算预期；忽略位置和最后一行 logits 不应影响 loss；梯度对照保护 shift 与 ignore_index；collator 集成验证标签真的能进入模型目标。

## 14. 真实 Batch 与一次更新

### 14.1 接回数据入口

第一周的 accepted.jsonl 缺失时先恢复与审计，不换数据绕过。创建小模型时 vocab_size=len(tokenizer)，避免用手造例子的 V=20 接收真实 ID。

在 scripts/inspect_pretrain_batch.py 的数据观察之后增加模型计算。下面代码依赖前面已经得到 tokenizer 与 batch，不是另一个独立文件：

~~~python
import torch
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.loss import causal_lm_loss

torch.manual_seed(42)
model = MinimindForCausalLM(
    ModelConfig(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32)
)
model.backbone.apply(initialize_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
optimizer.zero_grad(set_to_none=True)
logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
loss = causal_lm_loss(logits, batch["labels"])
loss.backward()
before = model.lm_head.weight.detach().clone()
optimizer.step()
with torch.no_grad():
    after_loss = causal_lm_loss(
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]),
        batch["labels"],
    )
print("loss before/after:", loss.item(), after_loss.item())
print("max parameter change:", (model.lm_head.weight - before).abs().max().item())
~~~

### 14.2 结果解释

冻结二十条样本的首批 shape 为 [2,512]，有效输入为 355、137，shift 后目标为 354、136，共 490。logits 为 [2,512,6400]。历史单次更新 loss 约从 8.76858 降到 8.75211，这是同一 batch 上的改善，不是验证集泛化结论。

更新前 logits 不会自动随参数刷新，观察更新后效果必须重新前向。AdamW 首次变化接近学习率，也不能反推梯度接近 1，下一周会拆开优化器公式。

当前仓库的 inspect_pretrain_batch.py 已包括下一节生成调用。先完成生成模块，再运行这个最终脚本；单独检查本节时只执行截至更新的代码。

## 15. Greedy 生成

### 15.1 从一次预测到循环

生成没有未来标签：取最后一个位置的最大 logit 对应 ID，追加到前缀，再用增长后的前缀继续预测。取最后位置可用 [:,-1:,:] 保留长度维，argmax 后为 [1,1]，便于沿序列轴拼接。

新建 models/generation.py：

~~~python
import torch

from torch import nn

def greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1  or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, T] with T > 0")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    generated_ids = input_ids.detach().clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            logits = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token],dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    return generated_ids
~~~

函数使用 no_grad，但保留调用前的 model.training，不自动切 eval。推理调用方应先 model.eval()。当前合同为单条、无 padding、非空前缀；每轮全量重算，不含 KV cache 或采样。

### 15.2 停止行为

新生成 EOS 后先追加，再停止；max_new_tokens 是新 token 上限，不是训练 padding 长度。零预算返回前缀副本且不调用模型，负预算拒绝。输入已经含 EOS，不等于当前实现自动拒绝或停止；停止条件针对本轮新生成 token。

~~~powershell
uv run --locked pytest tests/test_generation.py -q
uv run --locked python scripts/inspect_pretrain_batch.py
~~~

测试使用可控输出模型验证 EOS 和长度，不能依赖随机未训练模型碰巧生成 EOS。再用真实模型与手动 greedy 步骤对照。未训练文本难读不代表循环有 bug，但也不能以“能打印字”证明训练有效。

## 16. Tiny 过拟合与生成对照

### 16.1 最小可学习性

最后只用 [1,3,4,2] 一条人工序列，前缀 [1] 应逐步生成 3、4、2 并在 EOS=2 停止。这个实验连接模型、loss、优化器和生成，控制数据规模便于定位问题。

在 scripts/inspect_tiny_overfit.py 写入当前短循环：

~~~python
import torch

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.initialization import initialize_weights
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.generation import greedy_generate


torch.manual_seed(42)

config = ModelConfig(
    vocab_size=20,
    hidden_size=16,
    intermediate_size=32,
)
input_ids = torch.tensor([[1, 3, 4, 2]])
labels = input_ids.clone()

model = MinimindForCausalLM(config)
model.backbone.apply(initialize_weights)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-2, weight_decay=0.0)

for i in range(100):
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"loss:{loss.item()}")

final_logits = model(input_ids)
fianl_loss = causal_lm_loss(final_logits, labels)
print(f"final_loss:{fianl_loss.item()}")

model.eval()
generated_ids = greedy_generate(
    model,
    torch.tensor([[1]]),
    max_new_tokens=9,
    eos_token_id=2,
)

print(f"generated_ids:{generated_ids}")
~~~

### 16.2 运行与判断

~~~powershell
uv run --locked python scripts/inspect_tiny_overfit.py
~~~

历史在固定种子下 loss 约从 2.99900 降至 0.00388，生成 [[1,3,4,2]]。重点是能记住这条极小目标并正确停止，不把末尾小数作为跨版本合同，也不把记忆一条序列当作泛化能力。

若 Embedding 报索引类型错，先查 input_ids 是否 long；若 loss 降了却生成不对，检查 shift、最后位置取值、是否反馈增长前缀和 EOS 判断。先用这个极小问题定位，不立刻扩大数据或增加训练时长。

## 17. 验收与衔接

### 17.1 回归

~~~powershell
uv run --locked pytest tests/test_attention.py tests/test_attention_padding.py tests/test_rope.py tests/test_decoder_block.py tests/test_model.py tests/test_loss.py tests/test_generation.py -q
~~~

本周每类检查各有用途：

| 验证 | 排除的问题 |
| --- | --- |
| 数值、梯度与独立参考 | 公式或反向错误 |
| 因果与 PAD 扰动 | 不该读取的信息泄漏 |
| 非默认配置 | 参数未真正传入内部计算 |
| 两条零分支 | 残差缺失或重复相加 |
| 共享权重的双路径梯度 | 对象共享或更新范围错误 |
| 可控 EOS 与 tiny 过拟合 | 生成流程错误与整链不可学习 |

### 17.2 范围与下一周

历史第四周验收为全仓 221 项通过，tiny 过拟合与真实 batch 一次更新通过。当前测试已经扩展，数量不同不表示这周没完成。用户在指导下实现核心算法与短循环，助手协助参考验证、重复测试和整理。

尚不包含官方全部架构选项、多设备训练、KV cache、长上下文扩展和正式语料预训练。第五周把单次短循环整理为可解释的训练系统：AdamW、参数组、学习率、token 平均、累积与裁剪；第六周再做验证与恢复。

## 18. 学习问答

### 18.1 hidden_size、head_dim 和 Q/K/V 的宽度是什么关系？

hidden_size=D 是主干表示宽度。当前多头选择 d=D/H，Q 产生 H×d，GQA 的 K/V 只产生 Hkv×d。底层点积要求 Q/K 每头宽度相同，数学上 V 可有不同宽度；当前模型统一每头 d，拼接后经 o_proj 回到 D，所以完整模块输出仍为 hidden_size。

### 18.2 为什么 o_proj 输入输出一样宽还需要它？

相同宽度不等于恒等。它学习各头汇总特征的线性组合，输出的每个特征可以使用所有拼接特征，随后再进入残差主干。

### 18.3 transpose 是沿对角线翻转吗？

二维矩阵可用沿主对角线交换索引理解，K.T[j,i]=K[i,j]。不是顺时针或逆时针旋转。高维 transpose 交换指定两轴，不把整个张量当一张图片旋转。

### 18.4 reshape 会打乱元素吗，-1 保留什么？

reshape 按当前逻辑顺序重新分组，不随机打乱；-1 根据总元素数推算一个维度，不表示保护原来的某轴。transpose 后的逻辑排列已经不同，必须按正确语义换轴后再合并。

### 18.5 torch 的创建函数为什么有的用 tuple、有的不用？

不同 API 签名不同。zeros 支持多个尺寸或一个尺寸 tuple；tensor 接收实际数据；full 接收尺寸与填充值。float("-inf") 是 Python 标量，masked_fill 用它填指定元素，不是又创建一套矩阵接口。先区分“创建形状”“提供数据”“填一个标量”，无需强求语法都相同。

### 18.6 零 scores 为什么得到平均值？

Softmax 使用 exp(score)，exp(0)=1，N 个零分数各得 1/N。只有被设成负无穷的位置权重才为零；零 Q/K 是隔离 mask 行为的好输入。

### 18.7 广播为什么不能替 padding mask 自动插维？

广播只在左侧补单例轴并从右对齐已有轴，不识别 batch 的意义。[B,K] 对 [B,H,Q,K] 会把 B 对到 Q，需显式变成 [B,1,1,K]。尺寸碰巧相等可能静默计算错，比直接报错更值得测试。

### 18.8 rotate_half 为什么像旋转了九十度？

它只提供公式中的 [-y,x] 项。与 x*cos 和 sin 系数组合后才是实际角度旋转。可以完全使用直接二维公式；两种表达应在相同配对布局下对照，而不是额外做两次位置旋转。

### 18.9 一个 token 为什么有多个角度？

位置 m 只有一个，但一头里有 d/2 对坐标，每对有自己的频率。m 乘频率向量就得到多个角度。torch.cos、sin 对张量逐元素运算，一次处理整张角度表。

### 18.10 x 是一维张量，x[..., :half] 为什么可用？

省略号代表剩余的前置维度，一维时没有额外前置轴，相当于 x[:half]。在四维时同样写法保留 batch、head、位置，只切最后特征轴。

### 18.11 lm_head 是 Embedding 的逆运算吗？

不是。Embedding 已知 ID，只选一行；lm_head 接收上下文向量，对所有词表行点积打分。共享参数表也不使这两种操作互为数学逆运算。

### 18.12 赋值、copy_、clone、detach 怎么选？

共享同一个 Parameter 用对象赋值；改已有参数数值用 no_grad 下 copy_；clone 复制存储但通常保留可导关系；detach 去掉求导连接但仍可能共享存储；独立数值快照用 detach().clone()。创建 optimizer 后随意替换 Parameter 可能让优化器仍更新旧对象。

### 18.13 为什么模型接收 attention_mask，loss 又接收 labels？

前者决定当前表示读取哪些来源，后者决定哪些预测承担训练目标。它们对应前向信息流和监督目标，不是重复做同一件事。参数名要明确传入，避免把布尔 mask 误传为 input_ids。

### 18.14 程序不报错、旧测试全绿，能说明新模块正确吗？

不能。测试可能根本没有调用新类，或者参考路径重复同一个错误。先确认覆盖实际输入和分支，再检查数值与行为。例如多头循环忘记切片，或参考路径没加入 RoPE，都必须从数据流定位，而不是改 shape 让它勉强运行。
