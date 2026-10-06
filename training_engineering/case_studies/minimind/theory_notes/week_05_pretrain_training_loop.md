# 第 5 周：Pretrain 训练循环

## 1. 本周学习目标

第 5 周把前面学过的输入链路和模型链路接到训练脚本上：

```text
Dataset / DataLoader
-> batch(input_ids, labels)
-> model(input_ids, labels=labels)
-> loss
-> backward
-> optimizer.step
-> checkpoint
```

本周重点不是新模型结构，而是训练工程闭环：

- `DataLoader` 如何提供 batch。
- `loss.backward()` 和 `optimizer.step()` 分别做什么。
- 为什么需要 `zero_grad`。
- 为什么要做梯度累积。
- `autocast` 和 `GradScaler` 如何支持混合精度。
- `clip_grad_norm_` 为什么能让训练更稳。
- `out/*.pth` 和 `checkpoints/*_resume.pth` 分别保存什么。
- `train_pretrain.py` 和 `train_full_sft.py` 差异在哪里。

## 2. Pretrain 训练主线

源码位置：

```text
/Users/yanpeng/Documents/PythonProjects/datawhale/minimind/trainer/train_pretrain.py
```

核心流程：

```text
1. 解析参数。
2. 初始化分布式环境和随机种子。
3. 创建 MiniMindConfig。
4. 设置 autocast 混合精度上下文。
5. 初始化 model、tokenizer、PretrainDataset、optimizer、scaler。
6. 如果 from_resume=1，则恢复 checkpoint。
7. 可选 torch.compile / DDP。
8. 构造 DataLoader，进入 train_epoch。
9. 定期保存权重和 resume checkpoint。
```

## 3. train_epoch 主循环

核心代码可以概括为：

```python
for step, (input_ids, labels) in enumerate(loader):
    input_ids = input_ids.to(args.device)
    labels = labels.to(args.device)

    lr = get_lr(...)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / args.accumulation_steps

    scaler.scale(loss).backward()

    if step % args.accumulation_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

自然语言版本：

```text
取 batch。
搬到 GPU/CPU。
调整学习率。
前向传播得到 loss。
loss 除以梯度累积步数。
反向传播累积梯度。
累积够以后，恢复梯度尺度、裁剪梯度、更新参数、清空梯度。
```

## 4. Dataset 和 DataLoader

`Dataset` 负责取单个样本：

```text
input_ids: [T]
labels: [T]
```

`DataLoader` 负责组成 batch：

```text
input_ids: [B,T]
labels: [B,T]
```

MiniMind 的 `PretrainDataset` 不是先把所有文本拼成超长 token 序列再切块，而是：

```text
从 jsonl 中取一条 text
-> tokenizer 编码
-> 截断到 max_length - 2
-> 加 bos/eos
-> pad 到 max_length
-> labels = input_ids.clone()
-> pad 位置改成 -100
```

## 5. backward、step、zero_grad

三者分工：

```text
loss.backward():
计算每个可训练参数的梯度。

optimizer.step():
根据已经算好的梯度更新参数。

optimizer.zero_grad():
清空已经用完的梯度，避免影响后续 batch。
```

一句话：

```text
backward 算 grad，step 改 weight，zero_grad 清 grad。
```

## 6. 梯度累积

MiniMind 默认 pretrain：

```text
batch_size = 32
accumulation_steps = 8
```

有效 batch size：

```text
32 * 8 = 256
```

为什么要：

```python
loss = loss / args.accumulation_steps
```

因为要让多个小 batch 的梯度累积后，等价于大 batch 的平均梯度。

如果不除以 `accumulation_steps`，梯度尺度会变大，等价于学习率被放大，训练容易不稳定。

## 7. autocast 和 GradScaler

`autocast`：

```text
让适合低精度的前向计算自动使用 float16 / bfloat16，省显存并加速。
```

`GradScaler`：

```text
主要用于 float16，防止小梯度下溢成 0。
```

MiniMind 中：

```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
```

也就是说：

```text
float16 启用 scaler。
bfloat16 通常不启用 scaler，因为 bf16 动态范围更接近 fp32。
```

正确顺序：

```text
scale loss
-> backward
-> unscale gradients
-> clip gradients
-> optimizer step
-> update scaler
```

## 8. 梯度裁剪

`clip_grad_norm_` 裁剪的是所有参数梯度的整体范数。

它不是逐个元素硬截断，而是：

```text
如果整体梯度 norm 超过阈值，就按比例缩小所有梯度。
```

作用：

```text
防止某些 batch 导致梯度爆炸。
限制单次参数更新步子。
提升 LLM 训练稳定性。
```

## 9. 学习率调度

MiniMind 使用 cosine decay：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))
```

直觉：

```text
前期步子大，快速学习。
后期步子小，稳定收敛。
```

它不是自动保证最优，而是经验上常用且稳定的学习率曲线。

## 10. checkpoint 和断点续训

MiniMind 保存两类文件。

### 10.1 out/*.pth

位置示例：

```text
../out/pretrain_768.pth
```

保存内容：

```text
模型参数 state_dict
```

保存方式：

```text
参数转 half。
搬到 CPU。
torch.save 保存。
```

用途：

```text
推理。
作为 SFT / LoRA / DPO 等后续阶段的起点。
```

### 10.2 checkpoints/*_resume.pth

位置示例：

```text
../checkpoints/pretrain_768_resume.pth
```

保存内容：

```text
model
optimizer
scaler
epoch
step
world_size
wandb_id
```

用途：

```text
训练中断后真正恢复训练现场。
```

一句话：

```text
out/*.pth 是模型成果。
checkpoints/*_resume.pth 是训练现场快照。
```

## 11. Pretrain 和 SFT 脚本差异

训练循环几乎一样，差异主要在：

| 项目 | Pretrain | SFT |
|---|---|---|
| Dataset | `PretrainDataset` | `SFTDataset` |
| 默认数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 默认起点 | `from_weight=none` | `from_weight=pretrain` |
| 默认学习率 | `5e-4` | `1e-5` |
| 默认 batch | `32` | `16` |
| 默认 seq_len | `340` | `768` |
| 保存名 | `pretrain` | `full_sft` |

一句话：

```text
Pretrain 从头学语言续写。
SFT 从 pretrain 权重继续，学助手对话行为。
```

## 12. 实操前检查

服务器已初步确认：

```text
GPU: Tesla T4 16GB
CUDA: 12.2
torch: 2.4.1
transformers: 4.57.6
datasets: 3.6.0
数据:
  pretrain_t2t_mini.jsonl
  sft_t2t_mini.jsonl
```

进入下一阶段前，建议先完成：

```text
1. smoke test pretrain 跑通数个 step。
2. 确认 loss 正常打印。
3. 确认 out/ 和 checkpoints/ 能保存文件。
4. 再规划正式 pretrain 参数。
5. pretrain 完成后再进入 full_sft。
```

## 13. 当前进度

状态：已完成理论，待实操验证

已完成：

- [x] 读懂 pretrain 训练循环。
- [x] 理解 optimizer / backward / zero_grad。
- [x] 理解 gradient accumulation。
- [x] 理解 autocast / GradScaler。
- [x] 理解 clip_grad_norm_。
- [x] 理解 lr schedule。
- [x] 理解 checkpoint / resume checkpoint。
- [x] 对比 pretrain 和 SFT 脚本差异。

待完成：

- [ ] 在服务器完成 pretrain smoke test。
- [ ] 确认 checkpoint 保存正常。
- [ ] 制定 T4 16GB 下的正式训练参数。

