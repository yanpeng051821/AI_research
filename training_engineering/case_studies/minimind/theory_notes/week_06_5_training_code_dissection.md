# 第 6.5 周：Pretrain / SFT 训练代码手撕复盘

## 1. 为什么插入这一周

你已经完成了 pretrain 和 SFT 的流程理解，也已经在服务器上跑过训练。现在是最适合回头手撕训练代码的时间点。

原因是：

```text
前面看代码：容易觉得全是工程细节，抓不住重点。
跑完训练后看代码：每一行都能对应到真实训练现象。
```

后面的 LoRA、蒸馏、DPO、PPO、GRPO 都会复用这套训练骨架。如果现在只停留在“知道怎么跑”，后面会很容易变成：

```text
概念大概懂。
代码不敢改。
报错不知道从哪里查。
```

所以第 6.5 周的目标不是背源码，而是建立自己的训练脚本骨架。

最终你要能做到：

```text
给我一个 Dataset、一个模型、一个 loss，
我能自己写出一个稳定训练脚本，
能保存、能恢复、能调 batch、能看 loss、能排查显存。
```

## 2. 本周学习文件

核心文件：

```text
trainer/train_pretrain.py
trainer/train_full_sft.py
trainer/trainer_utils.py
dataset/lm_dataset.py
model/model_minimind.py
```

主线优先级：

```text
train_full_sft.py
-> train_pretrain.py
-> trainer_utils.py
-> dataset/lm_dataset.py
-> model/model_minimind.py
```

为什么先看 `train_full_sft.py`？

```text
因为 SFT 是你刚跑完的阶段，记忆最新。
而且 SFT 和 pretrain 的训练引擎几乎一样，只是 Dataset 和默认参数不同。
```

## 3. 训练脚本的主线地图

不管是 pretrain 还是 SFT，训练脚本都可以压缩成这条线：

```text
parse args
-> init distributed / seed
-> build config
-> build autocast / scaler
-> init model / tokenizer
-> build dataset
-> build optimizer
-> optionally load resume checkpoint
-> build dataloader
-> train_epoch
-> save model / checkpoint
```

更像训练时的视角：

```text
Dataset 取样本
-> DataLoader 打包 batch
-> input_ids / labels to device
-> model forward
-> loss
-> backward
-> gradient accumulation
-> grad clip
-> optimizer.step
-> zero_grad
-> log
-> save
```

这就是所有后续训练阶段的底座。

## 4. main 函数按执行顺序拆解

### 4.1 解析参数

代码：

```python
parser = argparse.ArgumentParser(...)
parser.add_argument(...)
args = parser.parse_args()
```

作用：

```text
把命令行中的 --batch_size、--max_seq_len、--from_weight 等参数变成 args.xxx。
```

你训练时敲的：

```bash
--batch_size 8
--max_seq_len 512
--from_weight pretrain_edu
```

最后都会进入：

```python
args.batch_size
args.max_seq_len
args.from_weight
```

这一层要会改，不需要背。

### 4.2 初始化分布式和随机种子

代码：

```python
local_rank = init_distributed_mode()
if dist.is_initialized():
    args.device = f"cuda:{local_rank}"
setup_seed(42 + ...)
```

普通单卡训练时：

```text
init_distributed_mode 会发现没有 RANK 环境变量。
直接返回 0。
不会真的启动 DDP。
```

所以你当前 T4 单卡训练时，可以先把它理解成：

```text
设置训练设备。
设置随机种子。
```

随机种子的意义：

```text
让 shuffle、初始化、部分随机操作尽量可复现。
```

### 4.3 构造模型配置

代码：

```python
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe)
)
```

作用：

```text
决定模型结构。
```

比如：

```text
hidden_size=512
num_hidden_layers=4
```

会决定：

```text
embedding 宽度
Transformer block 层数
attention/ffn 参数 shape
lm_head shape
```

重要规则：

```text
训练、续训、SFT、推理时的 hidden_size / num_hidden_layers 必须和权重匹配。
```

否则会出现 shape mismatch。

### 4.4 检查 resume checkpoint

代码：

```python
ckp_data = lm_checkpoint(...) if args.from_resume == 1 else None
```

这里不是加载 pretrain 权重，而是检查是否要恢复当前训练现场。

区分：

```text
from_weight：加载阶段起点，比如 pretrain -> SFT。
from_resume：恢复同一次训练现场，比如 SFT 训到一半中断后继续。
```

### 4.5 设置混合精度

代码：

```python
device_type = "cuda" if "cuda" in args.device else "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
```

作用：

```text
在 forward 和 loss 计算阶段，让部分算子用低精度计算。
```

好处：

```text
省显存。
通常更快。
```

对 T4：

```text
T4 更适合 float16。
bfloat16 支持不如新卡友好。
```

所以你的服务器命令里加：

```bash
--dtype float16
```

是合理的。

### 4.6 初始化模型和 tokenizer

代码：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
```

`init_model` 做三件事：

```text
加载 tokenizer。
根据 lm_config 创建 MiniMindForCausalLM。
如果 from_weight != none，就从 out/ 加载已有权重。
```

权重路径规则：

```text
../out/{from_weight}_{hidden_size}.pth
```

例如：

```text
--from_weight pretrain_edu
--hidden_size 512
```

加载的是：

```text
../out/pretrain_edu_512.pth
```

### 4.7 构造 Dataset

Pretrain：

```python
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

SFT：

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

这是 pretrain 和 SFT 最重要的差异之一。

PretrainDataset 返回：

```text
普通文本 token 序列。
labels 基本等于 input_ids，只屏蔽 pad。
```

SFTDataset 返回：

```text
对话模板 token 序列。
labels 只恢复 assistant 内容和结束符，其余都是 -100。
```

### 4.8 构造 optimizer 和 GradScaler

代码：

```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

optimizer 的作用：

```text
根据梯度更新模型参数。
```

GradScaler 的作用：

```text
float16 训练时放大 loss，减少梯度下溢风险。
```

注意：

```text
GradScaler 只在 dtype=float16 时启用。
bfloat16 时默认不启用。
```

### 4.9 恢复训练现场

代码：

```python
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

如果 `from_resume=1` 且 resume 文件存在，会恢复：

```text
模型参数
优化器状态
scaler 状态
epoch
step
```

这就是为什么断点续训不只是加载模型权重。

如果只加载模型，不恢复 optimizer，训练虽然能继续，但学习率、动量、AdamW 状态都不是原来的训练现场。

### 4.10 DDP / compile 包装

代码：

```python
if args.use_compile == 1:
    model = torch.compile(model)
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

当前阶段只需要理解：

```text
torch.compile 是编译优化。
DDP 是多卡训练包装。
```

这两个不是当前主线。

先做到：

```text
知道它们会包一层 model。
保存时要取 raw_model。
```

### 4.11 构造 DataLoader 并开始训练

代码：

```python
indices = torch.randperm(len(train_ds)).tolist()
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
train_epoch(epoch, loader, len(loader), 0, wandb)
```

这里做了几件事：

```text
打乱数据顺序。
按 batch_size 分组。
如果断点续训，跳过已经训过的 batch。
把 batch 送进 train_epoch。
```

`SkipBatchSampler` 是为了 resume：

```text
如果已经训到 step=1000，
恢复时就跳过前 1000 个 batch，
从第 1001 个 batch 继续。
```

## 5. train_epoch 主循环手撕

核心代码可以简化成：

```python
for step, (input_ids, labels) in enumerate(loader):
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    lr = get_lr(...)
    set_optimizer_lr(lr)

    with autocast_ctx:
        res = model(input_ids, labels=labels)
        loss = res.loss + res.aux_loss
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    if step % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    log()
    save()
```

这就是训练脚本的心脏。

### 5.1 batch to device

代码：

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
```

DataLoader 默认在 CPU 上准备数据。模型在 GPU 上，所以要把数据搬到 GPU：

```text
CPU tensor -> CUDA tensor
```

### 5.2 动态学习率

代码：

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

`get_lr` 是余弦衰减：

```python
return lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))
```

含义：

```text
训练开始接近初始学习率。
训练后期逐渐降到初始学习率的 0.1 倍。
```

### 5.3 forward 和 loss

代码：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

模型 forward 做了：

```text
input_ids [B,T]
-> embedding [B,T,C]
-> transformer blocks [B,T,C]
-> lm_head [B,T,V]
-> shift logits / labels
-> cross entropy loss
```

如果没开 MoE：

```text
aux_loss = 0
```

所以普通 dense 模型里：

```text
loss = logits_loss
```

### 5.4 为什么 loss 要除以 accumulation_steps

代码：

```python
loss = loss / args.accumulation_steps
```

如果：

```text
batch_size=8
accumulation_steps=4
```

等价于用 4 个小 batch 模拟一个大 batch：

```text
effective batch = 8 * 4 = 32
```

每个小 batch 的 loss 除以 4 后再 backward，累计起来就是 4 个小 batch 的平均梯度。

如果不除以 4：

```text
梯度会放大 4 倍。
等效学习率也会变大。
训练更容易不稳定。
```

### 5.5 backward

代码：

```python
scaler.scale(loss).backward()
```

作用：

```text
根据 loss 反向传播，计算每个可训练参数的梯度。
```

注意：

```text
backward 只是计算梯度。
不会更新参数。
```

### 5.6 optimizer.step

代码：

```python
scaler.step(optimizer)
scaler.update()
```

作用：

```text
optimizer 根据梯度和学习率更新参数。
```

注意：

```text
step 才是真正改变模型权重的地方。
```

`scaler.update()` 会根据本轮是否出现 inf/nan 动态调整缩放因子。

### 5.7 梯度裁剪

代码：

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

为什么要先 `unscale_`？

```text
因为 GradScaler 放大过 loss，也放大了梯度。
裁剪前要先把梯度还原到真实尺度。
```

`clip_grad_norm_` 的作用：

```text
如果所有参数的梯度整体范数超过 grad_clip，就按比例缩小。
```

它防的是：

```text
梯度爆炸。
单次更新过猛。
loss 突然崩掉。
```

### 5.8 zero_grad

代码：

```python
optimizer.zero_grad(set_to_none=True)
```

作用：

```text
清空已经用来更新过参数的梯度。
```

为什么不每个小 batch 都清？

```text
因为要做 gradient accumulation。
```

正确顺序是：

```text
多个小 batch backward 累积梯度
-> optimizer.step
-> zero_grad
```

### 5.9 日志

代码：

```python
Logger(f'Epoch:[...], loss: ..., logits_loss: ..., aux_loss: ..., lr: ..., epoch_time: ...')
```

要看：

```text
loss 是否正常下降。
loss 是否 NaN。
lr 是否按计划衰减。
aux_loss 是否符合当前模型类型。
epoch_time 是否大致稳定。
```

注意：

```text
这里的 epoch_time 实际是剩余时间 ETA，不是已经训练多久。
```

### 5.10 保存

代码：

```python
if step % args.save_interval == 0 or step == iters:
    torch.save(...)
    lm_checkpoint(...)
```

保存两类东西：

```text
out/*.pth：模型成果，主要用于推理和下一阶段训练。
checkpoints/*_resume.pth：训练现场快照，用于断点续训。
```

### 5.11 补充：PretrainDataset 和 SFTDataset 到底在做什么

训练脚本里 `Dataset` 的作用不是简单“读文件”。它真正决定了模型的训练目标。

一句话区分：

```text
PretrainDataset：让模型学习任意文本的下一个 token。
SFTDataset：让模型只学习 assistant 应该如何回答。
```

两者最后都会返回：

```python
input_ids, labels
```

单条样本的 shape 是：

```text
[max_length]
```

进入 `DataLoader` 后变成：

```text
[batch_size, max_length]
```

#### 5.11.1 PretrainDataset

Pretrain 数据格式很简单，一行 JSON 大概是：

```json
{"text": "大语言模型是一种基于 Transformer 的神经网络模型。"}
```

核心流程：

```python
sample = self.samples[index]
tokens = tokenizer(sample["text"], add_special_tokens=False).input_ids
tokens = [bos_token_id] + tokens[:max_length - 2] + [eos_token_id]
input_ids = tokens + [pad_token_id] * (max_length - len(tokens))
labels = input_ids.copy()
labels[pad位置] = -100
```

例如：

```text
input_ids:
[BOS, 我, 是, 学, 生, EOS, PAD, PAD]

labels:
[BOS, 我, 是, 学, 生, EOS, -100, -100]
```

这里的 `-100` 很重要。模型计算交叉熵时用了：

```python
F.cross_entropy(..., ignore_index=-100)
```

所以 `labels == -100` 的位置不会参与 loss。

PretrainDataset 的本质是：

```text
整段文本都参与 next-token prediction。
只有 PAD 不参与 loss。
```

#### 5.11.2 labels 为什么不用手动 shift

Dataset 直接返回和 `input_ids` 等长的 `labels`，但模型内部会做 shift。

在 `MiniMindForCausalLM.forward` 里：

```python
x = logits[..., :-1, :]
y = labels[..., 1:]
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

也就是说：

```text
当前位置的 logits 预测下一个位置的 label。
```

例如：

```text
input_ids:
[BOS, 我, 是, 学, 生, EOS]

真实训练目标：
看到 BOS -> 预测 我
看到 我  -> 预测 是
看到 是  -> 预测 学
看到 学  -> 预测 生
看到 生  -> 预测 EOS
```

所以 Dataset 不负责手动构造 `x[:-1]` 和 `y[1:]`，这个对齐发生在模型 forward 里。

#### 5.11.3 SFTDataset

SFT 数据是对话格式，大概长这样：

```json
{
  "conversations": [
    {"role": "user", "content": "请解释什么是大语言模型"},
    {"role": "assistant", "content": "大语言模型是一种基于大量文本训练的模型。"}
  ]
}
```

SFTDataset 的主流程：

```python
sample = self.samples[index]
conversations = pre_processing_chat(sample["conversations"])
prompt = self.create_chat_prompt(conversations)
prompt = post_processing_chat(prompt)
input_ids = tokenizer(prompt).input_ids[:max_length]
input_ids += [pad_token_id] * (max_length - len(input_ids))
labels = self.generate_labels(input_ids)
```

其中 `create_chat_prompt` 会调用：

```python
tokenizer.apply_chat_template(...)
```

把多轮对话转成模型训练用的模板文本。

MiniMind 的特殊 token 是：

```text
bos_token: <|im_start|>
eos_token: <|im_end|>
pad_token: <|endoftext|>
```

所以一段 SFT prompt 大概会变成：

```text
<|im_start|>user
请解释什么是大语言模型<|im_end|>
<|im_start|>assistant
大语言模型是一种基于大量文本训练的模型。<|im_end|>
```

#### 5.11.4 SFT 的 labels 是核心

SFT 和 Pretrain 最大区别就在 `labels`。

Pretrain：

```text
所有非 PAD token 都参与 loss。
```

SFT：

```text
只有 assistant 回答内容参与 loss。
user / system / prompt 模板不参与 loss。
```

SFTDataset 先把所有 label 都设为 `-100`：

```python
labels = [-100] * len(input_ids)
```

然后扫描 token 序列，寻找 assistant 起点：

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
```

也就是寻找：

```text
<|im_start|>assistant\n
```

找到以后，从 assistant 内容开始，到 `<|im_end|>\n` 结束，把这一段 labels 打开：

```python
labels[j] = input_ids[j]
```

简化理解：

```text
user 部分：不算 loss
assistant 标记头：不算 loss
assistant 回答正文：算 loss
assistant 的 <|im_end|>：算 loss
pad：不算 loss
```

为什么 assistant 的结束符也参与 loss？

```text
因为模型不仅要学会怎么回答，也要学会什么时候停止回答。
```

#### 5.11.5 SFT 的真实训练目标

假设 prompt 是：

```text
<|im_start|>user
什么是 LLM？<|im_end|>
<|im_start|>assistant
LLM 是大语言模型。<|im_end|>
```

那么 SFT 不会训练模型去复读 user 问题。

它真正训练的是：

```text
看到 user 问题和 assistant 起始标记 -> 预测 LLM
看到 LLM -> 预测 是
看到 是 -> 预测 大语言模型
看到 大语言模型 -> 预测 。
看到 。 -> 预测 <|im_end|>
```

也就是说：

```text
整段对话都是上下文。
只有 assistant 输出是监督信号。
```

这是 SFTDataset 的灵魂。

#### 5.11.6 PretrainDataset 和 SFTDataset 对比

| 项目 | PretrainDataset | SFTDataset |
|---|---|---|
| 输入数据 | `{"text": "..."}` | `{"conversations": [...]}` |
| 训练目标 | 学语言续写 | 学助手回答 |
| prompt 格式 | 普通文本 + BOS/EOS | chat template |
| labels 构造 | 复制 `input_ids`，PAD 改为 `-100` | 默认全 `-100`，只打开 assistant span |
| user/system 是否参与 loss | 没有这个概念 | 不参与 |
| assistant 是否参与 loss | 没有这个概念 | 参与 |
| EOS 是否参与 loss | 参与 | assistant 的 EOS 参与 |
| PAD 是否参与 loss | 不参与 | 不参与 |

你要能手撕到这个程度：

```python
# Pretrain
ids = tokenizer(text, add_special_tokens=False).input_ids
ids = [bos] + ids[:max_length - 2] + [eos]
ids = ids + [pad] * (max_length - len(ids))
labels = [-100 if x == pad else x for x in ids]
```

```python
# SFT
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
ids = tokenizer(prompt).input_ids[:max_length]
ids = ids + [pad] * (max_length - len(ids))
labels = [-100] * len(ids)

for each assistant span:
    labels[assistant_content_start:assistant_eos_end] = ids[assistant_content_start:assistant_eos_end]
```

## 6. out 权重和 resume checkpoint 的区别

### 6.1 out/*.pth

保存路径：

```text
../out/{save_weight}_{hidden_size}.pth
```

保存内容：

```text
模型参数 state_dict。
```

用途：

```text
推理。
作为下一阶段 from_weight。
比如 pretrain -> SFT。
```

### 6.2 checkpoints/*_resume.pth

保存路径：

```text
../checkpoints/{save_weight}_{hidden_size}_resume.pth
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
恢复同一次训练现场。
```

举例：

```text
SFT 训练到 step=2000 中断。
用 from_resume=1 恢复。
```

不是用于普通推理，也不是推荐作为下一阶段起点。

## 7. Pretrain 和 SFT 的最小代码差异

训练引擎基本一样。

真正差异集中在：

| 项目 | Pretrain | SFT |
|---|---|---|
| Dataset import | `PretrainDataset` | `SFTDataset` |
| Dataset 构造 | `PretrainDataset(...)` | `SFTDataset(...)` |
| 默认 data_path | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 默认 from_weight | `none` | `pretrain` |
| 默认 learning_rate | `5e-4` | `1e-5` |
| 默认 batch_size | `32` | `16` |
| 默认 accumulation_steps | `8` | `1` |
| 默认 max_seq_len | `340` | `768` |
| label 逻辑 | pad 位置为 -100 | 只有 assistant 内容参与 loss |

你可以这样理解：

```text
训练代码没有变聪明。
是数据和 label mask 改变了训练目标。
```

## 8. 哪些工程细节现在必须掌握

必须掌握：

```text
argparse 参数如何影响训练。
Dataset 和 DataLoader 如何提供 batch。
forward / loss / backward / step / zero_grad 的顺序。
gradient accumulation 为什么要除以 accumulation_steps。
autocast 和 GradScaler 的基本作用。
clip_grad_norm_ 为什么在 step 前。
from_weight 和 from_resume 的区别。
out 权重和 resume checkpoint 的区别。
```

暂时知道即可：

```text
DDP 多卡训练。
torch.compile。
wandb / swanlab。
MoE aux_loss。
world_size 改变时 step 如何折算。
```

这些后面遇到再细拆。

## 9. 手撕训练代码的推荐方式

不要从第一行开始机械读。

按这条顺序读：

```text
1. 先读 main 里第 5 步：模型、数据、优化器怎么定义。
2. 再读 train_epoch：一个 batch 怎么训练。
3. 再回到 main：resume、dataloader、epoch 循环。
4. 再读 trainer_utils：init_model、lm_checkpoint、get_lr、SkipBatchSampler。
5. 最后对比 pretrain 和 SFT 的 Dataset。
```

这样读会更像“训练真实发生了什么”，而不是“代码文件长什么样”。

## 10. 极简训练脚本骨架

你可以把 MiniMind 训练脚本抽象成：

```python
model, tokenizer = init_model(config, from_weight)
dataset = SomeDataset(data_path, tokenizer)
loader = DataLoader(dataset, batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

MiniMind 完整代码是在这个骨架上加了：

```text
混合精度。
梯度累积。
梯度裁剪。
学习率调度。
日志。
保存。
断点续训。
DDP。
compile。
wandb。
```

所以你的理解顺序也应该是：

```text
先骨架，后插件。
```

## 11. 单个 Transformer Block 的完整数据流

这一节用一个只有 1 个 Transformer block 的 MiniMind，完整走一遍数据从 `input_ids` 到 `loss` 的 shape 变化。

假设配置如下：

```text
B = 4
T = 512
C = 512
V = 6400
num_hidden_layers = 1
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 64
intermediate_size = 1664
```

其中：

```text
B: batch size
T: sequence length
C: hidden size
V: vocab size
```

输入是：

```text
input_ids: [4,512]
labels:    [4,512]
```

### 11.1 进入 MiniMindForCausalLM

训练脚本中调用：

```python
res = model(input_ids, labels=labels)
```

这里的 `model` 是：

```text
MiniMindForCausalLM
```

它内部先调用：

```python
hidden_states, past_key_values, aux_loss = self.model(...)
```

也就是进入：

```text
MiniMindModel
```

### 11.2 Embedding

输入：

```text
input_ids [4,512]
```

经过：

```python
self.embed_tokens(input_ids)
```

shape 变成：

```text
[4,512] -> [4,512,512]
```

含义是：

```text
每个 token id 被查表映射成一个 512 维向量。
```

然后经过 dropout：

```text
hidden_states [4,512,512]
-> dropout
-> hidden_states [4,512,512]
```

dropout 不改变 shape。

### 11.3 准备 RoPE 位置编码

MiniMind 会提前注册 RoPE 的 cos/sin buffer。

当前序列长度是 512，所以取：

```text
freqs_cos[start_pos:start_pos+T] -> [512,64]
freqs_sin[start_pos:start_pos+T] -> [512,64]
```

为什么最后一维是 64？

```text
因为 RoPE 作用在每个 attention head 内部。
head_dim = C / num_attention_heads = 512 / 8 = 64
```

此时：

```text
position_embeddings = (cos [512,64], sin [512,64])
```

### 11.4 进入唯一一个 MiniMindBlock

block 输入：

```text
hidden_states [4,512,512]
```

先保存残差：

```text
residual = hidden_states
residual [4,512,512]
```

再做第一层 RMSNorm：

```text
input_layernorm(hidden_states)
[4,512,512] -> [4,512,512]
```

RMSNorm 只沿最后一维归一化，不改变 shape。

### 11.5 Attention: q/k/v 投影

Attention 输入：

```text
x [4,512,512]
```

q/k/v 投影：

```text
q_proj: [4,512,512] -> [4,512,8*64] = [4,512,512]
k_proj: [4,512,512] -> [4,512,4*64] = [4,512,256]
v_proj: [4,512,512] -> [4,512,4*64] = [4,512,256]
```

为什么 k/v 是 4 个头？

```text
因为 MiniMind 默认使用 GQA。
q 有 8 个头，k/v 只有 4 个头。
后面会通过 repeat_kv 让两个 q head 共用一组 k/v。
```

### 11.6 Attention: reshape 成多头

投影后 reshape：

```text
xq [4,512,512] -> [4,512,8,64]
xk [4,512,256] -> [4,512,4,64]
xv [4,512,256] -> [4,512,4,64]
```

这里每个维度的含义是：

```text
[batch, seq_len, num_heads, head_dim]
```

### 11.7 Attention: q/k head 内 RMSNorm

代码中只对 q/k 做 head 内 RMSNorm：

```text
xq [4,512,8,64] -> [4,512,8,64]
xk [4,512,4,64] -> [4,512,4,64]
```

v 不做这一步。

### 11.8 Attention: 对 q/k 加 RoPE

RoPE 作用在 q/k 上：

```text
xq [4,512,8,64] -> [4,512,8,64]
xk [4,512,4,64] -> [4,512,4,64]
```

v 不加 RoPE。

原因是：

```text
q/k 用来计算注意力分数，需要位置信息。
v 是被加权汇总的内容向量，不直接参与位置匹配分数计算。
```

### 11.9 Attention: KV cache

训练时通常：

```text
past_key_value = None
use_cache = False
```

所以不会拼接历史 k/v。

推理时如果有 cache，会发生：

```text
xk = concat(past_k, current_k) along seq_len
xv = concat(past_v, current_v) along seq_len
```

shape 可能从：

```text
current xk [1,1,4,64]
past_k     [1,100,4,64]
```

变成：

```text
xk [1,101,4,64]
xv [1,101,4,64]
```

训练 full sequence 时先不考虑 cache。

### 11.10 Attention: repeat_kv

因为 q 有 8 个头，k/v 只有 4 个头，需要复制 k/v：

```text
xk [4,512,4,64] -> repeat_kv -> [4,512,8,64]
xv [4,512,4,64] -> repeat_kv -> [4,512,8,64]
```

含义：

```text
q 的每 2 个 head 共用 1 组 k/v。
```

### 11.11 Attention: 转成计算格式

代码中会 transpose：

```text
xq [4,512,8,64] -> [4,8,512,64]
xk [4,512,8,64] -> [4,8,512,64]
xv [4,512,8,64] -> [4,8,512,64]
```

此时维度含义是：

```text
[batch, heads, seq_len, head_dim]
```

这个格式更适合做 attention 矩阵乘法。

### 11.12 Attention: 计算 scores

如果不用 flash attention，显式计算是：

```python
scores = (xq @ xk.transpose(-2, -1)) / sqrt(head_dim)
```

shape：

```text
xq              [4,8,512,64]
xk.transpose   [4,8,64,512]
scores          [4,8,512,512]
```

含义：

```text
4 个样本。
8 个 attention head。
每个 token 对 512 个 token 的注意力分数。
```

### 11.13 Attention: causal mask

因为是自回归语言模型，当前 token 不能看未来 token。

所以 `scores` 的上三角会被 mask：

```text
scores [4,8,512,512]
```

可见区域：

```text
对角线以及对角线下面。
```

不可见区域：

```text
对角线上方。
```

### 11.14 Attention: softmax 和加权 v

softmax 后：

```text
attention_weights [4,8,512,512]
```

再乘以 v：

```text
attention_weights [4,8,512,512]
xv                [4,8,512,64]
output            [4,8,512,64]
```

含义：

```text
每个 token 根据注意力分数，对可见 token 的 v 做加权汇总。
```

### 11.15 Attention: 合并多头并输出投影

先转回：

```text
output [4,8,512,64]
-> transpose
-> [4,512,8,64]
-> reshape
-> [4,512,512]
```

再经过输出投影：

```text
o_proj [4,512,512] -> [4,512,512]
```

Attention 返回：

```text
attention_output [4,512,512]
```

### 11.16 Block: attention 残差连接

回到 block：

```text
hidden_states = attention_output + residual
```

shape：

```text
[4,512,512] + [4,512,512]
-> [4,512,512]
```

残差连接让模块输出作为原始输入的增量信息，而不是完全覆盖原始信息。

### 11.17 Block: FFN 前 RMSNorm

进入 FFN 前先做：

```text
post_attention_layernorm(hidden_states)
[4,512,512] -> [4,512,512]
```

仍然不改变 shape。

### 11.18 FFN: SwiGLU

MiniMind 的 FFN 是：

```python
down_proj(act(gate_proj(x)) * up_proj(x))
```

先升维：

```text
gate_proj: [4,512,512] -> [4,512,1664]
up_proj:   [4,512,512] -> [4,512,1664]
```

gate 分支激活：

```text
silu(gate_proj(x))
[4,512,1664] -> [4,512,1664]
```

逐元素相乘：

```text
silu(gate) * up
[4,512,1664] * [4,512,1664]
-> [4,512,1664]
```

再降回 hidden size：

```text
down_proj
[4,512,1664] -> [4,512,512]
```

FFN 输出：

```text
mlp_output [4,512,512]
```

### 11.19 Block: FFN 残差连接

block 最后：

```text
hidden_states = hidden_states + mlp_output
```

shape：

```text
[4,512,512] + [4,512,512]
-> [4,512,512]
```

唯一一个 Transformer block 结束。

重点是：

```text
block 输入是 [B,T,C]
block 输出仍然是 [B,T,C]
```

这样多个 block 才能像积木一样堆起来。

### 11.20 Final RMSNorm

回到 `MiniMindModel`，所有 block 跑完后做最终 RMSNorm：

```text
hidden_states [4,512,512]
-> norm
-> hidden_states [4,512,512]
```

如果不是 MoE：

```text
aux_loss = 0
```

`MiniMindModel` 返回：

```text
hidden_states [4,512,512]
past_key_values
aux_loss = 0
```

### 11.21 lm_head 映射到词表

回到 `MiniMindForCausalLM`，经过：

```python
logits = self.lm_head(hidden_states)
```

shape：

```text
[4,512,512] -> [4,512,6400]
```

得到：

```text
logits [4,512,6400]
```

含义：

```text
每个样本、每个位置，都输出一个 6400 维词表分数。
```

### 11.22 shift 后计算 CE loss

如果传入 `labels`，开始算 loss。

代码：

```python
x = logits[..., :-1, :].contiguous()
y = labels[..., 1:].contiguous()
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

先 shift：

```text
logits [4,512,6400]
-> x [4,511,6400]

labels [4,512]
-> y [4,511]
```

含义：

```text
第 0 个位置的 logits 预测第 1 个 token。
第 1 个位置的 logits 预测第 2 个 token。
...
第 510 个位置的 logits 预测第 511 个 token。
```

拉平：

```text
x.view(-1, V)
[4,511,6400] -> [2044,6400]

y.view(-1)
[4,511] -> [2044]
```

交叉熵输出：

```text
loss: scalar
```

其中：

```text
ignore_index=-100
```

表示 labels 中为 `-100` 的位置不参与 loss。

Pretrain 中通常只屏蔽 pad。

SFT 中通常屏蔽 system / user / role / pad，只训练 assistant 内容和结束符。

### 11.23 最终输出

`MiniMindForCausalLM.forward` 返回：

```text
loss: scalar
aux_loss: 0
logits: [4,512,6400]
past_key_values
hidden_states: [4,512,512]
```

### 11.24 一句话总览

完整 shape 主线：

```text
input_ids [4,512]
-> embedding [4,512,512]
-> 1 个 Transformer block [4,512,512]
-> final norm [4,512,512]
-> lm_head [4,512,6400]
-> shift CE loss
-> scalar loss
```

block 内部主线：

```text
[4,512,512]
-> RMSNorm [4,512,512]
-> q/k/v:
   q [4,512,8,64]
   k [4,512,4,64]
   v [4,512,4,64]
-> RoPE on q/k
-> repeat_kv:
   k/v [4,512,8,64]
-> attention scores [4,8,512,512]
-> attention output [4,512,512]
-> residual [4,512,512]
-> RMSNorm [4,512,512]
-> FFN up/gate [4,512,1664]
-> FFN down [4,512,512]
-> residual [4,512,512]
```

最核心的理解：

```text
Transformer block 内部虽然做了 q/k/v、多头、RoPE、attention、FFN，
但 block 的输入输出 shape 始终保持 [B,T,C]。
```

## 12. 本周检查题

你需要能回答：

```text
1. `from_weight` 和 `from_resume` 分别发生在训练流程的哪个位置？
2. 为什么 SFT 和 pretrain 训练代码几乎一样，但结果行为不同？
3. 为什么 loss 要除以 accumulation_steps？
4. 为什么 clip_grad_norm_ 要在 optimizer.step 前？
5. 为什么用了 GradScaler 后，clip 前要先 unscale？
6. `out/*.pth` 和 `checkpoints/*_resume.pth` 哪个给 eval_llm 用？
7. 如果你想把 batch_size 从 8 提到 16，哪些地方需要重点观察？
8. 如果恢复训练时 hidden_size 写错，会发生什么？
```

参考答案方向：

```text
1. from_weight 在 init_model 阶段加载阶段起点；from_resume 在 ckp_data 恢复训练现场。
2. Dataset、data distribution、label mask、学习率和训练起点不同。
3. 为了让多个小 batch 累积后的梯度等价于平均大 batch 梯度。
4. 裁剪的是即将用于更新的梯度，必须在参数更新前。
5. scaler 放大过梯度，要先还原真实梯度再裁剪。
6. eval_llm 用 out/*.pth。
7. 显存、速度、loss 稳定性、是否 OOM。
8. 权重 shape mismatch，通常会加载失败。
```

## 13. 完成标准

完成第 6.5 周后，你应该能做到：

```text
能按执行顺序讲清楚 train_full_sft.py。
能指出 train_pretrain.py 和 train_full_sft.py 的最小差异。
能解释一次 batch 从 DataLoader 到 optimizer.step 的全过程。
能解释混合精度、梯度累积、梯度裁剪、学习率调度各自解决什么问题。
能区分模型成果保存和训练现场保存。
能知道哪些工程细节现在必须掌握，哪些可以后置。
```

达到这个标准后，再进入第 7 周 `eval_llm.py`，推理和采样会顺很多。
