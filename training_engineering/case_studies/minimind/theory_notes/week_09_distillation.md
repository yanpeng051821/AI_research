# 第 9 周：知识蒸馏 Distillation

## 1. 这一周学什么

知识蒸馏的目标是：

```text
让 student model 学 teacher model 的行为。
```

在 MiniMind 里，它对应的核心代码是：

```text
trainer/train_distillation.py
```

这一周不是只学一个新 loss，而是学一条新的训练思想：

```text
数据集标签告诉模型“标准答案是什么”。
teacher model 告诉模型“更强模型怎么看所有可能答案”。
student model 同时学习这两种信号。
```

## 2. 为什么需要蒸馏

大模型能力强，但通常有几个问题：

```text
参数多。
推理慢。
显存贵。
部署成本高。
不适合大量后台小任务。
```

小模型便宜、快、容易部署，但能力弱。

蒸馏想解决的问题是：

```text
能不能让小模型尽量学到大模型的一部分行为模式？
```

所以蒸馏特别适合：

- 把大模型能力压缩到小模型。
- 用强 teacher 帮 student 学得更平滑。
- 训练 Agent 系统里的小 worker 模型。
- 让小模型学习某类结构化任务，比如摘要、分类、画像抽取、上下文压缩。

## 3. 和 Pretrain、SFT、LoRA 的区别

可以这样对比：

```text
Pretrain：
从大量普通文本中学下一个 token。

SFT：
从人工或模型构造的问答数据中学 assistant 应该怎么回答。

LoRA：
冻结原模型，只训练少量增量参数。

Distillation：
让 student 不只学标准答案，还学 teacher 的概率分布。
```

一句话：

```text
SFT 学答案。
蒸馏学答案背后的分布。
```

## 4. Hard label 和 soft label

普通 CE loss 使用的是 hard label。

比如下一个 token 是“猫”，标签可以理解为：

```text
猫：1
狗：0
兔子：0
汽车：0
```

这很硬。它只告诉 student：

```text
正确 token 是猫。
```

但 teacher model 会输出一个 soft distribution：

```text
猫：0.60
狗：0.25
兔子：0.08
老虎：0.04
汽车：0.0001
```

这个分布信息更丰富。它告诉 student：

```text
猫最可能。
狗和兔子也比汽车更接近当前语境。
不同错误答案之间也有远近关系。
```

这就是蒸馏有价值的地方。

## 5. Teacher 和 student 分别是什么

teacher model：

```text
更强。
参数冻结。
只负责产生 logits/probabilities。
不参与参数更新。
```

student model：

```text
更小或更便宜。
真正被训练。
既学习数据标签，也学习 teacher 分布。
```

MiniMind 里：

```python
model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
teacher_model.eval()
teacher_model.requires_grad_(False)
```

这里 `model` 是 student。

`teacher_model` 是 teacher。

## 6. 蒸馏训练的数据流

一批 SFT 数据进来：

```text
input_ids: [B, T]
labels:    [B, T]
```

student 前向：

```python
res = model(input_ids)
student_logits = res.logits[..., :-1, :].contiguous()
```

teacher 前向：

```python
teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
```

shape 是：

```text
student_logits: [B, T-1, V]
teacher_logits: [B, T-1, V]
shift_labels:   [B, T-1]
```

为什么是 `T-1`？

因为自回归训练中：

```text
第 0 个位置的 logits 预测第 1 个 token。
第 1 个位置的 logits 预测第 2 个 token。
...
```

所以代码使用：

```python
student_logits = res.logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

## 6.5 蒸馏训练的完整 shape 流

这一节专门解决训练代码里 shape 容易绕的问题。

先约定几个符号：

```text
B: batch size
T: max_seq_len，也就是每条样本 padding/truncation 后的长度
V_s: student 的词表大小
V_t: teacher 的词表大小
C_s: student 的 hidden size
C_t: teacher 的 hidden size
N: B * (T - 1)
N_valid: assistant 有效 label 的 token 数量
```

假设：

```text
B = 4
T = 256
V_s = 6400
V_t = 6400
C_s = 512
C_t = 768
```

### 1. Dataset 输出单条样本

`SFTDataset` 每次返回一条已经模板化、分词、padding/truncation 后的数据：

```text
input_ids: [T]
labels:    [T]
```

其中：

```text
input_ids 里所有位置都是 token id。
labels 里只有 assistant 回复部分是真实 token id。
labels 里 user/system/pad 等不需要学习的位置是 -100。
```

### 2. DataLoader 拼成 batch

`DataLoader` 把多条样本堆在一起：

```text
input_ids: [B, T] = [4, 256]
labels:    [B, T] = [4, 256]
```

这就是训练循环里拿到的：

```python
for step, (input_ids, labels) in enumerate(loader):
```

### 3. 构造 loss_mask

代码：

```python
loss_mask = (labels[..., 1:] != -100).float()
```

shape 变化：

```text
labels:              [B, T]     = [4, 256]
labels[..., 1:]:     [B, T-1]   = [4, 255]
loss_mask:           [B, T-1]   = [4, 255]
```

为什么这里用 `labels[..., 1:]`？

因为自回归训练是：

```text
第 0 个位置的 logits 预测第 1 个 token。
第 1 个位置的 logits 预测第 2 个 token。
第 2 个位置的 logits 预测第 3 个 token。
```

所以 loss 只和“下一个 token 的 label”对齐。

`loss_mask` 里：

```text
1 表示这个位置是 assistant 的有效答案 token，要参与 CE 和蒸馏。
0 表示这个位置是 user/system/pad，或者不需要学习的位置。
```

### 4. student 前向传播

代码：

```python
res = model(input_ids)
student_logits = res.logits[..., :-1, :].contiguous()
```

student 内部大致 shape：

```text
input_ids:      [B, T]       = [4, 256]
embedding 后:   [B, T, C_s]  = [4, 256, 512]
blocks 后:      [B, T, C_s]  = [4, 256, 512]
lm_head 后:     [B, T, V_s]  = [4, 256, 6400]
```

切掉最后一个位置后：

```text
res.logits:      [B, T, V_s]    = [4, 256, 6400]
student_logits:  [B, T-1, V_s]  = [4, 255, 6400]
```

为什么要切掉最后一个位置？

因为最后一个位置的 logits 理论上要预测第 `T+1` 个 token，但当前 `labels` 里没有这个目标。

### 5. teacher 前向传播

代码：

```python
with torch.no_grad():
    teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
    vocab_size_student = student_logits.size(-1)
    teacher_logits = teacher_logits[..., :vocab_size_student]
```

teacher 内部大致 shape：

```text
input_ids:      [B, T]       = [4, 256]
embedding 后:   [B, T, C_t]  = [4, 256, 768]
blocks 后:      [B, T, C_t]  = [4, 256, 768]
lm_head 后:     [B, T, V_t]  = [4, 256, 6400]
```

切掉最后一个位置后：

```text
teacher_logits: [B, T-1, V_t] = [4, 255, 6400]
```

如果 teacher 词表比 student 大，代码会裁剪到 student 的词表大小：

```text
teacher_logits: [B, T-1, V_s]
```

蒸馏要求 student 和 teacher 在同一个 vocab 维度上比较分布。

### 6. labels 对齐

代码：

```python
shift_labels = labels[..., 1:].contiguous()
```

shape：

```text
labels:       [B, T]    = [4, 256]
shift_labels: [B, T-1]  = [4, 255]
```

现在三者对齐：

```text
student_logits[:, 0, :] 预测 shift_labels[:, 0]
student_logits[:, 1, :] 预测 shift_labels[:, 1]
student_logits[:, 2, :] 预测 shift_labels[:, 2]
```

对应真实位置就是：

```text
logits 第 0 位预测原 labels 第 1 位。
logits 第 1 位预测原 labels 第 2 位。
logits 第 2 位预测原 labels 第 3 位。
```

### 7. 展平成 token 级别

代码：

```python
loss_mask_flat = loss_mask.view(-1)
student_logits.view(-1, student_logits.size(-1))
shift_labels.view(-1)
```

shape：

```text
loss_mask:                  [B, T-1]       = [4, 255]
loss_mask_flat:             [N]            = [1020]

student_logits:             [B, T-1, V_s]  = [4, 255, 6400]
student_logits.view(-1,V):  [N, V_s]       = [1020, 6400]

shift_labels:               [B, T-1]       = [4, 255]
shift_labels.view(-1):      [N]            = [1020]
```

这里的 `N = B * (T - 1)`。

也就是说，训练时会把一个 batch 里的所有 token 位置摊平成一大排 token 级训练样本。

### 8. CE loss 的 shape

代码：

```python
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
```

输入 shape：

```text
student logits: [N, V_s] = [1020, 6400]
target labels:  [N]      = [1020]
```

输出 shape：

```text
ce_loss: [N] = [1020]
```

因为 `reduction='none'`，所以这里不是一个标量，而是每个 token 位置都有一个 loss。

接着：

```python
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
```

shape：

```text
ce_loss:        [N] = [1020]
loss_mask_flat: [N] = [1020]
ce_loss * mask: [N] = [1020]
sum 后:         标量
除以有效数量后: 标量
```

这一步的意义是：

```text
只统计 assistant 有效 token 的平均 CE loss。
```

### 9. 蒸馏 loss 的 shape

代码：

```python
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

先展平：

```text
student_logits.view(-1,V_s): [N, V_s] = [1020, 6400]
teacher_logits.view(-1,V_s): [N, V_s] = [1020, 6400]
loss_mask_flat == 1:         [N]      = [1020]
```

再筛选有效 token：

```text
student_valid_logits: [N_valid, V_s]
teacher_valid_logits: [N_valid, V_s]
```

比如 1020 个 token 位置里，只有 300 个是 assistant 需要学习的 token：

```text
student_valid_logits: [300, 6400]
teacher_valid_logits: [300, 6400]
```

进入 `distillation_loss`：

```python
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```

shape：

```text
teacher_logits:     [N_valid, V_s]
teacher_probs:      [N_valid, V_s]
student_logits:     [N_valid, V_s]
student_log_probs:  [N_valid, V_s]
kl 输出:            标量
distill_loss:       标量
```

这里不是比较某一个正确 token，而是比较每个有效位置上的完整词表分布。

### 10. 总 loss

代码：

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

shape：

```text
ce_loss:       标量
distill_loss:  标量
loss:          标量
```

含义：

```text
ce_loss 让 student 学数据集标准答案。
distill_loss 让 student 学 teacher 的概率分布。
alpha 控制两者比例。
```

最后：

```python
scaler.scale(loss).backward()
```

反向传播只更新 student。

teacher 因为 `eval()`、`requires_grad_(False)`、`torch.no_grad()`，不会被更新。

### 11. 一句话串起来

蒸馏训练的完整 shape 流是：

```text
input_ids [B,T]
-> student model
-> student logits [B,T,V_s]
-> student_logits[..., :-1, :] [B,T-1,V_s]
-> flatten [B*(T-1),V_s]
-> mask valid assistant tokens [N_valid,V_s]
-> CE: student vs hard labels
-> KL: student distribution vs teacher distribution
-> total scalar loss
-> backward update student
```

teacher 这条支路是：

```text
input_ids [B,T]
-> teacher model
-> teacher logits [B,T,V_t]
-> teacher_logits[..., :-1, :] [B,T-1,V_t]
-> vocab 对齐 [B,T-1,V_s]
-> flatten [B*(T-1),V_s]
-> mask valid assistant tokens [N_valid,V_s]
-> 给 KL loss 当 soft target
```

## 7. CE loss 学什么

MiniMind 里 CE loss 是：

```python
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)
```

它让 student 学数据集里的真实 token。

对于 SFT 数据：

```text
user/system/pad 的 labels 是 -100。
assistant 内容的 labels 是真实 token id。
```

所以 CE loss 主要让 student 学 assistant 应该怎么回答。

代码里还做了一层 mask：

```python
loss_mask = (labels[..., 1:] != -100).float()
ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
```

这保证只有有效 label 位置参与平均。

## 8. Distillation loss 学什么

distillation loss 不是拿 student 去对齐某个唯一 token。

它对齐的是：

```text
student 在每个位置上的 vocab 概率分布
teacher 在每个位置上的 vocab 概率分布
```

MiniMind 里：

```python
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

注意这里只在有效 label 位置上做蒸馏：

```text
loss_mask_flat == 1
```

也就是说：

```text
user/system/pad 不参与 CE。
user/system/pad 也不参与 distillation loss。
```

## 9. KL divergence 是什么

KL divergence 用来衡量两个概率分布有多不一样。

在蒸馏里，它衡量：

```text
student distribution 和 teacher distribution 差多少。
```

如果 teacher 认为：

```text
猫：0.60
狗：0.25
兔子：0.08
```

student 也认为：

```text
猫：0.58
狗：0.26
兔子：0.09
```

那么 KL 小。

如果 student 认为：

```text
汽车：0.90
香蕉：0.05
猫：0.01
```

那么 KL 大。

训练目标就是：

```text
让 student 的分布越来越像 teacher。
```

## 10. distillation_loss 代码

MiniMind 里的实现：

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl
```

分三步：

```text
1. teacher_logits / temperature -> softmax -> teacher_probs。
2. student_logits / temperature -> log_softmax -> student_log_probs。
3. 用 KL 衡量 student_log_probs 和 teacher_probs 的差异。
```

为什么 teacher 用 `torch.no_grad()`？

```text
teacher 只提供训练信号，不更新参数。
不需要保存 teacher 的梯度图。
这样更省显存，也避免误更新 teacher。
```

## 11. temperature 是什么

temperature 用来调节分布的软硬程度。

原始 logits：

```text
[10, 2, 1]
```

temperature = 1：

```text
softmax 后最高项会非常突出。
```

temperature = 2：

```text
logits 先除以 2，变成 [5, 1, 0.5]。
softmax 后分布更平滑，低概率 token 也能保留更多信息。
```

temperature 越大：

```text
分布越软。
teacher 暗含的相似性信息越容易被 student 看到。
```

但 temperature 不能无限大。太大时：

```text
分布过平，teacher 的偏好也被冲淡。
```

MiniMind 默认：

```python
--temperature 1.5
```

文档里也写了推荐范围：

```text
1.0 - 2.0
```

## 12. 为什么返回 temperature ** 2 * KL

代码最后：

```python
return (temperature ** 2) * kl
```

这是蒸馏里的常见做法。

因为 logits 除以 temperature 后，梯度尺度会发生变化。

乘以：

```text
temperature ** 2
```

是为了让不同 temperature 下的蒸馏 loss 梯度量级更可比。

先记住结论：

```text
temperature 负责软化分布。
temperature ** 2 负责补偿梯度尺度。
```

## 13. alpha 是什么

MiniMind 的总 loss：

```python
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

也就是：

```text
总 loss = alpha * CE loss + (1 - alpha) * distill loss
```

如果：

```text
alpha = 1.0
```

等价于只做普通 SFT，不看 teacher。

如果：

```text
alpha = 0.0
```

等价于只模仿 teacher，不看 ground truth label。

如果：

```text
alpha = 0.5
```

表示：

```text
一半学标准答案，一半学 teacher 分布。
```

MiniMind 默认：

```python
--alpha 0.5
```

## 14. 为什么 teacher/student vocab 要对齐

代码里：

```python
vocab_size_student = student_logits.size(-1)
teacher_logits = teacher_logits[..., :vocab_size_student]
```

含义是：

```text
让 teacher logits 的 vocab 维度和 student logits 对齐。
```

KL 要比较两个分布。

如果 student 是：

```text
[B, T, 6400]
```

teacher 是：

```text
[B, T, 7000]
```

它们不能直接计算 KL。

MiniMind 这里用截断方式让 teacher vocab 对齐 student vocab。

这是一种教学项目里的简化写法。更严谨的工程里，通常希望：

```text
teacher 和 student 使用同一个 tokenizer。
teacher 和 student 的 vocab 完全一致。
```

## 15. 为什么蒸馏用 SFTDataset

`train_distillation.py` 里：

```python
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

这说明 MiniMind 的蒸馏是在 SFT 数据格式上做的。

也就是说，输入仍然是一条完整对话：

```text
system + user + assistant
```

但是 loss 只在 assistant 的有效 label 位置上计算。

蒸馏也是如此：

```text
teacher 和 student 都看完整对话上下文。
只在 assistant 需要学习的位置上比较分布。
```

## 16. 一条完整数据流

```text
一条 SFT 对话样本
-> chat template
-> tokenizer
-> input_ids: [B, T]
-> labels: [B, T]
-> loss_mask = labels[..., 1:] != -100

student(input_ids)
-> student_logits: [B, T-1, V]

teacher(input_ids), no_grad
-> teacher_logits: [B, T-1, V]

CE:
student_logits vs shift_labels

Distill:
student distribution vs teacher distribution

总 loss:
alpha * CE + (1-alpha) * KL

backward:
只更新 student model
teacher model 不更新
```

## 17. 和小模型 Agent 的关系

你未来想做教育 Agent，小模型不一定承担最终教学回答，而是负责后台 worker 任务。

蒸馏非常适合这些任务：

```text
用户画像抽取。
学习状态摘要。
上下文压缩。
意图分类。
任务路由。
输出质检。
视觉模型结果的文本结构化整理。
```

一个典型思路：

```text
用大模型生成高质量结构化答案。
用这些答案训练小模型。
让小模型在 Agent 系统后台低成本高频运行。
```

例如：

```text
输入：最近 20 轮学习对话。
teacher 输出：用户画像 JSON + 当前学习状态 JSON。
student 学习这个输出格式和判断逻辑。
```

所以蒸馏不只是“压缩大模型”，也是：

```text
把强模型的某种工作能力迁移给小模型 worker。
```

## 18. 本周阅读顺序

建议按这个顺序读 `train_distillation.py`：

```text
1. 先读参数区：student/teacher/alpha/temperature。
2. 再读 init_model：student 和 teacher 分别怎么加载。
3. 再读 distillation_loss：softmax、log_softmax、KL。
4. 再读 train_epoch：student_logits、teacher_logits、ce_loss、distill_loss。
5. 最后读保存与断点续训。
```

不要一开始就卡在 KL 公式上。先把数据流讲清楚：

```text
同一批 input_ids，同时喂给 student 和 teacher。
student 学 labels。
student 也学 teacher logits。
最后只更新 student。
```

## 19. 本周自测问题

- teacher model 为什么要 `eval()`？
- teacher model 为什么要 `requires_grad_(False)`？
- CE loss 学的是什么？
- distillation loss 学的是什么？
- temperature 为什么能让分布变软？
- alpha 越大，训练越像什么？
- alpha 越小，训练越像什么？
- 为什么只在 `loss_mask == 1` 的位置做蒸馏？
- teacher/student 的 vocab size 为什么要对齐？
- 蒸馏为什么适合你未来做 Agent 小模型 worker？
