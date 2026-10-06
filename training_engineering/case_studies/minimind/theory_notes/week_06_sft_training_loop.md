# 第 6 周：SFT 训练循环

## 1. 本周目标

这一周开始从 pretrain 进入 SFT。

你现在已经知道：

```text
文本 -> tokenizer -> input_ids / labels
-> MiniMind -> logits -> CE loss
-> backward -> optimizer.step
```

SFT 不会推翻这条链路。它仍然是 causal language modeling，仍然是预测下一个 token，仍然用交叉熵。

真正变化的是：

```text
数据从普通文本变成对话。
输入中出现 system / user / assistant 角色。
loss 只计算 assistant 应该生成的部分。
训练从 pretrain 权重继续，而不是从随机初始化开始。
```

所以本周的核心问题不是“代码怎么跑”，而是：

```text
为什么同样是 CE loss，pretrain 后像续写模型，SFT 后像聊天助手？
```

## 2. SFT 在完整训练流水线中的位置

MiniMind 主线可以这样看：

```text
Tokenizer
-> Pretrain
-> Full SFT
-> LoRA / Distillation / DPO / PPO / GRPO
-> Agent / Tool Use
-> Inference / API / Web Demo
```

Pretrain 的作用：

```text
学习语言、知识、基本模式、文本续写能力。
```

SFT 的作用：

```text
学习对话格式、指令响应、助手口吻、回答结构。
```

如果用人的学习类比：

```text
Pretrain 像大量阅读书籍、网页、百科、代码。
SFT 像有人告诉你：现在你是助手，用户这样问时，你应该这样回答。
```

## 3. Pretrain 和 SFT 的核心区别

| 项目 | Pretrain | SFT |
|---|---|---|
| 数据类型 | 普通文本 | 多轮对话 |
| 数据字段 | `text` | `conversations` |
| Dataset | `PretrainDataset` | `SFTDataset` |
| 脚本 | `train_pretrain.py` | `train_full_sft.py` |
| 默认数据 | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` |
| 训练起点 | 通常 `from_weight=none` | 通常 `from_weight=pretrain` |
| 学习率 | 较大，如 `5e-4` | 较小，如 `1e-5` |
| label mask | 只屏蔽 pad | 屏蔽 system/user/role/pad，只训练 assistant |
| 模型行为 | 会续写文本 | 会按指令回答 |

一句话记忆：

```text
Pretrain 让模型有“语言底座”。
SFT 让模型知道“什么时候该由 assistant 说话，以及该怎么说”。
```

## 4. SFT 的数据流

一条 SFT 原始样本大概是：

```json
{
  "conversations": [
    {"role": "user", "content": "解释一下机器学习是什么"},
    {"role": "assistant", "content": "机器学习是让计算机从数据中学习规律的方法..."}
  ]
}
```

进入 `SFTDataset.__getitem__` 后，主线是：

```text
sample['conversations']
-> pre_processing_chat
-> create_chat_prompt
-> tokenizer.apply_chat_template
-> prompt string
-> tokenizer(prompt).input_ids
-> padding / truncation
-> generate_labels(input_ids)
-> input_ids, labels
```

对应代码位置：

```text
dataset/lm_dataset.py
SFTDataset.__getitem__
SFTDataset.create_chat_prompt
SFTDataset.generate_labels
```

## 5. chat template 到底做了什么

SFT 数据原本是结构化列表：

```text
role = user
content = ...

role = assistant
content = ...
```

模型不能直接吃 Python dict，所以要通过 chat template 转成一段字符串。

它会变成类似：

```text
<|im_start|>user
解释一下机器学习是什么<|im_end|>
<|im_start|>assistant
机器学习是让计算机从数据中学习规律的方法...<|im_end|>
```

注意：具体 token 名称以 MiniMind tokenizer 为准。代码里用的是：

```python
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

也就是说，`generate_labels` 会在 token 序列里寻找：

```text
assistant 开始标记
assistant 结束标记
```

然后只让 assistant 内容参与 loss。

## 6. SFT 的 input_ids 是什么

`input_ids` 不是一句话，也不是一个 token，更不是单个 content。

在 SFT 中，`input_ids` 是一整条训练样本的 token id 序列。

它可能包含：

```text
system role 标记
system content
user role 标记
user content
assistant role 标记
assistant content
assistant end token
下一轮 user role 标记
下一轮 user content
下一轮 assistant role 标记
下一轮 assistant content
pad token
```

所以它的 shape 是：

```text
[T]
```

经过 DataLoader batch 后变成：

```text
input_ids: [B, T]
labels:    [B, T]
```

进入模型后：

```text
input_ids [B,T]
-> embedding [B,T,C]
-> blocks [B,T,C]
-> lm_head [B,T,V]
-> logits [B,T,V]
```

## 7. SFT 的 labels 为什么大部分是 -100

SFT 的 labels 先全部设置成：

```text
-100
```

然后只恢复 assistant 内容和 assistant 结束 token。

原因是：

```text
system 是条件，不是要模型学习生成的答案。
user 是问题，不是要模型学习模仿用户提问。
role 标记是对话结构，大多数不作为主要答案内容训练。
pad 是补齐长度，更不应该参与 loss。
assistant 内容才是我们希望模型在这个上下文下生成的目标。
assistant end token 也要学，因为模型需要知道什么时候停止回答。
```

举一个简化例子：

```text
input_ids:
[user_start, 11, 12, user_end, assistant_start, 21, 22, assistant_end, pad]

labels:
[-100, -100, -100, -100, -100, 21, 22, assistant_end, -100]
```

严格来说，模型 forward 里还会做 shift：

```text
logits[..., :-1, :] 预测 labels[..., 1:]
```

所以训练目标是：

```text
看到前面的 system/user/assistant_start 后，预测 assistant 的下一个 token。
```

## 8. 为什么 SFT 仍然使用 CE loss

因为 SFT 本质仍然是 next-token prediction。

只是 pretrain 的目标是：

```text
给定普通文本前缀，预测任意下一个 token。
```

SFT 的目标是：

```text
给定 system/user/history/assistant_start，预测 assistant 应该说的下一个 token。
```

loss 函数形式没有变：

```python
F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)
```

但被计算 loss 的 token 变了，所以模型行为会变。

这就是一个很重要的思想：

```text
训练目标 = loss 形式 + 数据分布 + label mask。
```

只看 loss 形式，会误以为 pretrain 和 SFT 一样。

真正决定模型学什么的，是哪些位置的 token 被允许产生梯度。

## 9. train_full_sft.py 主线

`train_full_sft.py` 和 `train_pretrain.py` 很像。

主流程是：

```text
1. 解析参数
2. 初始化分布式、随机种子
3. 构造 MiniMindConfig
4. 读取 resume checkpoint
5. 设置 autocast 和 GradScaler
6. init_model 加载 pretrain 权重
7. 构造 SFTDataset
8. 构造 DataLoader
9. 前向传播得到 loss
10. loss / accumulation_steps
11. backward
12. 梯度裁剪
13. optimizer.step
14. scaler.update
15. zero_grad
16. 保存 out 权重和 checkpoints 现场快照
```

注意第 6 步：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
```

`from_weight` 会加载：

```text
../out/{from_weight}_{hidden_size}.pth
```

比如：

```text
--from_weight pretrain
```

会加载：

```text
../out/pretrain_512.pth
```

如果你 pretrain 时用了：

```text
--save_weight pretrain_edu
```

那么 SFT 必须写：

```text
--from_weight pretrain_edu
```

否则会找不到权重，或者加载到不是你刚训练的那份权重。

## 10. SFT 推荐实操前检查

在服务器上先确认你到底有哪些 pretrain 权重：

```bash
cd /data/workspace/ai-projects/minimind
ls -lh out | grep pretrain
ls -lh checkpoints | grep pretrain
```

常见情况有两种：

```text
情况 A：有 out/pretrain_512.pth
SFT 使用 --from_weight pretrain

情况 B：有 out/pretrain_edu_512.pth
SFT 使用 --from_weight pretrain_edu
```

还要确认模型结构参数一致：

```text
pretrain 用 hidden_size=512, num_hidden_layers=4
SFT 也必须用 hidden_size=512, num_hidden_layers=4
```

否则参数 shape 对不上。

## 11. 推荐 SFT 训练命令

如果你的 pretrain 权重是：

```text
out/pretrain_512.pth
```

使用：

```bash
cd /data/workspace/ai-projects/minimind/trainer

python train_full_sft.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --max_seq_len 512 \
  --num_workers 4 \
  --save_interval 1000 \
  --log_interval 50 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --save_weight full_sft_edu \
  --dtype float16
```

如果你的 pretrain 权重是：

```text
out/pretrain_edu_512.pth
```

把命令中的：

```text
--from_weight pretrain
```

改为：

```text
--from_weight pretrain_edu
```

推荐先用 `batch_size=8`、`max_seq_len=512`，不是因为 T4 不能更大，而是 SFT 文本更长、mask 更复杂，第一轮要优先保证稳定跑通。

如果显存仍然很低，可以再尝试：

```text
batch_size=16
accumulation_steps=2 或 4
max_seq_len=512
```

先不要一上来把 `max_seq_len` 拉到 768。因为 attention 计算和显存大致会随着序列长度平方增长。

## 12. SFT 日志怎么看

日志类似：

```text
Epoch:[1/1](100/xxxxx), loss: 3.2, logits_loss: 3.2, aux_loss: 0.0000, lr: 0.00001000, epoch_time: xxxmin
```

观察重点：

```text
loss 是否是正常数字。
loss 是否 NaN。
loss 是否整体缓慢下降。
lr 是否从 1e-5 附近逐渐衰减。
aux_loss 是否为 0。
epoch_time 是否越来越稳定。
```

`aux_loss=0` 是正常的，因为没开 MoE。

SFT 的 loss 不一定能直接和 pretrain loss 横向比较，因为：

```text
pretrain 几乎所有非 pad token 都参与 loss。
SFT 只有 assistant token 参与 loss。
两者数据分布、mask、序列结构都不同。
```

更应该比较的是：

```text
训练是否稳定。
输出是否从续写风格变成助手回答风格。
是否能遵守用户问题。
是否能自然停止。
```

## 13. SFT 断点续训

SFT 也有两类保存：

```text
out/full_sft_edu_512.pth
checkpoints/full_sft_edu_512_resume.pth
```

如果中断后要恢复同一次 SFT 训练，用：

```bash
cd /data/workspace/ai-projects/minimind/trainer

python train_full_sft.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --max_seq_len 512 \
  --num_workers 4 \
  --save_interval 1000 \
  --log_interval 50 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --from_resume 1 \
  --save_weight full_sft_edu \
  --dtype float16
```

如果你的起点权重是 `pretrain_edu`，同样把 `--from_weight pretrain` 改成：

```text
--from_weight pretrain_edu
```

注意：

```text
from_weight 用于加载阶段起点。
from_resume 用于恢复当前 SFT 训练现场。
```

第一次跑 SFT：

```text
from_weight=pretrain 或 pretrain_edu
from_resume=0
```

中断后恢复同一次 SFT：

```text
from_weight 保持不变
from_resume=1
save_weight 保持不变
```

## 14. SFT 后推理测试

SFT 训练到一个保存点后，就可以做 smoke test。

```bash
cd /data/workspace/ai-projects/minimind

python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.9
```

进入后选：

```text
0
```

自动测试。

SFT 权重和 pretrain 权重在 `eval_llm.py` 里走不同 prompt 逻辑：

```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt
else:
    inputs = tokenizer.apply_chat_template(...)
```

所以：

```text
pretrain 权重测试的是续写。
SFT 权重测试的是对话。
```

这一步的目的不是证明模型很聪明，而是确认：

```text
权重能加载。
shape 没问题。
generate 能跑。
chat_template 能跑。
输出风格已经开始像 assistant。
```

## 15. 你这阶段最应该盯的点

第一优先级：

```text
我能不能解释 SFT 数据如何变成 input_ids 和 labels。
```

第二优先级：

```text
我能不能解释为什么只训练 assistant token。
```

第三优先级：

```text
我能不能独立判断 from_weight、save_weight、from_resume 应该怎么写。
```

第四优先级：

```text
我能不能对比 pretrain 输出和 SFT 输出的差异。
```

## 16. 一张总流程图

```text
SFT jsonl
  |
  v
conversations
  |
  v
apply_chat_template
  |
  v
prompt string with roles
  |
  v
tokenizer
  |
  v
input_ids [T]
  |
  v
generate_labels
  |
  v
labels [T]
  |
  v
DataLoader
  |
  v
input_ids [B,T], labels [B,T]
  |
  v
MiniMindForCausalLM
  |
  v
logits [B,T,V]
  |
  v
shift logits / shift labels
  |
  v
CE loss, ignore_index=-100
  |
  v
只更新 assistant token 对应的预测错误
```

## 17. 本周检查题

你需要能回答：

```text
1. SFT 为什么不是让模型学习 user 提问？
2. SFT 为什么仍然是预测下一个 token？
3. assistant end token 为什么也要参与 loss？
4. `from_weight` 和 `from_resume` 有什么区别？
5. 为什么 SFT 学习率通常比 pretrain 小？
6. 为什么 SFT 后推理要用 chat_template？
7. 如果 pretrain 是 hidden_size=512，SFT 能不能改成 hidden_size=768？
```

参考答案方向：

```text
1. user/system 是上下文条件，不是目标输出。
2. Causal LM 的训练形式没有变，只是 loss mask 变了。
3. 模型需要学会什么时候结束回答。
4. from_weight 是阶段起点，from_resume 是训练现场恢复。
5. SFT 是在已有底座上调整行为，学习率太大容易破坏底座。
6. SFT 的训练输入就是 chat_template 格式，推理也要匹配训练格式。
7. 不能直接改，权重 shape 会对不上。
```

## 18. 本周完成标准

完成本周不要求模型效果很好，但要求你能做到：

```text
能跑起 train_full_sft.py。
能说清楚每个关键参数。
能判断自己应该从哪个 pretrain 权重加载。
能解释 SFTDataset 的 labels。
能用 eval_llm.py 测试 SFT 权重。
能说出 pretrain 输出和 SFT 输出的行为差异。
```

如果这些都能做到，就可以进入第 7 周：推理与采样。

