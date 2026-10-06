# 第 5.5 步：Pretrain 实操训练指南

## 1. 为什么先做实操

第 1-5 周我们已经把 MiniMind 的理论主线走通：

```text
文本 -> tokenizer -> input_ids / labels
-> MiniMind 模型 -> logits -> loss
-> backward -> optimizer.step
-> checkpoint
```

现在不要急着进入 SFT。先通过一次受控的 pretrain 实操，把训练流程真正跑通。

本阶段目标不是训练出最强模型，而是熟悉完整训练过程：

- 启动训练命令。
- 理解关键参数。
- 观察 loss、lr、显存和训练速度。
- 确认 checkpoint 保存正常。
- 理解如何中断和续训。
- 为后续 SFT 准备可靠的 pretrain 权重。

## 2. 当前服务器环境

已确认服务器环境：

```text
项目目录: /data/workspace/ai-projects/minimind
GPU: Tesla T4 16GB
CUDA: 12.2
torch: 2.4.1
transformers: 4.57.6
datasets: 3.6.0
```

已确认数据存在：

```text
dataset/pretrain_t2t_mini.jsonl
dataset/sft_t2t_mini.jsonl
```

已完成 smoke test：

```text
hidden_size = 512
num_hidden_layers = 4
max_seq_len = 128
batch_size = 2
accumulation_steps = 4
```

结果：

```text
训练能跑。
loss 正常打印。
out/pretrain_512.pth 正常保存。
checkpoints/pretrain_512_resume.pth 正常保存。
```

## 3. 推荐实操路线

建议先走学习闭环优先路线：

```text
阶段 A：512 hidden / 4 layers 完整跑通 pretrain
阶段 B：基于 pretrain 权重跑 full_sft
阶段 C：测试对话效果
阶段 D：理解完整闭环后，再考虑 768 / 8 layers
```

原因：

```text
T4 16GB 跑默认 768/8 层会慢。
512/4 层更适合学习、复盘、反复试错。
先拿到完整流程，比一开始追求效果更重要。
```

## 4. 推荐 Pretrain 命令

在服务器执行：

```bash
cd /data/workspace/ai-projects/minimind/trainer

python train_pretrain.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 8 \
  --max_seq_len 256 \
  --num_workers 4 \
  --save_interval 1000 \
  --log_interval 50 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight none \
  --save_weight pretrain_edu
```

输出权重：

```text
../out/pretrain_edu_512.pth
../checkpoints/pretrain_edu_512.pth
../checkpoints/pretrain_edu_512_resume.pth
```

这样不会覆盖已有的：

```text
../out/pretrain_512.pth
../out/pretrain_768.pth
```

## 5. 参数重点解释

### 5.1 `--hidden_size 512`

控制 hidden state 宽度。

```text
input_ids [B,T]
-> embedding [B,T,512]
-> logits [B,T,V]
```

越大模型容量越强，但显存和计算更高。

### 5.2 `--num_hidden_layers 4`

控制 Transformer block 层数。

```text
4 层：更快，适合学习闭环。
8 层：默认主线，更慢，效果更好。
```

### 5.3 `--max_seq_len 256`

每条样本最大 token 长度。

越大：

```text
上下文更长。
显存占用更高。
attention 计算更慢。
```

T4 学习阶段建议先用：

```text
128 或 256
```

### 5.4 `--batch_size 8`

每个 step 喂给模型的样本数。

如果显存爆，可以降为：

```text
4 或 2
```

### 5.5 `--accumulation_steps 8`

梯度累积步数。

有效 batch size：

```text
batch_size * accumulation_steps = 8 * 8 = 64
```

如果 batch_size 因显存下降，可以适当增大 accumulation_steps 来保持有效 batch。

### 5.6 `--save_interval 1000`

每 1000 step 保存一次。

保存包括：

```text
out/*.pth
checkpoints/*_resume.pth
```

### 5.7 `--log_interval 50`

每 50 step 打印一次日志。

观察重点：

```text
loss
logits_loss
aux_loss
lr
epoch_time
```

### 5.8 `--from_weight none`

从随机初始化开始预训练。

Pretrain 阶段通常用：

```text
none
```

SFT 阶段才会用：

```text
pretrain_edu
```

### 5.9 `--save_weight pretrain_edu`

自定义保存名前缀，避免覆盖已有权重。

最终保存类似：

```text
pretrain_edu_512.pth
```

## 6. 训练过程中观察什么

日志示例：

```text
Epoch:[1/1](700/xxxxx), loss: 6.6, logits_loss: 6.6, aux_loss: 0.0, lr: 0.0005, epoch_time: xxxmin
```

重点看：

```text
loss 是否是正常数字。
loss 是否长期 NaN。
loss 是否出现极端爆炸。
GPU 显存是否接近满。
保存 interval 到达后是否生成 checkpoint。
```

初期 loss 在 6-8 附近是正常的。

不要期待几百 step 内明显变聪明。Pretrain 是慢过程。

## 7. 显存监控

建议另开一个终端：

```bash
watch -n 2 nvidia-smi
```

观察：

```text
显存占用。
GPU-Util。
是否有其他进程占用 GPU。
```

如果 OOM：

优先按这个顺序降参数：

```text
1. batch_size: 8 -> 4 -> 2
2. max_seq_len: 256 -> 128
3. num_hidden_layers: 4 保持不动
4. hidden_size: 512 保持不动
```

不要一开始就动所有参数，否则不容易判断是哪一项造成问题。

## 8. 保存结果检查

训练跑到第一个 `save_interval` 后，检查：

```bash
cd /data/workspace/ai-projects/minimind
ls -lh out | grep pretrain_edu
ls -lh checkpoints | grep pretrain_edu
```

期望看到：

```text
out/pretrain_edu_512.pth
checkpoints/pretrain_edu_512.pth
checkpoints/pretrain_edu_512_resume.pth
```

如果只有 checkpoints 里有，也先别慌，重点确认后续 SFT 能否从 `out/` 加载。

## 9. 如何中断和从终端恢复训练

可以用：

```text
Ctrl + C
```

中断训练前，最好已经至少到达过一次 `save_interval`，因为只有保存过 resume checkpoint，才能从比较近的位置恢复。

### 9.1 中断后先检查 checkpoint

回到项目根目录：

```bash
cd /data/workspace/ai-projects/minimind
```

检查文件：

```bash
ls -lh out | grep pretrain_edu
ls -lh checkpoints | grep pretrain_edu
```

重点确认：

```text
out/pretrain_edu_512.pth
checkpoints/pretrain_edu_512_resume.pth
```

如果没有 `checkpoints/pretrain_edu_512_resume.pth`，说明还没有可恢复的训练现场，只能从 `out/pretrain_edu_512.pth` 作为权重重新开始，不能严格续训。

### 9.2 从终端恢复训练

重新进入 trainer 目录：

```bash
cd /data/workspace/ai-projects/minimind/trainer
```

执行续训命令：

```bash
python train_pretrain.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 8 \
  --max_seq_len 256 \
  --num_workers 4 \
  --save_interval 1000 \
  --log_interval 50 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_resume 1 \
  --save_weight pretrain_edu
```

注意：恢复训练时，除了 `--from_resume 1`，其他关键参数尽量和中断前保持一致：

```text
hidden_size
num_hidden_layers
max_seq_len
batch_size
accumulation_steps
save_weight
data_path
```

尤其是：

```text
hidden_size 和 num_hidden_layers 必须一致，否则模型结构对不上。
save_weight 必须一致，否则会找不到对应 resume 文件。
```

### 9.3 如何判断真的恢复成功

恢复成功时，日志里应该看到类似：

```text
跳过前 xxx 个step，从step xxx+1开始
```

这说明：

```text
已经读取 checkpoints/pretrain_edu_512_resume.pth。
已经恢复 start_step。
SkipBatchSampler 正在跳过之前训练过的 batch。
```

如果没有看到跳过 step，可能说明：

```text
没有找到 resume checkpoint。
from_resume 没有设为 1。
save_weight 和之前不一致。
checkpoint 文件名不匹配。
```

注意：

```text
from_resume 用 checkpoints/*_resume.pth。
from_weight 用 out/*.pth。
```

### 9.4 `from_resume` 和 `from_weight` 的区别

如果你想严格接着中断位置训练：

```bash
--from_resume 1
```

它会恢复：

```text
model
optimizer
scaler
epoch
step
```

如果你只是想加载某个已有模型权重重新开始训练：

```bash
--from_weight pretrain_edu
```

它只加载：

```text
out/pretrain_edu_512.pth 中的模型参数
```

不会恢复：

```text
optimizer
scaler
epoch
step
```

所以：

```text
中断续训：用 from_resume。
换阶段训练或基于某个模型继续新任务：用 from_weight。
```

## 10. 进入 SFT 前必须确认

进入 SFT 前，至少确认：

```text
1. pretrain loss 正常，不是 NaN。
2. out/pretrain_edu_512.pth 存在。
3. checkpoints/pretrain_edu_512_resume.pth 存在。
4. pretrain 权重能被 eval_llm.py 加载并生成文本。
5. 你能解释当前训练命令里的每个参数。
6. 你知道如果中断，如何用 --from_resume 1 继续。
```

如果这些都满足，再进入 SFT。

SFT 阶段会用：

```text
train_full_sft.py
SFTDataset
sft_t2t_mini.jsonl
--from_weight pretrain_edu
```

## 11. Pretrain 后推理 smoke test

Pretrain 结束后需要做一次轻量推理测试。

注意：这里不是测试“助手能力”，只是测试：

```text
权重能加载。
tokenizer 能工作。
generate 能跑。
输出不是 NaN 或程序崩溃。
pretrain 权重可以作为后续 SFT 起点。
```

Pretrain 模型还没有学会稳定地以 assistant 身份回答问题，所以输出可能不够像聊天助手，这是正常的。

### 11.1 推荐测试命令

在服务器项目根目录执行：

```bash
cd /data/workspace/ai-projects/minimind

python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight pretrain_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 80 \
  --temperature 0.8 \
  --top_p 0.9
```

进入后选择：

```text
0
```

表示自动测试内置 prompts。

### 11.2 为什么 pretrain 测试不用 chat_template

`eval_llm.py` 里有一段逻辑：

```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt
else:
    inputs = tokenizer.apply_chat_template(...)
```

也就是说：

```text
pretrain 权重使用普通文本续写方式测试。
SFT 权重才使用 chat_template 对话方式测试。
```

原因：

```text
Pretrain 学的是文本续写。
SFT 学的是对话助手格式。
```

### 11.3 结果如何判断

正常情况：

```text
模型成功加载。
能打印 Model Params。
能生成若干 token。
不会 CUDA OOM。
不会报 state_dict shape mismatch。
不会输出 NaN。
```

不要过度关注回答质量。

如果输出很怪：

```text
可能是 pretrain 训练步数少。
可能还没经过 SFT。
可能模型规模较小。
```

这些都不影响进入 SFT，只要推理链路是通的。

### 11.4 常见错误

如果报找不到权重：

```text
检查 out/pretrain_edu_512.pth 是否存在。
检查 --weight pretrain_edu 是否和保存前缀一致。
检查 --hidden_size 512 是否和训练时一致。
```

如果报 shape mismatch：

```text
通常是 hidden_size 或 num_hidden_layers 和训练时不一致。
```

如果 OOM：

```text
降低 --max_new_tokens。
确认没有其他 GPU 进程。
```

## 12. 本阶段学习任务

训练时你需要主动观察并记录：

- 当前命令参数。
- 第一次 loss 大概是多少。
- 训练 500/1000/2000 step 后 loss 有没有明显变化。
- 显存占用是多少。
- 每个 log interval 大概耗时多久。
- checkpoint 是否正常保存。
- 中断后能否 resume。
- pretrain 权重是否能通过 `eval_llm.py` 做续写推理。

建议在训练完成后，用自己的话复述：

```text
这次 pretrain 从数据读取到参数更新，再到保存 checkpoint 的完整流程。
```
