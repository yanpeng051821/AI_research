# 04 核心实现与数值验证

## 1. 目的

在真实 GPU 上运行并不能证明训练目标正确。这个节点先用小张量、tiny model 和少量真实
数据验证每个核心合同，再把成熟框架接入。测试的目标是定位错误，不是为了增加测试数量。

## 2. 需要验证的完整数据流

```text
messages
-> apply_chat_template
-> input_ids / labels
-> Dataset
-> Sampler
-> DataLoader
-> dynamic-padding collator
-> input_ids / attention_mask / labels
-> model forward
-> logits
-> causal shift
-> completion-only token loss
-> accumulation-window global token mean
-> backward
-> gradient clipping
-> AdamW step
-> scheduler step
-> zero_grad
-> checkpoint / resume
```

## 3. Masked SFT loss

给定：

```text
logits [B, T, V]
labels [B, T]
```

计算：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
valid = shift_labels != -100
token_losses = cross_entropy(..., reduction="none", ignore_index=-100)
loss = token_losses[valid].sum() / valid.sum()
```

必须测试：

- causal shift 的位置正确；
- prompt/padding 的 `-100` 不参与 loss；
- assistant 结束 token 仍受监督；
- 全 mask、空 batch、shape/dtype 错误按合同处理；
- 与 PyTorch `cross_entropy` 数值一致；
- backward 后梯度有限且形状与参数一致；
- tiny batch 可以 overfit，并朝目标 token 方向提高概率。

## 4. Gradient accumulation 的关键合同

不同 micro-batch 的有效 completion token 数不相同。目标是整个 accumulation window 的
token 级全局平均：

```text
global_loss = sum(all valid token losses) / sum(all valid token counts)
```

不能简单平均每个 micro-batch 的 mean loss，否则短样本会获得过高样本级权重。

验证方法：使用同一初始模型，比较两条路径。

```text
路径 A：完整 batch 一次计算全局 token mean，再 backward + step
路径 B：拆成多个 micro-batch，按全局 token count 缩放后累计梯度，再 step
```

比较点：

- `optimizer.step()` 前比较每个参数的梯度；
- step 后比较每个参数值；
- 比较 scheduler 的 step/LR；
- 最后一个不足完整 accumulation window 的 tail batch 也必须更新一次。

## 5. 数据、Sampler 与 DataLoader

分别验证职责：

```text
Dataset：index -> 单条样本
Sampler：决定 index 顺序
DataLoader：按顺序读取并组织 batch
Collator：动态 padding，生成统一形状
```

对于可恢复训练，还要验证 sampler 的“已交付位置”和“已提交位置”。DataLoader 可能预取，
但只有真正完成训练消费的样本才能写入 checkpoint；否则恢复时会跳过或重复样本。

正式 TRL 路径采用预先冻结的完整 epoch order，并把 `sample_order_sha256` 写入 run identity。

## 6. 真实 tokenizer 与模型接口

小 tokenizer fixture 只能证明代码形状，不能证明 Qwen 模板边界。至少执行一次锁定 revision
的真实集成测试，检查：

```text
system/user/assistant 边界 token
assistant 开始位置
assistant EOS 是否在 labels 中
padding side 和 pad_token_id
add_generation_prompt 在训练/推理中的差异
```

模型接口必须走真实 HF 调用形状：

```python
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
)
logits = outputs.logits
```

labels 可以由我们在外部计算 loss，也可以传入模型让 HF 返回 loss，但对照实验中两条路径
必须使用相同 shift、mask 和 reduction 语义。

## 7. TRL / Transformers 对照

正式 S1 使用 TRL，不代表跳过独立实现。我们要验证：

- 相同 batch 的有效 token 数一致；
- completion-only mask 一致；
- loss 数值和梯度方向一致；
- optimizer、scheduler、warmup 和 tail update 一致；
- checkpoint 恢复后的 global step 与参数更新连续；
- 数据顺序固定且可核对。

first-batch shadow 只比较同一初始模型、同一 batch、未更新参数的两条路径。A 路径更新模型后
再运行 B 路径，会混入参数变化，失去公平性。

## 8. 测试层次

```text
Level 1：纯函数单元测试
  shift、mask、loss、token count、schedule

Level 2：tiny model 组件测试
  forward/backward、overfit、optimizer、scheduler

Level 3：进程级测试
  CLI、失败退出码、checkpoint/resume、final export

Level 4：真实 tokenizer/model 集成测试
  固定 revision，允许单独标记 integration

Level 5：GPU smoke
  真实精度、attention backend、显存和保存
```

## 9. 验收门与产物

验收：

- [ ] 单元测试和静态检查通过。
- [ ] tiny overfit 不只让 loss 下降，也提高目标 token 概率。
- [ ] full batch 与 accumulation 参数更新对齐。
- [ ] completion-only mask 与 TRL collator 对齐。
- [ ] checkpoint/resume 的参数、optimizer、scheduler 和 step 连续。
- [ ] first-batch shadow 有机器可读报告。

落盘：

```text
pytest report / command
integration-test result
first-batch shadow JSON
runtime-tree hash
dependency lock
失败快照（若有）
```

## 10. 本案例结论

本机已完成 masked loss、全局 token 归一化、真实 Qwen tokenizer、HF 模型接口、
optimizer/scheduler、checkpoint/resume 和 TRL 对齐测试。它证明核心语义可进入服务器验证，
不替代真实最长样本和正式 S1。
