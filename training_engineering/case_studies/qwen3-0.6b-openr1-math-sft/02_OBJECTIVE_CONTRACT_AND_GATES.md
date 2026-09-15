# 02 目标、实验合同与门禁

## 1. 目的

训练开始前先回答“为什么训练、希望改变什么、什么结果算成功、什么结果必须停止”。
实验合同不是形式文档，它负责阻止我们在看到结果后临时修改评测口径或挑选有利分数。

本文保留正式训练前冻结的合同视角。案例当前已经完成正式 S1；实际执行结果、合同偏离和
最终结论分别见 `10_INCIDENTS_AND_REUSABLE_LESSONS.md` 与
`11_FORMAL_S1_RESULTS_AND_RELEASE.md`。

## 2. 需要冻结的五类问题

### 2.1 问题定义

本案例的问题是：

> 对 Qwen3-0.6B-Base 进行数学 SFT 后，模型能否更好地拟合经过验证的数学
> completion 数据，同时不出现不可接受的通用能力回退？

这里包含两个方向：目标能力改善和原有能力保护。只有 validation NLL 下降，不足以回答
第二个方向。

### 2.2 实验对象

至少固定：

- Base 模型名称和不可变 revision；
- tokenizer/chat template revision；
- 训练、验证和评测数据 revision；
- 训练阶段名称，例如 B0、pilot、S1；
- 正式 S1 必须从 Base 启动，不能从 pilot 延续。

### 2.3 训练合同

本案例正式配置的关键值是：

```yaml
model: Qwen/Qwen3-0.6B-Base
train_records: 61224
max_sequence_length: 16384  # 来自 artifact 过滤策略，不是运行时截断
per_device_train_batch_size: 1
gradient_accumulation_steps: 128
num_train_epochs: 1
optimizer_steps: 479
learning_rate: 4.0e-5
warmup_steps: 15
parameter_dtype: float32
autocast_dtype: bfloat16
attention: flash_attention_2
seed: 42
```

### 2.4 评测合同

每个 suite 都要固定：

```text
dataset revision
task name/version
prompt/chat-template mode
few-shot 数
scorer
sampling 参数
max_new_tokens / max_model_length
seed
每题采样数
逐样本输出格式
```

B0 与 S1 只要其中一项不同，就不能把分数差直接归因于训练。

### 2.5 决策规则

至少预先定义：

- 什么条件允许从 preflight 进入 smoke；
- 什么条件允许从 smoke 进入 pilot；
- 什么条件允许从 pilot 进入 formal S1；
- 哪些错误立即停止；
- 哪些指标只是诊断信号，不能作为能力结论；
- 哪些能力回退会否决本次模型。

## 3. 推荐的门禁结构

```text
Gate A：本地计算正确
  单元测试、真实 tokenizer、tiny overfit、resume round-trip

Gate B：数据可信
  自动审计、人工复核、hash、manifest、无隐藏 truncation

Gate C：B0 可信
  全量基线、逐样本结果、评测合同固定

Gate D：服务器能跑
  环境、最长样本、首 batch、一次更新、checkpoint

Gate E：pilot 可解释
  有终态、能恢复、NLL 趋势合理、成本可接受

Gate F：正式 S1 GO
  telemetry 和 paired-eval preflight 完成，所有输入冻结
```

任何门禁失败时，先记录失败属于哪一层。不要同时改数据、模型、attention backend 和
训练参数，否则下一次成功也无法确定是哪项修改起作用。

## 4. 必须落盘的合同文件

```text
experiment contract
resolved training config
evaluation config
model/tokenizer revision
data manifest and file hashes
source Git commit and runtime-tree hash
dependency lock hash
sample-order hash
GO/NO-GO decision and reason
```

本案例的正式 dry-run manifest 已包含 Git commit、runtime tree、合同、`uv.lock`、
模型/Tokenizer revision、训练/验证文件与样本顺序 hash。

## 5. 本案例暴露的合同问题

### 预过滤数据量被继续用于步数计算

16K 过滤前训练集是 62,208 条，过滤后是 61,224 条。旧文档曾出现 486/487 steps，
但正式合同应为：

```text
ceil(61,224 / 128) = 479 optimizer updates
ceil(479 * 0.03) = 15 warmup updates
```

经验：任何数据过滤都会改变 sample count、总 token、训练步数、scheduler 和成本估计；
artifact 变化后必须重新解析合同，不能只替换数据路径。

### pilot 与 formal run 的身份混淆

100-step pilot 可以证明训练与恢复路径，但不是正式 S1。正式 S1 必须创建新 run ID、空
输出目录，并从 Base 重新开始。

### 历史 baseline 数字不可追溯

仓库曾有 MATH-500 约 26% 至 28% 的历史记录，但缺少可核对合同。冻结合同实测为
pass@1:1 `0.432`。不可复现的历史数字被降级为线索，不能作为比较基线。

## 6. 本案例执行回填

以下项目在本案例收尾时均可由落盘证据回答；它们在下一次实验中必须重新清零和验证：

- [x] 能用一句话说明要改变的能力和保护的能力。
- [x] Base、数据、tokenizer、代码和依赖均有 revision/hash。
- [x] 训练步数由最终 artifact 重新计算。
- [x] B0 与 S1 的可比较结果使用同一评测合同。
- [x] smoke、pilot 和 formal run 使用不同 run ID/输出目录。
- [x] 失败、暂停、完成都有明确终态。
- [x] 没有把局部子集分数和全量分数直接比较。
- [x] 没有把 NLL 下降直接写成任务能力提升。

需要单独保留的偏离是：原计划中的 B0/checkpoint MATH paired preflight 没有在正式 S1 前
可靠关闭。它没有破坏训练身份，但造成了训练后 MATH 评测的成本风险，不能事后勾成通过。
