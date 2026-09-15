# 11 正式 S1 结果、结论边界与发布

## 1. 最终状态

本案例已经完成一次真实的全参数 SFT 工程闭环：数据冻结、B0、预飞行、pilot、恢复、正式
训练、validation NLL、GSM8K、通用回归面板、异常评测中止和模型发布准备均有落盘证据。

最终决策为：

```text
训练工程：COMPLETE
实验结果：ITERATE
模型发布：RESEARCH CHECKPOINT
数学能力结论：NOT ACCEPTED
```

`ITERATE` 不是训练失败。它表示训练目标确实被优化，但现有证据同时发现通用能力回退和
MATH-500 生成预算异常，不能把 checkpoint 提升为最终可用模型。

## 2. 训练身份与成本

| 项目 | 冻结值 |
| --- | --- |
| Base model | `Qwen/Qwen3-0.6B-Base` |
| Base revision | `311c62e88814bff7206909ccd330bab0a784743b` |
| 数据 | OpenR1-Math 派生的 16K artifact |
| 训练记录 | 61,224 |
| Validation 记录 | 1,968 |
| 参数更新 | 全参数 |
| 精度 | BF16 计算，FP32 参数加载 |
| 单卡 batch | 1 |
| 梯度累积 | 128 |
| Epoch / optimizer steps | 1 / 479 |
| 学习率 | `4e-5`，cosine + 3% warmup |
| 训练硬件 | 1 x A100-SXM4-80GB |
| 训练时长 | 31,867 秒，约 8 小时 51 分钟 |
| 训练 token | 约 3.55 亿 |
| 最终 train loss | `0.568926` |
| CUDA peak allocated / reserved | 38.97 / 76.63 GiB |

最终权重为约 2.3 GB 的完整 `safetensors` 模型。14 GB 的训练器目录包含 AdamW、scheduler、
RNG 和 checkpoint 状态，只用于恢复与审计，不上传模型仓库。

## 3. 已完成的结果

### 3.1 目标数据拟合

同一 1,968 条 validation artifact、同一 11,367,594 个有效 completion token：

```text
B0 token-weighted NLL: 0.780483
S1 token-weighted NLL: 0.546614
相对下降:              29.96%
```

逐样本 mean NLL 的配对平均从 `0.764678` 降至 `0.511764`，差值 95% bootstrap CI 为
`[-0.255148, -0.250673]`；1,968 条样本全部改善。它证明目标分布拟合明显增强，但不等价
于数学答案正确率一定提高。

### 3.2 GSM8K

1,319 题的 `qem`：

```text
B0: 0.476118
S1: 0.514784
差值: +0.038666（+3.87 个百分点）
```

其中 199 题由错变对，148 题由对变错，972 题不变。这是当前最明确的目标任务正向证据。

### 3.3 通用回归面板

59 项、25,256 个配对样本的主指标宏平均从 `0.543412` 降至 `0.528562`，下降 1.49 个
百分点；按样本加权平均下降 1.43 个百分点。57 项 MMLU 宏平均下降 1.48 个百分点，
ARC-Challenge `acc_norm` 下降 2.22 个百分点，HellaSwag `acc_norm` 下降 1.15 个百分点。

该面板中 15 项改善、43 项下降、1 项不变。不能用少数改善任务覆盖整体回退，也不能把
这种回退直接归因于单一机制；具体原因留给下一阶段的逐样本与数据分布分析。

## 4. MATH-500 为什么没有正式结果

20 题 x 4 次、512-token smoke 中，S1 的 80 条输出全部触及长度上限，pass@1:1 从 B0
`0.25` 降至 S1 `0.05`。这是“正确率 + 截断行为”的混合结果，不是正式 MATH 能力结论。

原 v1 full 合同为 500 题 x 4 次、最大 32K 新 token。S1 在约 2 小时 55 分钟内只完成
204/2000 条，动态预计还需约 17 小时，因此按预算门禁中止。这个运行没有完整结果，不能
进入指标表，也不能把未完成题目记为错误。

训练数据的 supervised token 长度中位数约 4,843、均值约 5,888、P95 约 13,972；SFT
明显强化了长推理风格。最终目录的 `generation_config.json` 把 EOS 设为 `<|im_end|>`，而
模型 `config.json` 保留 base 的 `<|endoftext|>`。不同推理后端是否读取 generation config
会影响停止行为。发布时保留已评测 artifact，不在事后静默修改 EOS，并在模型卡中要求
显式设置 stop tokens。

## 5. 评测合同如何修订

本次实际运行使用的 v1 合同被保存为历史文件。新的 v2 合同不追补或篡改历史结果，而是
约束未来运行：

```text
先做固定规模预算探针
-> 检查停止率与截断率
-> 估算总 GPU 小时和费用
-> 全部门禁通过
-> 才允许扩大评测
```

MATH 相关入口当前全部为 `deferred` 且要求显式授权。用户已决定本阶段不继续 MATH，所以下
一步是分析已完成的 GSM8K、regression 和训练数据，不是继续租卡补跑。

## 6. Hugging Face 发布合同

模型仓库只包含：

- 最终模型、tokenizer、chat template 和 generation config；
- 模型卡；
- 冻结训练配置与 run manifest；
- NLL、GSM8K、regression、MATH smoke 和 MATH 中止摘要；
- 文件 SHA-256 清单。

不包含原始训练数据、14 GB optimizer checkpoint、完整逐样本回归输出或任何 token。模型卡
必须明确 `research checkpoint`、通用回退、MATH 未完成以及 EOS/stop-token 注意事项。

## 7. 工程收尾验收

- [x] 正式训练 manifest 为 `completed`，global step 为 479。
- [x] 最终模型与 tokenizer 文件齐全。
- [x] checkpoint-479 含 optimizer、scheduler 和 RNG，可用于恢复审计。
- [x] NLL、GSM8K 和 regression 配对结果完成。
- [x] MATH 不完整运行被标记为 `aborted_by_budget_guard`。
- [x] 新合同禁止未授权的 MATH 运行。
- [x] Hugging Face 文件上传完成并校验远端可加载：
  `yanpeng051821/qwen3-0.6b-openr1-math-sft-s1`，revision
  `45b20dd99ce04515212a98bde019c1ffcbaece80`。
- [x] 完整 regression 明细与轻量证据包已保存到移动硬盘：
  `E:/small_model_post_training_gate0b/formal_s1_2026-09-13/`，本地 SHA-256 与服务器一致。
- [x] 明确放弃 14 GB Trainer optimizer checkpoint；释放数据盘后不能从 step 479 精确续训。
- [x] Git 中提交最终文档、合同、测试与精简证据。

Git 与 Hugging Face 收尾均已完成，本案例的训练工程阶段已关闭。此后又完成了 54 条固定
分层 GSM8K 样本的错误分析，结果位于 `analysis/gsm8k_target_capability/`。当前仍不应直接
追加训练；需要先选择并冻结下一项可证伪假设，例如 256/512-token 解码预算敏感性，或
训练数据与通用能力回退之间的归因分析。
