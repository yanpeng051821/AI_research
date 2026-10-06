# 实验编号：exp_001_pretrain_vs_sft

## 1. 实验目标

本次实验想回答的问题是：

SFT 相比 pretrain，到底让模型多学会了什么？

更具体地说，本实验不把 pretrain 和 SFT 简单评价成“差模型”和“好模型”，而是分别观察：

- pretrain 是否具备基本的文本续写能力。
- SFT 是否在 pretrain 基础上学会了对话格式、指令遵循和助手回答风格。
- SFT 是否更能处理用户问题、代码请求和实时信息边界。

## 2. Baseline

Baseline 使用 pretrain 模型。

推荐权重：

```text
pretrain_edu
```

或者使用当前训练得到的 pretrain 权重，例如：

```text
pretrain_512
pretrain_768
```

Baseline 的预期能力不是“回答问题”，而是“根据前文继续生成自然文本”。

## 3. 实验组

实验组使用 SFT 模型。

推荐权重：

```text
full_sft_768
```

或者使用当前训练得到的 SFT 权重，例如：

```text
full_sft_512
full_sft_edu_512
```

实验组的预期能力是：

- 能理解用户的问题。
- 能以助手身份直接回答。
- 能遵循格式、长度、代码等指令。
- 能在无法获取实时信息时说明边界。

## 4. 固定变量

为了保证实验可比较，除了模型权重之外，其他条件尽量保持一致。

```text
模型结构：
hidden_size、num_hidden_layers、max_seq_len 与对应权重保持一致。

数据：
不重新训练，只比较已有权重。

prompt 集：
使用本文件第 9 节固定的 5 个 prompt。

采样参数：
temperature = 0.3
top_p = 0.8
top_k = 30
repetition_penalty = 1.15

max_new_tokens：
续写任务建议 120。
问答任务建议 120。
代码任务可放宽到 180。

GPU：
Tesla T4 16GB。
```

## 5. 变化变量

本次实验只改变一个变量：

```text
模型权重：
pretrain 权重 vs SFT 权重
```

不要同时改变采样参数、prompt、max_new_tokens 或模型结构。

否则实验结论会混在一起，无法判断提升来自 SFT，还是来自采样参数。

## 6. 训练命令

本实验不重新训练模型，只比较已有权重。

如果需要补充训练来源，可以记录原始训练命令：

```bash
python trainer/train_pretrain.py \
  --epochs 1 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --max_seq_len 128 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path dataset/pretrain_t2t_mini.jsonl \
  --from_weight none
```

```bash
python trainer/train_full_sft.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --max_seq_len 512 \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --data_path dataset/sft_t2t_mini.jsonl
```

具体命令以服务器实际训练记录为准。

## 7. 推理命令

### 7.1 pretrain 模型

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight pretrain_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8
```

如果使用结构化评估脚本：

```bash
python eval_llm_report.py \
  --load_from model \
  --save_dir out \
  --weight pretrain_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8 \
  --top_k 30 \
  --repetition_penalty 1.15
```

### 7.2 SFT 模型

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 16 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8
```

如果使用结构化评估脚本：

```bash
python eval_llm_report.py \
  --load_from model \
  --save_dir out \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 16 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8 \
  --top_k 30 \
  --repetition_penalty 1.15
```

如果要评估 `full_sft_edu_512.pth`，命令应写成：

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8
```

MiniMind 的 `eval_llm.py` 会按下面规则拼接权重文件名：

```text
./out/{weight}_{hidden_size}.pth
```

所以：

```text
--weight full_sft --hidden_size 768
=> ./out/full_sft_768.pth

--weight full_sft_edu --hidden_size 512
=> ./out/full_sft_edu_512.pth
```

不要写成：

```text
--weight full_sft_768 --hidden_size 768
```

否则会寻找：

```text
./out/full_sft_768_768.pth
```

这个文件通常不存在。

注意：

如果比较的是 512 hidden size 的 pretrain 和 768 hidden size 的 SFT，那么结论里要明确写出：

```text
本实验同时改变了训练阶段和模型规模，因此只能观察整体差异，不能严格归因于 SFT。
```

更严谨的比较方式是：

```text
pretrain_512 vs full_sft_512
pretrain_768 vs full_sft_768
```

## 8. 训练日志摘要

本实验以已有权重为主，训练日志可以先空着。

后续复盘时建议补充：

```text
pretrain loss：
SFT loss：
pretrain 显存：
SFT 显存：
pretrain tokens/s：
SFT tokens/s：
异常：
```

## 9. 评估 prompt 集

本实验使用 5 个 prompt。

### 9.1 续写 1：教育场景

```text
人工智能正在改变教育行业。过去，学生主要依靠教材和课堂学习，而现在
```

观察重点：

- 是否能围绕教育继续写。
- 是否语义连贯。
- 是否出现明显重复或跑题。

### 9.2 续写 2：训练流程

```text
深度学习模型的训练通常包括数据准备、模型构建、损失计算和参数更新。具体来说
```

观察重点：

- 是否能接住“训练流程”这个主题。
- 是否知道数据、模型、loss、optimizer 的基本关系。
- 是否出现伪概念或胡乱堆词。

### 9.3 问答 1：概念解释

```text
请用三句话解释什么是机器学习。
```

观察重点：

- 是否直接回答问题。
- 是否真的控制在三句话左右。
- 是否能给出准确、清楚的解释。

### 9.4 问答 2：代码能力

```text
请用 Python 写一个计算斐波那契数列的函数，并简单解释。
```

观察重点：

- 是否输出 Python 代码。
- 代码是否能运行。
- 是否解释递推关系。
- 是否把斐波那契数列说错。

### 9.5 边界测试：实时信息

```text
北京今天的天气怎么样？
```

观察重点：

- 是否承认无法获取实时天气。
- 是否建议用户查看天气应用或调用工具。
- 是否直接编造天气。

## 10. 输出样例对比

| Prompt | Baseline 输出摘要 | 实验组输出摘要 | 观察 |
|---|---|---|---|
| 续写 1 | 能围绕教育继续写，但明显复读“智能辅导”等短语 | 会进入教育主题，但严重复读“从教育机构到教育机构” | 两者都能接住主题，但都存在重复；SFT 不一定比 pretrain 更适合纯续写 |
| 续写 2 | 能提到训练目的、参数调整、测试结果，但后半段反复重复“调整模型参数” | 更像助手回答，会列出数据预处理、特征工程等条目，但条目严重重复 | SFT 输出形式更结构化，但内容质量和停止控制仍弱 |
| 问答 1 | 能解释机器学习，但只生成 2 句左右，内容重复 | 能以问答方式解释，但没有严格遵守“三句话”，后面继续复读 | SFT 更像助手，但指令遵循和自然停止仍不稳定 |
| 问答 2 | 没有写 Python 函数，只解释斐波那契，并出现错误序列和重复 | 输出了 Python 代码块，但代码逻辑错误，把 `fibonacci` 写成类似阶乘递归，还出现重复注释和未定义变量 | SFT 学会了“代码回答格式”，但没有真正掌握代码语义 |
| 边界测试 | 能说明无法实时获取天气，并建议查看天气预报 | 能说明无法提供实时天气，但说“我可以帮助查询”有轻微工具能力幻觉 | 两者在实时边界上表现都比预期好，SFT 仍可能暗示自己能查询 |

## 11. 指标记录

| 指标 | Baseline | 实验组 | 结论 |
|---|---:|---:|---|
| 平均 tokens/s | 约 37-60 | 约 45-60 | T4 上 16.65M 模型推理速度较稳定 |
| 是否复读 | 是 | 是 | 两者都有重复，SFT 结构更像助手但仍会复读 |
| 是否遵循指令 | 弱 | 中等偏弱 | SFT 有提升，但“三句话”和代码指令没有完全满足 |
| 是否自然停止 | 弱 | 弱 | 两者都容易在 max_new_tokens 附近继续重复 |
| 是否编造实时信息 | 较少 | 较少但有轻微工具幻觉 | SFT 会说无法提供实时天气，但“我可以帮助查询”需要警惕 |
| 显存占用 | 待填写 | 待填写 | 待填写 |

## 12. 失败案例

建议至少记录 2 个失败案例。

```text
失败案例 1：

失败模型：

失败表现：

失败原因猜测：

改进方向：
```

```text
失败案例 2：

失败模型：

失败表现：

失败原因猜测：

改进方向：
```

常见失败类型：

- pretrain 把问题当成普通文本继续写，没有直接回答。
- SFT 回答格式像助手，但事实内容错误。
- 模型复读同一句话。
- 模型没有自然停止。
- 模型编造实时天气、版本、新闻等信息。
- 代码题生成了伪代码或错误递推。

本轮已观察到的失败案例：

```text
失败案例 1：

失败模型：
pretrain_edu_512

失败表现：
代码 prompt 要求“用 Python 写一个计算斐波那契数列的函数”，模型没有输出代码，只解释了斐波那契数列，并且生成了错误、重复的数列。

失败原因猜测：
pretrain 主要学习文本续写，不稳定具备指令遵循能力。它看到“斐波那契”后更倾向于继续写百科式解释，而不是执行“写函数”的指令。

改进方向：
使用 SFT 或代码数据微调；后续可以用 LoRA 做代码/教学场景适配。
```

```text
失败案例 2：

失败模型：
full_sft_edu_512

失败表现：
回答“请用三句话解释什么是机器学习”时，前几句接近正确，但后面继续重复相似句子，没有自然停止。

失败原因猜测：
SFT 学到了助手式回答，但模型规模和训练数据有限，对长度约束和停止 token 的控制不足。采样参数虽然偏稳定，但仍无法完全解决模型自身的重复倾向。

改进方向：
尝试提高 repetition_penalty；检查 SFT 数据中是否存在重复模板；后续用 DPO 或偏好数据优化“简洁、不复读”的回答。
```

```text
失败案例 3：

失败模型：
full_sft_edu_512

失败表现：
回答“北京今天的天气怎么样？”时，能说明无法提供实时天气，但又说“我可以帮助您查询北京的天气情况”。

失败原因猜测：
模型没有真实工具调用能力，但 SFT 数据中可能包含助手承诺帮助查询的表达，因此产生了轻微工具能力幻觉。

改进方向：
在 Agent 或工具调用阶段明确区分“不能查实时信息”和“调用天气工具后可以查”；在普通 SFT 中加入更严格的边界样本。
```

## 12.1 本轮实验状态

当前已经完成：

```text
pretrain_edu_512：
5 个 prompt 已跑完。

full_sft_edu_512：
5 个 prompt 已跑完。
```

本实验已经完成第一轮输出收集。

## 12.2 代码 prompt 补充观察

SFT 代码 prompt 输出摘要：

```text
模型输出了 Python 代码块，并尝试定义 fibonacci(n)。
但函数内部把 n 重复赋值为 1，判断条件写成 if n <= n，并在 else 中返回 n * factorial(n-1)。
这不是斐波那契数列，而是混入了阶乘 factorial 的错误逻辑。
```

这个结果很重要：

```text
SFT 让模型学会了“看到代码请求时应该输出代码块”。
但当前模型规模和训练数据不足以保证代码逻辑正确。
```

所以对代码能力的判断不能只看：

```text
有没有代码块
```

还要看：

```text
代码是否可运行
函数语义是否正确
变量是否定义
是否混入其他算法
```

## 13. 结论模板

完成实验后，可以按下面格式写结论：

```text
本实验比较了 pretrain 模型和 SFT 模型在续写、问答、代码和实时信息边界上的表现。

pretrain 模型在续写类 prompt 上具备一定语言建模能力，但在问答类 prompt 上容易把问题当成普通文本继续生成，缺少助手角色和指令遵循能力。

SFT 模型相比 pretrain 更像对话助手，能更直接回答用户问题，也更容易遵循格式要求。但它的知识准确性、代码稳定性和长回答控制仍然有限。尤其在代码任务中，SFT 可以学会输出代码块，却不一定能保证代码逻辑正确。

因此，本实验说明：SFT 的主要作用不是凭空注入大量新知识，而是在 pretrain 语言能力基础上，学习对话格式、回答风格、指令遵循和安全边界。
```

本轮实际结论：

```text
1. pretrain_edu_512 已具备基础续写能力，但容易复读，问答和代码任务表现弱。
2. full_sft_edu_512 更像助手，会尝试按用户要求组织答案，例如列表、解释、代码块。
3. SFT 并没有自动解决事实准确性、代码正确性和自然停止问题。
4. 实时天气问题上，两个模型都能表达无法获取实时天气，但 SFT 仍出现轻微工具能力幻觉。
5. 下一步需要通过采样参数实验，判断复读问题能否通过 decoding 缓解；更深层的知识和代码能力问题，则需要更好的数据、模型规模或后续 LoRA/DPO 等方法。
```

## 14. 下一步

本实验完成后，下一步建议做：

```text
exp_002_sampling_params
```

实验问题：

同一个 SFT 模型，在不同 temperature、top_p、top_k、repetition_penalty 下，输出质量有什么变化？

这样可以把“模型能力”和“采样参数影响”拆开看。

## 15. 当前学习者理解记录

当前你已经给出了正确的实验设计直觉：

```text
如果分别评价两个模型：
pretrain 应该提供一段文字，看续写能力。
SFT 应该使用问答形式 prompt，看对话和指令遵循能力。

如果做一个轻量对比实验：
2 个续写 prompt。
2 个问答 prompt。
1 个边界测试 prompt。
```

这说明你已经开始从“会跑模型”进入“会设计模型实验”的阶段。
