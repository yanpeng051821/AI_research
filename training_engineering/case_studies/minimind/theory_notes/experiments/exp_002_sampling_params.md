# 实验编号：exp_002_sampling_params

## 1. 实验目标

本次实验想回答的问题是：

同一个 SFT 模型，在不同采样参数下，输出质量会发生什么变化？

重点观察：

- temperature 变低后，输出是否更稳定。
- temperature 变高后，输出是否更发散、更容易幻觉。
- repetition_penalty 变大后，复读是否缓解。
- top_p、top_k 收紧后，回答是否更保守。

实验一已经说明：

```text
full_sft_edu_512 更像助手，但仍然存在复读、停止控制差、代码逻辑错误等问题。
```

实验二要进一步区分：

```text
哪些问题可以通过采样参数缓解？
哪些问题属于模型本身能力不足？
```

## 2. Baseline

Baseline 使用同一个 SFT 模型：

```text
full_sft_edu_512
```

模型结构：

```text
hidden_size = 512
num_hidden_layers = 4
```

权重文件：

```text
./out/full_sft_edu_512.pth
```

## 3. 实验组

本实验不换模型，只换采样参数。

建议跑 3 组：

| 组别 | temperature | top_p | top_k | repetition_penalty | 预期风格 |
|---|---:|---:|---:|---:|---|
| safe | 0.3 | 0.8 | 默认或 30 | 1.15 | 稳定、保守、较少发散 |
| balanced | 0.8 | 0.9 | 默认或 50 | 1.10 | 平衡、自然、有一定变化 |
| creative | 1.1 | 0.95 | 默认或 80 | 1.05 | 更发散、更容易幻觉 |

如果当前 `eval_llm.py` 不支持 `top_k` 或 `repetition_penalty` 参数，就先只改：

```text
temperature
top_p
```

如果使用 `eval_llm_report.py`，再同时记录：

```text
top_k
repetition_penalty
```

## 4. 固定变量

本实验固定：

```text
模型：
full_sft_edu_512

prompt：
使用第 9 节的 3 个 prompt。

max_new_tokens：
120

load_from：
model

save_dir：
out

GPU：
Tesla T4 16GB
```

本实验只改变：

```text
temperature
top_p
top_k
repetition_penalty
```

## 5. 推理命令

### 5.1 safe 组

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

### 5.2 balanced 组

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_p 0.9
```

### 5.3 creative 组

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 1.1 \
  --top_p 0.95
```

## 6. 如果使用结构化评估脚本

如果服务器上已经有 `eval_llm_report.py`，建议用下面的方式跑。

### 6.1 safe 组

```bash
python eval_llm_report.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.3 \
  --top_p 0.8 \
  --top_k 30 \
  --repetition_penalty 1.15
```

### 6.2 balanced 组

```bash
python eval_llm_report.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 50 \
  --repetition_penalty 1.10
```

### 6.3 creative 组

```bash
python eval_llm_report.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_edu \
  --hidden_size 512 \
  --num_hidden_layers 4 \
  --max_new_tokens 120 \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 80 \
  --repetition_penalty 1.05
```

## 7. 评估 prompt 集

为了控制实验成本，本次只用 3 个 prompt。

### 7.1 复读测试

```text
请用三句话解释什么是机器学习。
```

观察重点：

- 是否严格三句话。
- 是否复读相同表达。
- 是否自然停止。

### 7.2 代码测试

```text
请用 Python 写一个计算斐波那契数列的函数，并简单解释。
```

观察重点：

- 是否输出代码块。
- 代码是否可运行。
- 是否仍然混入 factorial。
- 是否解释正确。

### 7.3 边界测试

```text
北京今天的天气怎么样？
```

观察重点：

- 是否编造实时天气。
- 是否承认无法获取实时信息。
- 是否暗示自己能查询但实际没有工具。

## 8. 输出样例记录

| Prompt | safe 输出摘要 | balanced 输出摘要 | creative 输出摘要 | 观察 |
|---|---|---|---|---|
| 机器学习三句话 | 基本能解释机器学习，但明显复读“从数据中学习，并从中学习”，没有严格三句话 | 表达比 safe 略自然，但仍重复“目标”“数据训练模型”等句式，没有严格三句话 | 更发散，出现“多维度模型”“多个独立模型”等不够准确的表达 | safe 更稳但仍复读；balanced 稍自然；creative 准确性下降 |
| Python 斐波那契 | 严重崩坏，输出 `find_find_find...` 重复串，没有代码逻辑 | 输出代码块，但混入伪代码、括号不闭合、函数语义错误 | 输出代码块，但混入欧几里得算法、乱码式代码和错误递归 | 代码正确性不能靠采样参数解决，属于模型能力和训练数据问题 |
| 北京天气 | 正确说明无法提供实时天气，建议查看应用或网站 | 正确说明无法获取实时数据，建议使用天气应用或网站 | 说明无法实时查询，但表达略别扭：“请提供北京今天的天气信息” | 实时边界相对稳定，creative 组语言自然度下降 |

## 9. 指标记录

| 指标 | safe | balanced | creative | 结论 |
|---|---:|---:|---:|---|
| 平均 tokens/s | 约 43.5 / 60.1 / 59.3 | 约 39.0 / 60.2 / 59.1 | 约 41.6 / 60.1 / 58.8 | 速度主要受生成长度和首次响应影响，采样参数影响不大 |
| 是否复读 | 是 | 是 | 轻度复读且更发散 | 当前复读问题不能只靠 temperature/top_p 解决 |
| 是否跑题 | 代码题严重跑偏 | 代码题跑偏 | 机器学习解释和代码题都更跑偏 | temperature 升高后跑题风险更高 |
| 是否遵循三句话 | 否 | 否 | 否 | 长度/格式遵循能力较弱 |
| 代码是否可运行 | 否 | 否 | 否 | 代码能力不足是模型/数据问题 |
| 是否编造实时信息 | 否 | 否 | 否，但表达略别扭 | 天气边界相对稳定 |

## 10. 判断标准

不要只看哪组“看起来更长”或者“更像大模型”。

建议按下面顺序判断：

```text
第一优先级：
是否回答了用户问题。

第二优先级：
是否事实正确或代码正确。

第三优先级：
是否复读、跑题、停不下来。

第四优先级：
语言是否自然、有条理。
```

如果某组回答更丰富，但事实更错，不能算更好。

如果某组回答更短，但更准确、更稳定，反而可能更适合教学场景。

## 11. 预期结论

大概率会看到：

```text
safe：
更稳定，但可能更模板化。

balanced：
回答更自然，但可能偶尔复读或发散。

creative：
更容易出现幻觉、跑题、重复或奇怪表达。
```

但是也要注意：

```text
采样参数只能改变“怎么从 logits 里选 token”。
它不能凭空补足模型不知道的知识。
```

所以如果代码逻辑错了，调 temperature 可能偶尔碰巧变好，但不是根本解决方案。

根本解决方向包括：

- 更好的代码数据。
- 更强的 SFT 数据。
- 代码垂类 LoRA。
- DPO 或偏好数据，让模型偏向更简洁、更正确的答案。
- 更大模型或更长训练。

## 12. 实验结论

本实验比较了 `full_sft_edu_512` 在 safe、balanced、creative 三组采样参数下的输出差异。

实验结果显示：

```text
safe 组：
在实时天气边界上最稳定，能够明确说明无法提供实时天气。
但在机器学习解释中仍然复读，在代码任务中甚至出现 find_find_find 的严重重复崩坏。

balanced 组：
机器学习解释比 safe 略自然，天气边界也稳定。
代码任务能输出代码块，但代码结构和语义仍然错误。

creative 组：
输出更发散，机器学习解释准确性下降，代码任务混入欧几里得算法、乱码式代码和错误递归。
天气边界仍能保持，但表达略别扭。
```

因此，对于当前 MiniMind 小模型：

```text
1. 教学问答默认更适合 safe 或 balanced 参数，不建议使用 creative 参数。
2. temperature/top_p 可以影响稳定性和发散程度，但不能根本解决代码正确性问题。
3. 复读问题只能部分通过采样缓解，根本上仍依赖训练数据、模型能力、停止控制和偏好优化。
4. 实时信息边界在三组参数下都相对稳定，说明这类拒答/边界表达已经被 SFT 学到了一部分。
5. 后续如果要面向教育 Agent，默认策略应该偏稳定：低 temperature、较保守 top_p、适度 repetition penalty。
```

本实验的核心结论：

```text
采样参数决定“怎么说”，不决定“会不会”。

模型不会写正确的斐波那契函数时，调 temperature 不能真正让它获得代码能力。
如果模型已经知道答案，采样参数可以影响答案是稳定、自然还是发散。
```

## 13. 下一步

完成本实验后，可以进入两条路线：

```text
路线 A：
进入第 8 周 LoRA，学习低成本垂类适配。

路线 B：
先做一个小型教育 prompt 评测集，建立更贴近未来教育 Agent 的 benchmark。
```

建议优先走路线 A。

原因：

```text
LoRA 是后续 DPO、教育场景适配、小模型垂类任务的基础。
```
