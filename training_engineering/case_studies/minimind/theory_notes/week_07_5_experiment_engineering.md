# 第 7.5 周：实验工程与评估流水线

## 1. 为什么要补这一周

自己学习和实验室训练最大的差别之一，不是懂不懂模型，而是有没有一套稳定的实验方法。

自己学习时很容易停在：

```text
代码跑通了。
loss 降了。
模型好像能回答。
```

实验室或工程团队更关注：

```text
这个改动到底有没有提升？
提升在哪里？
有没有副作用？
别人能不能复现？
换一组样本是否还成立？
显存、速度、稳定性有没有变化？
```

所以这一周的目标不是追求复杂 benchmark，而是建立最小可用的实验流水线。

你后面要做 AI 教育 Agent 和小模型，这个能力非常关键。模型应用不是只会训练就够了，还要能证明：

```text
我的模型在目标场景中真的更好。
我的改动没有只是碰巧变好。
我的实验结果可以被自己和别人复现。
```

## 2. 放在路线中的位置

推荐位置：

```text
第 6.5 周：训练代码手撕复盘
第 7 周：推理与采样
第 7.5 周：实验工程与评估流水线
第 8 周：LoRA
```

原因：

```text
先会训，才能理解实验变量来自哪里。
先会推理，才能设计输出评估。
先会评估，再进入 LoRA / DPO / GRPO，才知道后续方法有没有真正带来提升。
```

这不是额外负担，而是后续所有训练阶段的基础设施。

## 3. 实验工程的核心思想

实验不是“多跑几次”。

实验是：

```text
固定目标。
固定基线。
只改变一个变量。
记录完整配置。
用一致的方法评估。
分析结果和失败案例。
```

最小实验闭环：

```text
实验问题
-> baseline
-> 实验变量
-> 训练或推理
-> 指标记录
-> 样例对比
-> 结论
-> 下一步
```

比如：

```text
问题：SFT 后模型是否比 pretrain 更像助手？
baseline：pretrain_edu_512
实验组：full_sft_edu_512
变量：是否经过 SFT
固定：hidden_size、num_layers、测试 prompt、采样参数、max_new_tokens
指标：人工打分、重复率、是否答非所问、tokens/s
结论：SFT 是否改善助手格式和指令遵循
```

## 4. MiniMind 最小评估维度

### 4.1 训练侧指标

训练时记录：

```text
loss
logits_loss
aux_loss
learning_rate
step
epoch_time / ETA
GPU memory
GPU utilization
tokens/s 或 samples/s
是否 NaN
是否 OOM
```

这些指标回答：

```text
训练是否稳定？
训练是否高效？
配置是否合理？
```

### 4.2 推理侧指标

推理时记录：

```text
输出质量
是否遵循指令
是否复读
是否幻觉
是否自然停止
tokens/s
首 token 延迟
总生成耗时
显存占用
```

这些指标回答：

```text
模型是否真的可用？
回答风格是否符合目标？
推理成本是否可接受？
```

### 4.3 任务侧指标

如果是 AI 教育场景，可以记录：

```text
解释是否清楚
是否能分步骤讲解
是否能发现学生误区
是否能给出合适练习
是否能根据学生水平调整难度
是否胡乱鼓励或过度自信
```

这类指标比通用 benchmark 更贴近你的目标方向。

## 5. 实验记录模板

每次实验至少记录：

```text
实验编号：
实验日期：
目标问题：
代码版本：
数据版本：
模型结构：
起点权重：
训练命令：
推理命令：
固定参数：
变化变量：
训练日志摘要：
评估 prompt 集：
结果摘要：
失败案例：
结论：
下一步：
```

示例：

```text
实验编号：exp_001
目标问题：SFT 是否提升助手回答格式？
baseline：pretrain_edu_512
实验组：full_sft_edu_512
固定参数：temperature=0.8, top_p=0.9, max_new_tokens=256
测试集：8 个 eval_llm 默认 prompt + 5 个自定义教育 prompt
观察结果：SFT 更像问答，但仍有重复和事实不稳
结论：SFT 有明显格式收益，但知识可靠性仍有限
下一步：构建教育场景小评测集
```

## 6. 推荐目录结构

建议在 `model_training_12_weeks/` 下新增实验记录目录：

```text
model_training_12_weeks/experiments/
  README.md
  exp_001_pretrain_vs_sft.md
  exp_002_sampling_params.md
  exp_003_lora_identity.md
```

先不要搞太复杂的 MLflow 或数据库。

现阶段最重要的是：

```text
每次实验有记录。
每次对比有 baseline。
每次结论有样例支撑。
```

## 7. 第一个实验：pretrain vs SFT

目标：

```text
比较 pretrain 权重和 SFT 权重的输出行为差异。
```

固定条件：

```text
同一个 hidden_size。
同一组 prompts。
同一组采样参数。
同一个 max_new_tokens。
同一张 GPU。
```

对比对象：

```text
pretrain_edu_512
full_sft_edu_512
```

记录维度：

```text
是否像助手回答。
是否直接续写用户问题。
是否结构清晰。
是否重复。
是否事实错误。
tokens/s。
```

输出形式：

```text
一张对比表。
3 到 5 个典型样例。
一句明确结论。
```

## 8. 第二个实验：采样参数对输出的影响

目标：

```text
理解 temperature、top_p、max_new_tokens 对模型输出的影响。
```

固定：

```text
同一个 SFT 权重。
同一个 prompt。
同一个 max_new_tokens，除非本轮实验就是测 max_new_tokens。
```

变量：

```text
temperature = 0.2 / 0.8 / 1.2
top_p = 0.7 / 0.9 / 0.95
```

观察：

```text
低 temperature 是否更稳定。
高 temperature 是否更发散。
top_p 变小时是否更保守。
重复是否增加。
幻觉是否增加。
```

## 9. 第三个实验：训练参数小消融

目标：

```text
理解 batch_size、max_seq_len、learning_rate 对训练稳定性和速度的影响。
```

推荐只做小实验，不要浪费大成本：

```text
每组只跑 200 到 1000 step。
只比较趋势，不追求最终效果。
```

可做变量：

```text
batch_size 8 vs 16
max_seq_len 256 vs 512
learning_rate 1e-5 vs 5e-5
```

记录：

```text
显存占用。
每 step 时间。
loss 是否稳定。
是否 OOM。
```

## 10. 实验工程里最重要的 5 个习惯

### 10.1 每次只改一个关键变量

不要同时改：

```text
数据
学习率
batch_size
模型大小
采样参数
```

否则结果变好了，也不知道是谁带来的。

### 10.2 永远保留 baseline

没有 baseline，就没有比较。

比如：

```text
LoRA 效果好不好，要和 full_sft 或未加 LoRA 的模型比。
DPO 效果好不好，要和 SFT 模型比。
GRPO 效果好不好，要和 DPO 或 SFT 模型比。
```

### 10.3 记录完整命令

包括训练命令和推理命令。

不要只写：

```text
跑了一版 SFT。
```

要写：

```text
python train_full_sft.py --epochs 1 --batch_size 8 ...
```

### 10.4 固定评估 prompt 集

每次用同一批 prompt 测，才能比较。

建议准备三类：

```text
通用问答 prompt。
代码 prompt。
教育场景 prompt。
```

### 10.5 记录失败案例

失败案例比成功案例更有价值。

要记录：

```text
模型什么时候复读。
什么时候胡说。
什么时候不遵守格式。
什么时候回答过短或过长。
什么时候看起来会但其实错。
```

## 11. 现阶段不急着做什么

暂时不急着做：

```text
复杂自动评测平台。
大规模 benchmark。
排行榜式评估。
MLflow / W&B 深度集成。
多机多卡性能 profiling。
```

先做最小闭环：

```text
固定 prompt
固定参数
记录输出
人工分类
总结结论
```

这已经足够把你从“能跑模型”推进到“能做实验”。

## 12. 完成标准

完成第 7.5 周后，你应该能做到：

```text
能设计一个 baseline vs experiment 的对比。
能写清楚实验变量和固定变量。
能记录训练命令、推理命令和结果。
能用同一组 prompt 对比两个权重。
能总结模型质量、速度、显存、失败案例。
能为 LoRA / DPO / GRPO 设计合理评估方式。
```

如果能做到这些，再进入 LoRA，你就不只是“训练一个 LoRA”，而是在做一个能证明价值的模型实验。

