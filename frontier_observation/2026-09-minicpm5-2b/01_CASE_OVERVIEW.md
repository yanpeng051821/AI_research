# MiniCPM5-2B：案例总览

> 记录日期：2026-09-12
> 研究类型：前沿观察，不是复现实验
> 当前结论：适合深读和做阶段 checkpoint 对照，不适合照官方规模完整复现

## 1. 为什么研究它

MiniCPM5-2B 与我们的长期目标接近：它不是只做分类的 encoder，而是能够生成文本、
推理、调用工具并支持长上下文的 dense causal language model。更重要的是，官方同时
公开了多个训练阶段的模型入口，使“能力在哪个阶段出现”有机会从猜测变成 checkpoint
对照问题。

## 2. 已核验的模型事实

根据官方模型卡和仓库 README：

- 架构为标准 `LlamaForCausalLM`；
- 总参数量约 25.17 亿，非 embedding 参数约 19.82 亿；
- 42 层 Transformer；
- GQA 使用 16 个 query heads 和 2 个 key/value heads；
- 原生上下文长度为 131,072；
- 同一最终 checkpoint 通过 chat template 的 `enable_thinking` 切换 Think/No-Think；
- 支持工具调用，官方推荐由 SGLang 的 MiniCPM5 parser 解析 XML 风格工具调用。

“标准 Llama 架构”意味着它可以被 Transformers、vLLM、SGLang 等常见栈直接加载，
但不意味着训练 recipe 也只是普通 Llama recipe。模型能力仍可能主要来自数据、训练阶段、
长上下文配方、RL 和蒸馏。

## 3. 官方阶段树

当前可见模型入口至少覆盖：

```text
MiniCPM5-2B-Base
        |
MiniCPM5-2B-Midtrain
        |
MiniCPM5-2B-SFT
        |
MiniCPM5-2B (RL + OPD 后的最终模型)
```

官方对训练过程的概括是：

```text
Base training
  -> stable training
  -> decay training
  -> mid-training
  -> SFT
  -> 专项 RL teachers
  -> OPD 合并多个专家能力
  -> final model
```

这不是“所有小模型都必须遵循的固定顺序”，而是 MiniCPM5 为自己的目标能力选择的
recipe。我们研究它，是为了理解每种训练目标解决的问题和证据，而不是机械照抄顺序。

## 4. 官方披露的数据和后训练方法

官方列出的训练数据入口包括 Ultra-FineWeb、Ultra-FineWeb-L3、UltraX、
UltraData-Code、UltraData-Math、UltraData-SFT-2605、UltraData-SFT-Agent-2609
和 UltraData-RL-2609。

后训练被分为三步：

1. SFT 建立 deep-thinking、通用对话和目标输出行为；官方给出的规模是 400B tokens。
2. 面向数学、代码、Agent 和写作等方向训练专项 RL teacher。
3. 使用 OPD 将多个专家能力合并回一个发布模型。官方描述为在 response 各位置比较
   teacher 与 student 的全词表分布，并使用反向 KL 构造训练信号。

400B-token SFT、多个 RL teacher 和专家合并远超当前个人复现预算。对我们有价值的是
阶段定义、数据职责、产物组织和对照方法，而不是复刻原始算力规模。

## 5. 当前证据边界

| 问题 | 当前能否确认 | 原因 |
| --- | --- | --- |
| 模型与阶段 checkpoint 存在 | 能 | 官方模型集合和模型卡可核验 |
| 官方描述的训练阶段与数据入口 | 能确认官方披露 | 官方 README 和数据页面给出说明 |
| 外部研究者可以完整复现最终模型 | 不能 | 仍需逐项核对数据覆盖率、精确配置、代码和日志 |
| RL + OPD 的增益全部来自算法本身 | 不能 | 数据、teacher、采样、评测和训练预算也同时变化 |
| 官方汇总 benchmark 可直接与我们的 S1 比较 | 不能 | 模型、数据、prompt、解码和评测 harness 均不同 |

因此后续需要把“发布方报告的结果”与“我们能独立复核的结果”分开。

## 6. 与当前路线的关系

当前 Qwen3-0.6B 正式 S1 继续运行，不因本案例切换模型。MiniCPM5 先用于校准：

- 为什么保存 Base、Midtrain、SFT、RL/蒸馏后的阶段 checkpoint；
- 如何用同一评测合同分析阶段增量和能力回归；
- 如何区分架构收益、数据收益、训练目标收益和推理设置收益；
- 一个小模型为什么可能同时具备生成、推理、工具调用和 Agent 行为。

当前不做两个模型绝对分数的横向比较。未来若做实验，应选择一个可控的阶段问题，
例如比较 SFT checkpoint 与 final checkpoint 在固定 prompt、固定 scorer 下的行为差异。

## 7. 官方来源

- [MiniCPM5-2B 官方模型卡](https://huggingface.co/openbmb/MiniCPM5-2B)
- [MiniCPM5 官方模型集合](https://huggingface.co/collections/openbmb/minicpm5)
- [OpenBMB MiniCPM 官方仓库](https://github.com/OpenBMB/MiniCPM)
- [UltraData 入口](https://ultradata.openbmb.cn/)
