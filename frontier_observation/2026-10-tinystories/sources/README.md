# TinyStories 原始资料

## 论文

- 标题：TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
- 本地文件：[TinyStories-2305.07759v2.pdf](TinyStories-2305.07759v2.pdf)。
- 版本：arXiv `2305.07759v2`，与当前带读版本一致。
- 下载来源：[arXiv 官方 PDF](https://arxiv.org/pdf/2305.07759v2)。
- 下载日期：2026-11-02。
- 文件大小：4,069,571 bytes。
- 页数：27。
- SHA-256：`e9d2d82534b4ccb7163559f00e68556988c49192a1433f860481544bc7dc2445`。

保存的是官方原始 PDF，未改写或重排。下载后已使用 `pdfinfo` 检查，文件可解析。
本目录用于原始资料归档；我们的理解与讨论继续写在案例总览中。

## 33M 发布模型的训练设置

2026-11-02 核对以下官方资料。这里保存来源摘记，具体教学内容待讨论后写入总览。

- [官方模型卡](https://huggingface.co/roneneldan/TinyStories-33M/blob/main/README.md)：
  学习率 `5e-4`、constant 学习率调度、weight decay `0.1`、Adam betas `0.9/0.95`、
  上下文长度 `512`、batch size `80`、梯度累积 `16`。
- [作者在讨论 #8 的回复](https://huggingface.co/roneneldan/TinyStories-33M/discussions/8)：
  Ronen Eldan 于 2023-09-02 说明，这个模型训练约 20 个 epoch，使用
  `TinyStories-train.txt`；4 张 V100，每张卡 batch 为 20，总 batch 为 80，
  梯度累积 16 次；其余列出的学习率、weight decay、betas 与模型卡一致。

按这些设置推算，一次完整累积后的参数更新对应 `20 * 4 * 16 = 1280` 条训练序列。
这是推算，不是作者直接报告的有效 batch 数字，也不代表 1280 篇完整原始故事。
未据此推算有效训练 token 总数，数据切分与 padding 等细节仍待确认。
这些说明针对 33M 发布模型，不证明所有规模、图 4 或固定 FLOPs 实验均使用同一配方。
来源列出了 Adam betas，但这里未确认具体优化器实现是 Adam 还是 AdamW。

## 优化器原理参考

- [PyTorch AdamW 官方实现与公式](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)：
  用于解释一阶矩、二阶矩、偏差修正与解耦权重衰减；2026-11-02 核对。
  这是通用算法参考，不是 TinyStories 原训练代码的证据。
