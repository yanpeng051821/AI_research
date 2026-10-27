---
pipeline_tag: text-generation
datasets:
  - jingyaogong/minimind_dataset
tags:
  - minimind
  - pretraining
  - from-scratch
  - research-checkpoint
  - custom-runtime
---

[English](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain) | **简体中文**

# MiniMind Dense 64M：mini 单轮预训练

> 公开研究 checkpoint。本文说明本次实际训练和原生加载方式，不宣称完整复现官方模型、可靠聊天能力或已获无条件商用授权；未选择统一的模型权重许可证。HF 地址与下载验证结果以实际发布记录为准。

## 1. 模型概况

本模型是一次从随机初始化开始的小规模预训练实验。学习者在带练中实现模型与训练组件，参照 MiniMind-3 Dense 对齐结构，用官方 mini 语料完成一个 epoch。目标是完成可追溯的训练、恢复、验证与交付流程，并观察固定同分布验证 NLL 是否下降，不是复现官方发布模型的完整能力。

| 项目 | 本次模型 |
| --- | --- |
| 类型 | Decoder-only 自回归语言模型，Dense |
| 独立参数 | 63,912,192，约 63.91M |
| 结构 | 8 层，hidden size 768，FFN 中间维度 2,432 |
| Attention | Q 头 8、KV 头 4，Q/K RMSNorm，RoPE base 1,000,000 |
| 词表 | 6,400，使用 MiniMind tokenizer |
| 权重共享 | embedding 与 lm_head 共用 Parameter |
| 输入模型 | seed=42 随机初始化，未加载官方预训练权重 |
| 已经历阶段 | 本次 Pretrain；尚未进行本模型的 SFT、DPO 或 RL |
| 导出 | 原生 FP32 safetensors 包，约 256.20 MB |
| 加载接口 | 包内 load_model.py；不是 Transformers AutoModel 适配 |

结构和 tokenizer 的上游参考固定为 [MiniMind 源码提交 f659b557](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f)。引用它不表示继承其发布权重或评测成绩。

## 2. 训练数据

| 项目 | 实际采用版本 |
| --- | --- |
| 数据集 | [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5) |
| 固定 revision | 312afb4f76391145c6902f765bb51691c09a12f5 |
| 来源文件 | pretrain_t2t_mini.jsonl，1,270,238 条 |
| 正式训练 | 1,265,235 条 |
| 固定验证 | 5,000 条，不是独立最终测试集 |
| 排除 | 3 条；人工约束作用于整个相同正文组 |
| 划分 | 按原始正文精确哈希分组，seed=43 抽取验证组；重复正文留在同一切分 |
| 编码 | BOS + 最多 766 个正文 token + EOS，右侧 padding 到 768 |
| 监督 | causal shift 后的正文与 EOS，PAD 不参与 loss |

编码后的完整输入序列精确检查未发现 train/validation 交叉。训练集约 3.92% 的记录发生正文截断，丢弃约 8.55% 的原始正文 token；这两个比例分母不同。训练累计有效预测目标为 301,810,690，固定验证集有效目标为 1,206,758。

没有完成全量事实核查、近似去重、原始文档级隔离、全量个人信息或版权清理；精确不重复不代表没有语义重叠或污染风险。原始和派生 JSONL 不随本模型包重新分发。

公开的数据规则与身份见[固定数据合同](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/data_recipe/DATA_CONTRACT.md)。本模型仍按预训练目标监督正文，不因为原文中含问答就视为 assistant-only SFT。

## 3. 训练配方

| 项目 | 实际配置 |
| --- | --- |
| 设备 | 单张 RTX 4060 Ti 8 GB |
| 训练状态与计算 | FP32 参数状态，CUDA BF16 autocast，确定性算法，关闭 TF32 |
| 更新范围 | 全部独立可训练参数，单个 AdamW 参数组 |
| micro-batch / 累积 | 4 / 64，有效完整窗口 256 条 |
| Loss | 当前累积窗口 loss_sum / 有效 causal target 总数 |
| 优化器 | AdamW，lr=5e-4，betas=(0.9,0.999)，eps=1e-8，weight decay=0.01 |
| 学习率 | 无 warmup，按 optimizer update 的余弦计划衰减至 5e-5 |
| 梯度裁剪 | 全局 norm 上限 1.0 |
| 预算 | 一个 epoch，4,943 次更新；末尾窗口 83 条、21 个 micro-batch |
| 验证 | 初始化、每 500 次更新和最终，遍历固定 5,000 条 |
| 保存 | 每 100 次更新及最终，50 个完整恢复点 |

训练于北京时间 2026-10-25 10:43:09 开始，2026-10-26 02:56:35 完成，用时约 16 小时 13 分 26 秒。正式运行正常退出，没有使用恢复短跑的中间参数作为训练起点。

启动前以完整正式调度执行 4 步连续与 2+2 步恢复对照，另检查至 step 20 的短跑；对照通过限于已测条件，不承诺跨设备逐位恢复一致。

完整规则见[固定训练合同](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/training_recipe/TRAINING_CONTRACT.md)与[执行配置](https://github.com/yanpeng051821/AI_research/blob/34a26c3b33ed91486343ac11964eac317e8ed72b/training_engineering/case_studies/minimind/experiments/pretrain_mini_one_epoch_v1/training_recipe/training_config.json)。该 Git 提交包含已核对的核心运行源码和配方，不主张它就是训练启动当时的 HEAD；本次导出与发布工具也包含在所链接提交中。

## 4. 结果与成本

### 4.1 固定验证 NLL

| 指标 | 训练前随机初始化 | 本次最终模型 |
| --- | ---: | ---: |
| 固定完整验证集 token 加权 NLL | 8.912246 | 1.998545 |

11 个实际验证点持续下降，后期改善收窄。逐步训练 loss 涉及变化的 batch 和参数，不将它与固定验证 NLL 的差直接解释成泛化差距。

该结果支持本次固定文本分布上的预测改善，不证明聊天、事实、数学、代码、工具调用或安全能力提升；没有独立最终测试或下游基准成绩，也不把结果标为这些任务通过。

### 4.2 实际成本

本次 run 的训练窗口约占墙钟 96.01%，checkpoint 保存约占 2.33%，验证约占 1.64%。端到端吞吐约 5,167 个有效 target/秒。更新记录中的最大 peak allocated 约 3.77 GiB，更新结束 reserved 约 3.37～3.88 GiB；不是所有阶段的连续显存监控。未采集累计耗电或账单，不报告人民币成本。

### 4.3 人工生成抽查

2026-10-27，学习者在本机亲自执行 v2 包的 load_model.py，并在会话中提供以下三条终端输出。条件为 CPU FP32、eval、关闭梯度、贪心生成、最多 32 个新 token；输入为 BOS 加文本前缀，不使用对话模板。下列内容只展示新增文本，保持用户提供的原始输出，不替模型补全句尾。

**场景续写**

输入：`春天的风吹过田野，`

```text
树叶在微风中轻轻摇曳。小草在微风中轻轻摇曳，阳光照耀着大地，微
```

输出与春天场景相关，语句基本通顺，但“在微风中轻轻摇曳”重复，表达较程式化。

**问候续写**

输入：`你好，`

```text
我有一个问题想问。您好，欢迎来到这里。请问您有什么问题需要帮助吗？我最近在学习编程，但是不知道从
```

出现问候与问答式文本，但在提问者与应答者之间切换，未稳定维持单一助手身份；不能把它记为可靠对话能力。

**概念续写**

输入：`机器学习是什么，`

```text
它如何工作？机器学习是一种人工智能的分支，它使用算法和统计模型来让计算机从数据中学习，并根据这些数据做出决策。
```

模型先补完问题再接概括性解释，符合整段文本续写行为。这段概括基本合理，但单条输出不是系统事实正确性评测。

这三条是临时选择前缀的人工观察，不是预先冻结的独立测试，不计算准确率。它们支持“已观察到一定短中文续写能力，同时存在重复与说话者边界不稳定”，不能证明稳定聊天、指令遵循或广泛知识泛化。未核查是否存在相近训练语料。入口未记录本次实际生成 token 数和停止原因，前两条停在半句可能是 32-token 上限，不能确认是 EOS 停止或据此判断不会结束。

## 5. 加载与使用

使用时应从模型仓库的明确 revision 下载整个模型包，再在包目录中运行；仓库为 https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain；发布 revision 见 GitHub 发布记录，不能只下载权重后忽略自定义运行代码。

```shell
python -I -B load_model.py --prompt "春天的风吹过田野，" --max-new-tokens 8
```

依赖版本记录在包内 requirements.txt：PyTorch 2.14.0、Transformers 4.57.6、tokenizers 0.22.2、safetensors 0.8.0；选择适合平台的 PyTorch 构建。本次只验证已有依赖环境中的本机 CPU FP32 加载，没有完成另一台机器的干净安装验证。

加载入口先创建包内配置对应的模型，再加载 model.safetensors，使用包内 tokenizer；默认 CPU FP32、eval 和关闭梯度。它是预训练续写入口，不套用 SFT 对话模板。当前保守限制 prompt token 与生成预算之和不超过已训练长度 768；这不是架构绝对长度上限，长上下文质量未验证。

包内 Python 是自定义运行代码，应在审阅后执行。不能用 AutoModel.from_pretrained 代替此入口，也不承诺 HF 在线推理服务自动支持。

## 6. 导出验证与身份

比较 A（原工程直接加载最终 checkpoint）与 B（新进程从包内代码、配置和 safetensors 重建模型），两者使用相同 CPU FP32 条件。全部模型状态摘要、embedding/lm_head 共享关系一致；两条右侧 padding 短前缀的 token、mask、logits 及一次短贪心续写一致，logits 最大绝对差为 0。独立 UTF-8 续写命令也通过。

这验证的是已测条件下的导出正确性，不是能力评测、CUDA/BF16 等价或 HF 发布后下载验证。v2 修正 v1 的 Windows 标准输出编码，不改变逻辑参数状态。

| 身份 | SHA256 |
| --- | --- |
| 最终源 checkpoint | 8d3cb52867e0c74b35c5931bfb9c90de3a7b6589fd9466fd68d00c9593431e38 |
| 当前 v2 model.safetensors | 3f1d1e6f585d516149d62d33670ef20cb29caa11c1759786586879e19fa3cc22 |
| 模型内部状态摘要 | 8d10d76090b2714c055a95123c3583c6941b1ecb6af84a10f0f68dbe21cb783f |
| 核心运行合同 | b8c2d23cbecfb1d0c92aaa00c40ffb0d0af9ef0a379bae1b487ac9de23d4bf58 |

文件哈希依赖序列化字节，内部状态摘要依赖张量名称、dtype、形状和数值。公开包装发生变化时会产生新的文件清单，不能沿用旧清单伪称未改动。

## 7. 来源与许可

MiniMind 固定源码提交附有 [Apache-2.0 许可证](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE)。tokenizer 来自该提交的 model/tokenizer.json 与 model/tokenizer_config.json，当前 JSON 内容相同，仅换行字节不同；实际训练和导出身份使用本项目编码审计中的文件哈希。

[固定数据集卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md)同时声明 Apache-2.0 与 CC-BY-NC-2.0，并介绍多来源语料，但未在已查资料中建立本次 mini 文件到全部来源许可的明确映射。本审查也未确立训练后权重的完整再分发授权范围。

因此本说明不选择统一的整体或权重 license，不宣称可无条件商用，也不因公开可下载而授予未明确的额外权利。公开包附上游代码许可证及来源说明，不把它们改称本模型权重的统一许可证。原创部分与模型权重的完整许可范围仍待澄清；学习、非商用或私有标签不能自动补足缺失授权。这里记录来源与风险，不作法律意见。

## 8. 贡献与边界

核心模型、早期数据及训练组件由学习者在带练中实现。助手按明确委托完成部分整合、重复测试、数据与配方核验、运行操作、分析、导出及文档整理；不将全部工程记为学习者独立完成。

本模型用于学习和受控研究，不作为可靠事实来源或高风险决策系统。尚未做系统安全、偏差、训练文本记忆或个人信息泄露评测。推理包不含 optimizer、scheduler、sampler 与 RNG，不能替代完整恢复 checkpoint。

## 9. 发布文件

recipes/ 保存实际训练配置、核心合同及无本机路径的数据配方摘要；evidence/ 保存逐点验证 NLL、成本、曲线、原始人工抽查和本地导出核对。完整源码、冻结配方与数据准备入口见上述固定 GitHub 链接。上游许可证及来源说明位于 licenses/ 与 THIRD_PARTY_NOTICES.md，它们不替代尚未明确的整体或权重许可。
