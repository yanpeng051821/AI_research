---
language:
  - zh
pipeline_tag: text-generation
base_model: yanpeng051821/minimind-dense-64m-mini-pretrain
base_model_relation: finetune
datasets:
  - jingyaogong/minimind_dataset
tags:
  - minimind
  - sft
  - full-finetuning
  - research-checkpoint
  - custom-runtime
---

[English](README.md) | **简体中文**

# MiniMind Dense 64M：mini 单轮 SFT

> 本地模型卡草稿，更新于2026-11-01；本SFT模型尚未上传新HF仓库。这是教学与研究产物，不是可靠助手。验证NLL及部分结束行为改善，但固定诊断题上的任务完成未超过自己的预训练基线。评分为助手初评，包含1条待裁定，未经过独立复核。

## 1. 模型概况

本模型由yanpeng在AI辅助下，基于[自己训练的mini预训练模型](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain)完成全参数SFT。该预训练模型从随机初始化开始训练，没有加载官方MiniMind预训练权重。架构与tokenizer参考MiniMind，但本实验不等同于完整复现官方模型，也不继承其发布模型的评测成绩。

本次要验证的是：在固定条件下，仅监督助手目标，是否能让已有续写模型更好地完成助手任务。工程流程已完成，但**任务改善目标未达到**。本文同时保留这项负结果和已经观察到的局部变化。

| 项目 | 实际模型 |
| --- | --- |
| 类型 | Dense decoder-only自回归语言模型 |
| 独立参数 | 63,912,192，约63.91M |
| 结构 | 8层；hidden size=768；FFN中间维度2,432 |
| Attention | Q头8、KV头4；Q/K RMSNorm；RoPE base=1,000,000 |
| Tokenizer | MiniMind，词表6,400；BOS=1、EOS=2、PAD=0 |
| 权重共享 | embedding与lm_head共用一个Parameter |
| SFT起点 | 自己预训练第4,943步；新建本阶段AdamW与scheduler |
| 已经历阶段 | 自己的mini预训练及本次一轮SFT；未进行DPO或RL |
| 评测模型S1 | 最终第49,087步，未按checkpoint效果择优 |
| 导出 | 原生FP32 safetensors；权重文件255,659,304字节，约255.66 MB |
| 加载接口 | 本地自定义Python runtime，不是Transformers AutoModel适配 |
| 上下文范围 | 训练与评测预算768；更长上下文的质量未验证 |

架构及tokenizer的上游参考为[MiniMind提交f659b557](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f)。架构参考和本次训练配方选择是来源记录，不表示逐项复现官方完整训练链路。

## 2. 数据与监督

### 2.1 数据版本

| 项目 | 实际数据 |
| --- | --- |
| 上游来源 | [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5) |
| 固定revision | 312afb4f76391145c6902f765bb51691c09a12f5 |
| 来源文件 | sft_t2t_mini.jsonl，905,718条 |
| 最终接受完整对话 | 787,425条 |
| 训练 / 验证 | 785,377 / 2,048条 |
| 有限修订 | 14条身份介绍、2条具体内容修订 |
| 最终准备阶段暂存 | 21,073条；部分为未核实语境，不等于全部判错 |
| 修订后重复另存 | 15条 |
| 训练 / 验证有效助手目标 | 326,249,045 / 833,052，按causal shift后统计 |

准备过程检查基础结构、工具声明及参数Schema、真实模板编码、长度和有效目标。此前阶段先处理超长、精确重复、两条已确认错误示范和未声明调用；最终准备再按限定规则修订、暂存、重编码、去重并切分。不同阶段数量对应不同输入版本，不能全部独立从原始总数相减。

共享任一归一化user正文或归一化完整对话的记录形成连通组。归一化采用NFKC、casefold及空白合并，只用于分组，不修改正文。seed=20261031按完整组选择验证数据，开发与审核样例所在组保护在训练侧。定义范围内的跨集精确对话和归一化问题重叠为0；这不等于清除了全部语义近重复或外部评测污染，validation也不是独立最终盲测。

### 2.2 编码与身份规则

完整模板编码超过768的对话整条排除，不截断。模板先完整渲染，再以add_special_tokens=False编码，避免重复添加特殊token；右侧padding到768。助手正文、保留的思考与调用文本、完整消息结束序列参与监督；角色头、user/system、工具结果和padding不作为这些位置自身的目标。上下文仍参与模型计算，标签只在causal loss中shift一次。

14条限定身份修订将yanpeng说明为本次训练者，并保留MiniMind架构与项目贡献；没有全局替换人物或项目名称。对应旧思考清空。这个处理规则**不代表模型已经可靠记住自己的身份**，后文补充身份诊断没有通过。

内容正确性只完成有边界的抽样审核，匿名合并工具结果的对应关系只有有限结构保证。语义重叠、全量事实、安全、个人信息泄露及完整来源权利范围没有全部核清。计划发布的推理包不包含原始或派生训练JSONL。

## 3. 训练配方与成本

| 项目 | 实际配置 |
| --- | --- |
| 硬件 | 单张NVIDIA RTX 4060 Ti，8 GB显存 |
| 训练系统 | 本项目自写PyTorch循环，不是TRL Trainer |
| 参数 / 运算精度 | FP32参数状态，CUDA BF16 autocast |
| 参数更新范围 | 全部唯一可训练参数，同一AdamW组 |
| micro-batch / 梯度累积 | 4 / 4，常规窗口16条完整对话 |
| Loss | causal交叉熵；窗口loss sum除以该窗口全部有效助手目标数 |
| AdamW | lr=1e-5；betas=(0.9,0.999)；eps=1e-8；weight_decay=0.01；foreach/fused=False |
| 学习率调度 | 无warmup，按optimizer更新计数余弦衰减至1e-6 |
| 梯度裁剪 | token归一化后全局范数阈值1.0；日志记录裁剪前值 |
| 随机性 / worker | seed=42、sampler_seed=42；num_workers=0 |
| 训练量 | 完整1个epoch，49,087次更新、196,345个micro-batch |
| 最后一次更新 | 1条对话、1个micro-batch、413个有效目标，不丢尾批 |
| 验证 | 开头、每5,000步及结尾，全2,048条，BF16工程监控 |
| 保存 | 每1,000步与结尾，共50份完整恢复checkpoint |
| 停止策略 | 累计活动运行预算14小时，更新边界保存暂停；本轮未触发 |

训练于北京时间2026-10-31 17:43:07开始、2026-11-01 04:26:24成功退出，墙钟约10小时43分17秒。其中更新窗口约10小时12分24秒、保存与receipt约22分46秒、全验证约6分53秒，其余启动及编排约1分13秒。本轮没有额外租卡或付费评分调用；未测量累计耗电和货币运行成本。

更新日志中的峰值allocated约3.77 GiB，更新后allocated约0.789 GiB，reserved范围约3.37–3.88 GiB；这些是特定时点的allocator读数，不是整机显存的连续监控。

最终代码在完整调度计划下验证过同设备连续20步与10步后恢复到20步一致。正式训练的步数、记录和目标总数、尾批、scheduler、AdamW状态、sampler顺序、数值有限性及最终checkpoint字节已核验。50份checkpoint/receipt已盘点，但未逐份完整计算历史文件hash并加载；也不保证跨设备位级复现。

## 4. 评测与观察

### 4.1 固定配对诊断

B0为自己的预训练模型，S1为最终SFT模型。核心32题包括日常问答、指令约束、给定材料和固定多轮上下文，各8题；另有4条工具补充题，不执行真实服务。

两边实际prompt文本与input_ids逐条相同，共用system、固定历史、工具声明、tokenizer与模板，预填空think区域。生成采用CUDA FP32、eval/inference、greedy、batch=1、最大新增256、总预算768、EOS=2，关闭TF32/autocast。NLL覆盖全部2,048条验证记录、batch=4、833,052个助手目标。两边均使用原逐题检查点及训练前批准的V2地点题补丁。

| 指标 | B0 | S1 |
| --- | ---: | ---: |
| 全验证FP32 NLL | 2.3954561617 | 1.7210819774 |
| 核心任务判断 | 2通过、30失败 | 0确定通过、31失败、1不确定 |
| 核心确定通过率 | 6.25% | 0%；不确定题全算通过的上界3.125% |
| 工具补充任务 | 0/4通过 | 0/4通过 |
| EOS / 达到长度上限 | 0 / 36 | 17 / 19 |
| 平均 / 中位新增token | 256 / 256 | 155.19 / 256 |

NLL下降约28.15%，但它衡量在真实上下文下预测助手目标的表现，不是自由生成整段答案的正确率。这里使用SFT验证数据与助手监督，不能直接同预训练模型卡中另一验证集、全文监督的NLL比较。

核心配对为30条双方未通过、1条确定回退、1条不确定，无确定正向翻转。待裁定的钥匙位置题复述了正确位置，却没有清楚独立回答最终位置，仍保留为不确定；即使判通过也未超过B0的2题。

所有语义评分是助手阅读完整输出后的初评，尚未独立复核，也不表示用户已逐题全部审核。hash、覆盖、引句与聚合检查不能证明语义判断正确。36题是已接触的诊断集，不是公开基准或独立盲测。

### 4.2 局部变化与失败

一条地点题要求根据记录只回答开放地点，S1输出：

```text
青禾图书角周三开放，开始时间14:20，结束时间17:10，地点是二楼东侧。
```

S1保留了正确时间并生成EOS，而B0重复带错误开始时间的文本。但S1仍附加日期和时间，违反“只回答地点”，不能记整题通过。

收音机概括题中，S1保留人物、更换电池及恢复工作的因果信息，却直接复制原始两句话，没有按要求概括成一句。其他一些回复只是重复用户问题然后结束。

工具相关问题包括用lookup_book查天气、生成未闭合调用JSON、添加工具结果没有的湿度与风速等，没有实际执行工具。因此，重复减少或EOS增多不等于可靠遵循指令、事实正确或工具使用安全。

### 4.3 身份补充诊断

主评测后另补3条身份问题：你是谁、由谁训练、预训练从随机初始化还是接续官方权重。最终S1在相同FP32贪心设置下，三条均达到256-token上限，未正确说明预期的yanpeng/MiniMind来源；有一条声称没有实际训练数据。第三题给了两个候选描述，不是无提示召回测试。

这3题是事后诊断，不计入原36题分母、不是盲测，也没有做B0配对。模型卡中的训练者与权重来源依据工程记录，不依据模型自述。

## 5. 本地使用

草稿阶段，原生S1模型包与本目录的聊天辅助脚本分开保存，尚未组装正式上传包。计划的发布内容包括model.safetensors、config.json、tokenizer/、runtime/、load_model.py、requirements.txt、中英文模型卡及[chat.py](chat.py)。

将这些文件组装到同一目录并安装依赖后，可从包目录执行：

```shell
python -I -B chat.py --prompt "你好，请简单介绍一下你自己。" --max-new-tokens 256
```

这是观察模型行为的用法，**不承诺能正确自我介绍**。chat.py默认CPU FP32，使用与评测相同的普通中文助手system，经包内chat template渲染，设置add_generation_prompt=True、open_thinking=False，再以add_special_tokens=False编码。它只生成一次单轮回复，不执行工具、不采样重试；prompt加生成预算超过768时报错，不静默截断。不承诺CPU与正式CUDA评测逐值相同。

原load_model.py --prompt仍是原始文本续写/加载检查，不会自动加SFT聊天模板。不能改用AutoModel.from_pretrained代替，也不承诺HF在线推理服务自动支持。

依赖记录为PyTorch 2.14.0、Transformers 4.57.6、tokenizers 0.22.2、safetensors 0.8.0；PyTorch构建应匹配平台。当前只核验已有本地依赖环境，未在其他机器干净安装验证。运行前应审阅包内自定义Python；文件hash不是签名或代码沙箱。

## 6. 导出与复现

最终checkpoint按相同架构严格加载，不改变原生FP32逻辑参数，再导出为模型包。新进程离线使用包内runtime、配置和tokenizer重建，完整状态摘要与共享Parameter一致；两条短前缀的输入、mask、logits及一次短贪心生成逐值一致，最大logit差0。该检查不等于所有输入或跨设备等价。

| 身份 | SHA256 |
| --- | --- |
| 最终SFT checkpoint | 181ce5bbc64c0c6ad0f3b79682f71df33206665c8212182422ce1363fec1e03a |
| model.safetensors | 4fdc2b4d4f2bbc1f85ca0a7a68eb615fea96cfaf3c7ddcf9cb9c8e03dcb995b5 |
| 模型逻辑状态摘要 | 733c18c9775eadd7f67330fabb0568ab61103f9d5fba995e6d967405c904cc62 |
| B0初始状态摘要 | 8d10d76090b2714c055a95123c3583c6941b1ecb6af84a10f0f68dbe21cb783f |
| SFT训练合同 | 3ac716dee78ce33d5d418e7f1f5e7e6b176dcdad6d06c52ac608322172c1abe9 |
| SFT数据合同 | 55bea98df2af3f6105fc03e06711b2602ba38e678b412720dcfff928e9f1da4d |

源码项目为[AI_research](https://github.com/yanpeng051821/AI_research)，实验位于training_engineering/case_studies/minimind/experiments/sft_mini_v1。本草稿尚未推送；公开前需要绑定实际可访问的Git/HF revision、发布清单并完成下载重载，不把尚不存在的S1仓库或未公开提交写成可访问入口。

修改说明文档或添加chat.py需要生成新的发布清单；本次已评测原生模型包及其清单保持不变。完整恢复checkpoint中的optimizer、scheduler、sampler和RNG仍保存在本地，推理包不替代这些状态。

## 7. 用途与限制

本模型适合学习、观察训练效果与复现有边界的负结果，不是生产助手、可信知识来源、可靠身份回答者或自主工具Agent。已观察到严重重复、复制问题/材料、漏掉指令约束、无依据内容和工具误用。

768以外的长上下文、英文能力、系统安全与偏差、隐私泄漏、训练文本记忆等未充分评估。英文版模型卡不代表模型具备英文生成能力。看到结果后没有额外训练、挑选checkpoint、改解码或放宽规则来制造提升。

核心组件由学习者在带练中实现；委托的整合、核验、运行、评测、导出与文档由AI编程助手参与完成，不将整个工程描述成学习者全部独立实现。

## 8. 来源与许可状态

MiniMind固定提交的[源码许可证](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE)及tokenizer来源与本模型权重分别说明。[固定数据集卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md)标注Apache-2.0与CC-BY-NC-2.0；现有来源检查未建立每个文件的完整许可映射及训练后权重再分发的完整范围。

本文不宣称统一权重/整体包许可证或无条件商用权，也不自动选择license标签。研究或非商用标签不能自行补足缺失授权。适用声明、公开再分发范围及原创部分条款仍需发布前核对；这里记录不确定性，不作法律意见。

**发布状态：仅本地草稿；尚未上传S1到HF、完成固定revision下载验证或获得本次公开发布确认。**
