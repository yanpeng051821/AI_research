# 第 8 周：从对话数据到 SFT 训练实验

> 更新：2026-11-01。本轮正式SFT、固定FP32评测及HF研究模型发布已完成。NLL与部分停止行为改善，但核心任务初评未超过B0，有1题待人工裁定。另完成本机Attention后端短测，执行效率有明确优化空间，尚未替换正式实现。能力结果见[配对分析](../experiments/sft_mini_v1/results/paired_analysis_v1/README.md)，效率优化见10.8节。

## 1. 本周目标与完整链路

### 1.1 本周目标

1. 理解SFT如何利用对话上下文，只在约定的助手目标上学习。
2. 按格式与协议、内容质量、集合质量三条审核线处理真实语料，形成可追溯的数据配方。
3. 从第七周自己的预训练模型出发，确定模型与训练参数，验证短跑和恢复后完成正式训练。
4. 在相同评测条件下比较B0与S1，区分分布拟合、回复行为和任务完成。

第七周让模型从随机初始化开始学习续写，本周在已有权重上训练助手行为。我们继续复用前几周的模型、loss、优化器更新、sampler、验证和checkpoint组件；变化集中在对话表示、监督范围、数据选择、初始权重及实验配方，而不是从头重建训练系统。

### 1.2 输入、产物与完成标准

输入是第七周自己的mini预训练最终模型，以及上游提供的原始对话。第七周模型包及来源见[预训练归档](../experiments/pretrain_mini_one_epoch_v1/README.md)。本周交付的不仅是权重，还包括数据与训练合同、冻结评测、运行证据及训练前后分析。

```text
原始对话 -> 审核规则与处理 -> train / validation -> 数据配方冻结
                                                   |
                         模板 -> token -> 助手labels -> batch
                                                   |
B0模型与tokenizer + 已确认训练参数 -> 准备检查 -> 短跑 -> 恢复对照
                                                   |
                  最终配方冻结 -> 正式训练 -> 保存与运行观察
                                                   |
     训练前已冻结的评测与B0 ----------------------> S1配对评测 -> 分析与归档
```

本轮预期是相较B0有初步改善，不采用额外的8/32或EOS16/32门槛。主要观察核心任务通过数及正负翻转，辅以重复/结束行为和验证NLL；仅NLL下降不算助手能力已经改善。详细共识见[实验目标快照](../experiments/sft_mini_v1/EXPERIMENT_OBJECTIVES.md)。

### 1.3 阅读与实践顺序

| 章节 | 要解决的问题 | 本轮状态 |
| --- | --- | --- |
| 2–3 SFT特点与编码 | 一条对话怎样成为有正确监督的batch？ | 已实现并验证 |
| 4 数据处理与冻结 | 哪些示范能进入训练，怎样处理、切分和追溯？ | 数据配方已冻结，质量限制保留 |
| 5 评测约定与B0 | 在训练前怎样固定比较方式？ | B0采集与V2初评已完成 |
| 6 模型与训练配方 | 从哪里开始，更新哪些参数，采用什么计划？ | 已确认 |
| 7 短跑与恢复 | 链路可运行吗，恢复会不会改变训练轨迹，成本如何？ | 已验证 |
| 8 最终训练配方冻结 | 怎样绑定配置、身份与证据？ | 已冻结并复查 |
| 9 正式启动与运行观察 | 怎样后台运行、查看进度和处理恢复？ | 完整结束并验收 |
| 10 训练后评测与分析 | 改善来自哪里，退步在哪里，怎样归档并提出下一步实验？ | 评测、初步分析与发布完成；效率优化候选待验收 |
| 11 学习QA | 容易混淆的原理和用法 | 随实践维护在文末 |

评测分两次出现是有意的：**训练前固定规则并建立B0，训练后执行S1并分析**。不能等看到S1结果，再决定用哪些题或采用什么判法。

以下按工程依赖组织，不再把聊天中每次“下一步”原样追加。中间候选数量只解释对应处理阶段，最终身份以冻结文件为准；历史故障保留在产生它的环节，未完成的后续工作明确标注。

### 1.4 环境与执行约定

命令均从Windows PowerShell的implementation目录运行：

```powershell
Set-Location D:\pythonlearning\AI_research\training_engineering\case_studies\minimind\implementation
uv run --locked python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

新机器才需要先恢复依赖；不要在真实训练进行时重新同步或升级环境。必须使用匹配权重的本地tokenizer：assets/minimind_tokenizer，不能换成第二周教学BPE的词表。

所有生成任务拒绝覆盖既有输出。命令中指向历史版本的路径用于说明真实执行关系，复练应选择新输出目录，并使用相匹配的输入、代码和规则版本；不直接修改冻结文件或正在写入的训练目录。后文正式启动、恢复命令记录本次执行方式；本轮已经完成，不得对原目录重复启动。

原有助手标签、普通编码、Dataset与collator由学习者带练完成；后续工具适配、审计、批处理、配方与入口接线按用户委托由助手执行。用户确认处理规则与运行方案，不等于已亲自审核每条数据或独立编写全部实现。


## 2. SFT阶段的目标与监督特点

### 2.1 真实对话的输入形态

练习输入为 `data/source_samples/sft_sample.jsonl`。它保留来源文件开头的 20 条完整记录，不是正式训练集，也不是随机样本。若换机器缺少文件，可恢复既有样本并校验身份：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_source_samples.ps1 -DatasetType sft
```

每条记录通过 `conversations` 保存消息序列。以下摘自第一条记录的最后一组问答，仅用于观察编码，不代表已经审核为高质量示范：

```json
[
  {"role": "user", "content": "你最近在忙什么？"},
  {"role": "assistant", "content": "我目前专注于优化和提升自身的性能，持续改进我的响应能力与准确性。"}
]
```

这与 Agent 的消息列表形态相近，但普通问答不必出现 `tool`。完整候选中确实还有工具相关字段，后面需要单独验证，不能把普通对话测试推广到所有消息协议。

### 2.2 模板建立消息边界

模型不能直接处理 Python 字典。先用 tokenizer 的 chat template 把角色与正文排列成文本，再编码成 token ID：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
input_ids = tokenizer(text, add_special_tokens=False).input_ids
```

这里训练样本已经包含助手回答，不再追加等待生成的助手头；模板本身已插入消息边界，编码时也不额外添加一套 BOS/EOS。

本地模板会将没有思考正文的助手消息呈现为空 `<think>` 段；非空 `reasoning_content` 则进入对应思考区域。两者的关系是：`reasoning_content` 是源数据字段，think 标签是模板文本中的边界，二者都不等于模型实际内部计算过程。保留某段思考作监督，也不证明它正确。

### 2.3 输入位置与监督位置

SFT 仍使用自回归预测，但这里主要监督助手示范，而不是要求模型学习生成用户的提问。用户消息仍在 `input_ids` 中，并通过 attention 为助手预测提供上下文。

```text
用户消息、助手角色头：input_ids 保留，labels 设为 -100
助手监督区间：        input_ids 保留，labels 复制相应 token ID
padding：            input_ids 补 PAD，attention_mask=0，labels=-100
```

`attention_mask` 和 `labels` 的职责不同：前者描述输入的可见性，后者选择哪些目标参与 loss。用户标签为 -100，不等于用户输入从前向或反向计算中消失。

当前观察路径保留助手思考段、正文、`<|im_end|>` 和紧随其后的换行；不监督助手角色头。这是本轮数据合同接受的具体选择，不是所有 SFT 的固定规定。

## 3. 对话编码与训练输入链路

本章先把编码机制验证清楚，让数据审核有一个明确对象：不是只看JSON能否解析，而是知道它最终产生了哪些输入与预测目标。下面按“观察边界 → 构造labels → 封装编码 → 适配工具 → 组批 → 按索引读取”推进，后续审计和训练复用这条路径。

### 3.1 观察实际token与助手区间

#### 3.1.1 建立观察脚本

在 [scripts/inspect_sft_tokens.py](../implementation/scripts/inspect_sft_tokens.py) 中依次完成：读取第一条 JSONL、取得 `conversations`、套模板、编码、构造 labels，然后逐位置打印 ID、`repr` 文本及 label。

```powershell
uv run --locked python scripts/inspect_sft_tokens.py
```

脚本目前读取完整第一条记录；要复练上一节的单组问答，可在取得 messages 后临时使用 `messages = messages[-2:]`，观察完恢复完整对话。两种输入的长度不同，不应把单组问答的固定下标套到所有记录。

#### 3.1.2 核对实际边界

当前 tokenizer 的词表大小为 6,400，消息起止 ID 分别为 1、2，PAD 为 0。

| 文本片段 | 实际 token ID | 本步作用 |
| --- | --- | --- |
| assistant 角色头（含消息起始符与换行） | `[1, 1388, 570, 811, 234]` | 定位助手头，头部本身不监督 |
| 消息结束符与换行 | `[2, 234]` | 当前监督区间的结束序列 |
| `<think>`、`</think>` | `25`、`26` | 模板中的思考边界 |

两个完整边界文本分别是 `<|im_start|>assistant\n`、`<|im_end|>\n`，其中 `\n` 表示实际换行。`assistant` 被拆成 `ass`、`ist`、`ant` 三个普通 token。是否监督由其所在区间决定，不由“是不是特殊 token”决定。

最后一组问答共 45 token，分区如下，区间采用左闭右开：

| 区间 | 内容 | labels |
| --- | --- | --- |
| `[0, 13)` | user 消息及边界 | -100 |
| `[13, 18)` | assistant 角色头 | -100 |
| `[18, 24)` | 空 think 段及换行 | 对应 token ID |
| `[24, 43)` | 助手正文 | 对应 token ID |
| `[43, 45)` | EOS 与换行 | 对应 token ID |

因此当前规则得到 27 个监督目标。完整第一条记录包含 14 条消息、7 次助手回复，共 549 token、414 个监督目标。完整记录的第一个助手区间并不从位置 18 开始；通用实现必须查找边界，而不是写死下标。

打印单个 token 时使用 `repr(piece)`，能把换行显示为 `\n`，便于辨认边界。字节级 tokenizer 的单个 token 可能只是一个汉字的部分字节，单独 decode 出现替代字符不一定意味着数据损坏，应同时检查完整序列解码。

### 3.2 实现多轮助手标签

#### 3.2.1 实现标签构造

在 [src/minimind_lab/data/sft.py](../implementation/src/minimind_lab/data/sft.py) 中实现 `build_assistant_labels(input_ids, assistant_prefix, message_end)`。它只处理整数列表，不负责读文件、套模板或 padding。

先将所有位置设为 -100；找到助手头后，从头部之后寻找完整消息结束序列；只复制这段目标。其核心流程如下，与当前实现等价：

```python
labels = [-100] * len(input_ids)
found = False
for i in range(len(input_ids) - len(assistant_prefix) + 1):
    if input_ids[i:i + len(assistant_prefix)] != assistant_prefix:
        continue
    body_start = i + len(assistant_prefix)
    body_end = None
    for j in range(body_start, len(input_ids) - len(message_end) + 1):
        if input_ids[j:j + len(message_end)] == message_end:
            body_end = j + len(message_end)
            break
    if body_end is None:
        raise ValueError("assistant message end not found")
    labels[body_start:body_end] = input_ids[body_start:body_end]
    found = True
if not found:
    raise ValueError("no assistant message found")
```

完整函数还要在循环前拒绝空 prefix 和空 ending，避免空切片处处匹配。现有实现使用 `assistant_spans` 记录发现的区间，最终返回的是 labels，不是区间列表。

#### 3.2.2 循环边界与结束位置

长度为 N 的列表中查找长度为 M 的序列，最后合法起点为 `N-M`。因为 `range` 不包含右端点，所以需要 `range(N-M+1)`。例如 N=5、M=2，起点为 0、1、2、3；N=M 时仍应检查位置 0。

`body_end = j + len(message_end)` 则有另一层含义：j 是结束序列的起点，加上结束序列长度后，才能把 EOS 与换行一起纳入左闭右开区间。不要把“搜索最后合法起点的 +1”与“包含结束序列的长度”混为一谈。

内层 `break` 只停止寻找本次回复的结束位置。外层还要继续寻找下一次助手回复；若也在外层直接 break，就只会监督第一轮。

#### 3.2.3 验证多轮与异常

对照[助手监督测试](../implementation/tests/test_sft_data.py)，使用两段助手回复的小整数序列，手工写出每个位置的预期标签。这样能直接发现“只处理第一轮”“把角色头算入目标”“遗漏结束标记”等错误，而不依赖复杂真实文本。

还要检查：没有助手头应报错，助手消息缺少完整结束应报错，输入列表不应被修改。完整第一条真实记录另外用消息文本边界作独立参考，得到的标签与 token 搜索一致。

这些验证说明已测样例的边界正确，不代表原始token搜索能分辨任意正文中的伪角色标记。第3.8节说明后来增加的源边界拒绝与有效目标门禁。标签算法必须与这些输入前提一起理解；缺失结束直接拒绝，本轮不静默截断。

### 3.3 封装对话编码与causal shift

#### 3.3.1 编码器的职责

继续在同一文件实现 `encode_sft_conversation(messages, tokenizer)`，按下列顺序组织已经验证的步骤：

```text
messages
  -> prepare_sft_messages
       -> validate_sft_record({"conversations": messages})
       -> 深复制并转换工具字段，返回 prepared_messages 与 tools
  -> apply_chat_template(..., tools=tools, tokenize=False, add_generation_prompt=False)
  -> tokenizer(text, add_special_tokens=False).input_ids
  -> 用同一 tokenizer 编码 assistant_prefix 和 message_end
  -> build_assistant_labels(...)
  -> {"input_ids": list[int], "labels": list[int]}
```

把这些逻辑放在一个函数中，Dataset、观察脚本和后续数据审计才不会各自维护不同的模板与监督规则。这里的 validator 只解决基本输入结构，不会判断数学答案正确、请求是否有足够信息、工具调用是否合理。

加入工具适配后，基础 validator 由 `prepare_sft_messages` 在入口调用，编码器不重复校验。第3.5节解释转换接口及实现；普通消息仍保持原来的编码和监督行为。

#### 3.3.2 保持标签对齐

编码结束时 `input_ids` 与 `labels` 必须等长，label 仍放在对应目标 token 的原始位置，不提前左移或右移。后续已有 causal-LM loss 才执行：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

这使前一位置的 logits 预测后一位置的 token。例如第一个助手正文目标，由助手头最后一个位置的 logits 预测。编码阶段再 shift 一次，会造成预测错位。

训练统计应按 `labels[1:] != -100` 计算有效目标，而不盲目统计整个 labels；本节示例的位置 0 本来被屏蔽，所以两种计数恰好一致，不代表两种口径普遍等价。

#### 3.3.3 验证调用关系

测试使用一个受控 tokenizer 替身，固定模板返回值和边界 ID，检查函数是否正确调用模板、关闭额外 special tokens、保留原始消息，并得到预期 labels。其目的是隔离本地编排错误，不是拿替身证明真实 tokenizer 正确。随后再用真实 tokenizer 和 20 条本地记录核对实际路径。

### 3.4 工具字段与模板的接口

基础 validator 保证的是约定中的必需结构，不保证每个扩展字段已经满足模板接口。源行 794,513 提供了一条可用于追踪的单位换算对话：system 声明 `unit_converter`，user 要求将 100 公里转换成英里，assistant 发起调用，tool 返回结果，assistant 给出最终回答。已重新核对该行原始内容及 SHA256；这里验证数据流，不将其标为全项质量审核通过。

原始记录中存在嵌套 JSON：外层记录解析后，`tools` 仍是字符串，`tool_calls` 仍是字符串；解析调用列表后，`function.arguments` 又是一个 JSON 字符串。读取外层 JSON 并不会递归把所有字符串解析成对象。

| 对象 | 原始字段类型 | 本次模板输入 | 作用 |
| --- | --- | --- | --- |
| system.tools | JSON 字符串 | 解析成列表，作为模板的 `tools=` 参数 | 呈现工具名称、说明和参数定义 |
| assistant.tool_calls | JSON 字符串 | 解析成列表，保留在助手消息中 | 呈现助手调用动作 |
| function.arguments | JSON 字符串 | 当前模板允许保留该字符串；核验参数时另解析为字典 | 呈现调用参数，并检查必需字段 |
| tool.content | 文本，本例恰好是 JSON 文本 | 保留原文，不无条件解析所有 content | 作为工具观察进入上下文 |

只读实验对消息做深复制，再在副本上解析字段：

```python
messages = deepcopy(record["conversations"])
tools = json.loads(messages[0].pop("tools"))
for message in messages:
    if isinstance(message.get("tool_calls"), str):
        message["tool_calls"] = json.loads(message["tool_calls"])
text = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=False,
)
```

此片段需要 `from copy import deepcopy` 与 `import json`，仅说明这条已知结构记录的探针操作，不是正式通用转换函数。它没有处理所有空值、错误类型、多处工具声明或不合法参数，不应直接作为全量清洗入口。探针核对调用名与已声明工具一致，参数包含 value、from_unit、to_unit，并确认原始记录未改变；没有执行工具。

当前模板把 tool 消息包装成 user 区域里的 `<tool_response>`；这是该模板的文本表示，不应据此把原始 tool 角色改成真实用户发言。最终数据流为：

```text
工具声明 -> 用户请求 -> 助手工具调用 -> 工具结果 -> 助手最终回答
上下文     上下文       监督目标         上下文       监督目标
```

本例编码共 462 token，当前规则选中两个助手区间 `[327,393)`、`[442,462)`，shift 后共 86 个目标。独立检查选中片段包含工具调用和最终回答，不包含工具结果。角色头仍被屏蔽，保留的空 think 段与结束序列仍按现有规则参与监督。

这说明当前助手区间监督可以表达工具动作示范，但不表示有工具执行环境、训练已经启动或正式数据范围已经确定。这个阶段仅核对可行的数据流；随后用户明确委托实现，才将适配接入现有编码器，过程如下。

### 3.5 实现字段适配并接入编码器

新增的 `prepare_sft_messages` 放在已有 [data/sft.py](../implementation/src/minimind_lab/data/sft.py) 中，接收原始消息列表，返回两个对象：

```python
prepared_messages, tools = prepare_sft_messages(messages)
```

`prepared_messages` 是模板需要的消息副本，`tools` 是从首条 system 提取出的工具清单。为什么分开返回？因为当前模板通过独立的 `tools=` 参数呈现工具说明，而在消息列表里读取助手 tool_calls。两者接口不同，不应该只把原始字典原封不动传进去。

实现分四步：

1. **基础校验**：调用 `validate_sft_record`，结构不成立时先停止，不将坏输入交给模板。
2. **复制消息**：`prepared = deepcopy(messages)`，后续 pop、字段赋值和嵌套列表转换都只作用于副本。浅复制不能保护内层 function、参数等对象。
3. **转换并检查字段**：提取 tools，解析 tool_calls，检查模板所需的 function、名称与参数形态，保留工具正文、思考及其他字段。
4. **交给模板**：编码器使用这两个返回值，再继续原有 tokenization 与 labels 构造。

工具列表共用 `_parse_tool_list`：缺失、null、空白字符串、空列表和字符串 `"[]"` 归一为无列表内容；非空字符串用 `json.loads` 解析，结果必须是列表。`"{}"`、坏 JSON 和数字不会被当作空字段忽略，而是报出具体消息位置。

工具声明与调用采用本来源的 function 包装结构，名称必须为非空字符串。非空 tools 只支持首条 system；非空 tool_calls 必须属于 assistant。遇到后面的 system 再携带新工具，不擅自移动或合并，而是明确报错，等待确定新的表示规则。reasoning_content 接受字符串或 null；声明中的 parameters 若存在，必须是对象。

每次调用还必须显式携带 arguments。它可以是字典，也可以是能解析为字典的 JSON 字符串；字符串会被临时解析以核验形态，但输出仍保留原字符串的格式，因为当前模板支持这种输入。缺失参数不自动填 `"{}"`，数组或坏 JSON 也不交给模板。这里不验证参数是否满足完整 JSON Schema，不保证工具已声明、调用与结果已对应，不能把它当成完整协议审核器。

编码器的关键接入如下：

```python
prepared_messages, tools = prepare_sft_messages(messages)
text = tokenizer.apply_chat_template(
    prepared_messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=False,
)
input_ids = tokenizer(text, add_special_tokens=False).input_ids
# 后续仍调用 build_assistant_labels，不重新定义监督目标。
```

Dataset 和 collator 不需要重写：Dataset 原本就委托编码器，因此现在能经过新增适配；collator 仍只负责将已有 input_ids 与 labels 组批。适配没有重排对话、添加 system、移除 think 或执行任何工具。

测试见 [test_sft_tool_fields.py](../implementation/tests/test_sft_tool_fields.py) 和原有 [test_sft_data.py](../implementation/tests/test_sft_data.py)。它们分别验证字段转换/拒绝条件，以及编码器是否真的把转换后的消息和独立 tools 参数交给模板。深复制测试还修改返回对象的嵌套字段，确认不会反过来改动输入，避免只比较顶层对象就误判隔离已经成立。

真实核验结果：原有 20 条普通记录的 input_ids、labels 与适配前逐条完全一致；固定分组样例 plain、reasoning_only、tool_only 各 16 条，共 48 条全部可编码且有有效目标。单位换算记录仍为 462 token、86 个目标，工具调用被监督、结果未被监督。完整源文件 hash 重新核验一致，没有执行质量过滤或身份改写。详细身份、测试结果与范围见[适配验证报告](../experiments/sft_mini_data_inspection_v1/encoding_adapter_v1/README.md)。

源行 776,360 现在也能编码，但其调用未声明工具的问题仍未解决。这不是漏掉内容审核，而是有意分清“能被表示”和“可作为正确示范”。接下来仍需完成协议审核、正文边界处理、有效目标门禁与长度分析，再讨论数据取舍和冻结。

### 3.6 组批、padding与监督对齐

#### 3.6.1 collator 承接可变长样本

`SFTCollator(tokenizer, max_length)` 接收多条编码结果，补齐到指定长度，返回三个形状相同的 `torch.long` 张量。它不重新生成原有 labels，否则会把用户区域也变成监督目标。

以 PAD=0、max_length=5 为例：

```python
features = [
    {"input_ids": [90, 10, 30, 20, 21],
     "labels": [-100, -100, 30, 20, 21]},
    {"input_ids": [90, 30, 20],
     "labels": [-100, 30, 20]},
]
# 组批后
input_ids = [[90, 10, 30, 20, 21], [90, 30, 20, 0, 0]]
attention_mask = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
labels = [[-100, -100, 30, 20, 21], [-100, 30, 20, -100, -100]]
```

原有用户标签与新增 padding 标签都为 -100，但原因不同；只有 padding 对应 attention_mask=0。这样模型还能读取真实用户上下文，同时不把补齐位置当作预测任务。

#### 3.6.2 实现顺序与历史排错

每条 feature 先检查输入与 labels 等长、输入长度不超过上限，再执行：

```python
real_length = len(input_ids)
padding_length = self.max_length - real_length
input_ids = input_ids + [self.pad_token_id] * padding_length
attention_mask = [1] * real_length + [0] * padding_length
labels = labels + [-100] * padding_length
```

最后将三个 batch 列表分别转为 `torch.tensor(..., dtype=torch.long)`。初始化还要拒绝小于 2 的长度和缺失的 pad_token_id，调用时拒绝空 batch。

本次修正过两个边界问题：

1. 长度恰好等于 max_length 时可以直接使用，拒绝条件是 `>`，不是 `>=`。
2. attention mask 必须使用 padding 前保存的 real_length。补齐后再取 len，会把 padding 也算成真实输入，甚至得到错误长度的 mask。

当前 collator 采用固定指定长度，不是自动按 batch 最长样本动态补齐；超长直接报错，不在此处静默截断。max_length 因而只是当前调用的容纳上限，不能拿一次练习值当作正式训练选择。

#### 3.6.3 验证目的

测试同时检查张量形状、dtype、真实输入可见性、原始监督区间、padding 标签和输入 features 未被修改。形状正确还不够：若 mask 全为 1，形状可能完全正确，训练语义却已经改变。

### 3.7 Dataset、Sampler与DataLoader的职责

最终SFTDataset使用JSONL字节偏移索引读取原始记录，而不是将异构可选字段统一成Arrow表：

```text
Sampler提供记录索引
  -> SFTDataset[index]
       -> 按该行的字节偏移读取JSON
       -> 取conversations
       -> encode_sft_conversation
       -> 返回input_ids与labels
  -> DataLoader收集多条feature
  -> SFTCollator构造input_ids / attention_mask / labels
```

Dataset不决定样本顺序，collator不更新参数，DataLoader也不负责冻结切分。训练文件和验证文件已经由数据准备阶段生成，读取时不能再自动随机切分一次。sampler的固定排列和已提交位置属于训练状态，具体恢复见第7章。

**这次为什么改成偏移读取？** 早期实现使用load_dataset读取JSON。全量编码虽然通过，但正式构造Dataset时，Arrow根据前段普通记录推断的schema不能容纳后段新增tools/tool_calls，于是第一版数据准备失败。改为偏移索引后保留了异构原消息，没有删除工具字段来绕过错误。原故障见[第一版准备记录](../experiments/sft_mini_data_inspection_v1/preparation_768_v1/README.md)。

文件流由各进程按需独立打开，反序列化时不携带打开的流，避免共享seek游标。完整派生train/validation在最终验收中实际载入，并用首、中、尾样本与直接编码对照、实际组批；“分析脚本能编码”不能替代这次真实读取验证。

复练普通样本组批时，可运行：

```powershell
@'
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from minimind_lab.data.sft import SFTDataset, SFTCollator

tokenizer = AutoTokenizer.from_pretrained("assets/minimind_tokenizer", local_files_only=True)
dataset = SFTDataset("data/source_samples/sft_sample.jsonl", tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                    collate_fn=SFTCollator(tokenizer, max_length=809))
batch = next(iter(loader))
print("records:", len(dataset))
for key, value in batch.items():
    print(key, tuple(value.shape), value.dtype)
print("real lengths:", batch["attention_mask"].sum(dim=1).tolist())
print("targets:", (batch["labels"][:, 1:] != -100).sum(dim=1).tolist())
'@ | uv run --locked python -
```

这里809仅容纳原始前20条观察样本，不是正式长度。已核对结果为20条，三个张量均为[2,809]，首批真实长度[549,809]、有效助手目标[414,687]。正式数据使用768；不要把练习参数写回已冻结配置。

### 3.8 验证编码边界，而不只验证形状

这一层测试对应四类风险：

| 风险 | 应验证的行为 |
| --- | --- |
| 监督错位 | 角色头不监督，多轮助手区间、结束序列完整，shift只在loss中发生一次 |
| padding污染 | 真实上下文可见，padding不可见且labels=-100 |
| 转换副作用 | 修改适配后的嵌套字段不应改变原记录，普通样本编码保持一致 |
| 边界/零目标 | 源字段包含BOS/EOS字面边界则拒绝；编码器与collator拒绝shift后零有效目标 |

最初的构造探针暴露过两个问题：用户正文嵌入真实助手边界时，纯token搜索会误选用户区域；只有被屏蔽标签的feature也曾被collator接受。当前编码前递归拒绝消息与工具字段中的BOS/EOS字面标记，编码器和collator都检查有效目标。它们是已修复的前置门禁，不是仍然悬而未决的待办。

```powershell
uv run --locked pytest -q tests/test_sft_data.py tests/test_sft_tool_fields.py tests/test_sft_protocol.py
```

测试覆盖已约定的协议和实现范围，不证明被监督的内容正确。下一章才回答哪些完整记录适合作为本轮示范。


## 4. 数据处理与数据配方冻结

数据处理沿用预训练的“固定来源 → 检查与决定 → 派生构造 → 隔离切分 → 独立验收 → 冻结”。SFT增加了角色顺序、工具协议、思考与回答一致性、身份自述及助手监督边界，因此不能只复用预训练的text长度过滤。

### 4.1 三条审核线与处理进度

下面是本轮规则的阅读入口。完整小项编号、方法、产物和持续进度以[数据处理进度表](../experiments/sft_mini_data_inspection_v1/REVIEW_CRITERIA.md#2-数据处理进度表)为准，本文不维护第二套互相竞争的规则。

| 大项 | 小项 | 检查与处理方法 | 本轮完成范围 |
| --- | --- | --- | --- |
| 格式与协议 | 基础结构、角色与字段位置 | validator全量检查；按约定核对消息顺序与tools/tool_calls归属 | 当前字段与协议范围已检查 |
| 格式与协议 | 工具声明、参数、结果对应 | 名称集合、jsonschema、调用/结果块及显式ID；异常整条暂存 | 声明与参数已检查；匿名合并结果仅有块级顺序保证 |
| 格式与协议 | 模板、编码与监督 | 复用实际编码器，拒绝源伪边界与零目标；验证shift、EOS和padding | 最终保留数据已复验，不等于语义正确 |
| 格式与协议 | 长度与有效目标 | 全量真实模板长度统计，768超长整条排除；修订后重编码 | 最终最大768，无截断，所有保留记录有目标 |
| 内容质量 | 正确性、相关性、输入充分性 | 固定抽样、独立计算、定向事实核验，问题落实到具体来源 | 部分完成，不保证百万条事实正确 |
| 内容质量 | 身份、能力与思考一致性 | 区分自述/引用/角色，有限身份修订；其他语境暂存 | 14条身份修订、2条内容修订；限制保留 |
| 内容质量 | 表达、完整性与工具解读 | 看完整上下文，不把格式完整或计算正确当作整题正确 | 有界抽查，长尾质量未全量审核 |
| 集合质量 | 覆盖与组成 | 统计形态、长度、目标数，比较处理前后分布 | 形态已统计，不等于语义领域标注 |
| 集合质量 | 精确重复与高频模板 | 完整对话分组去重；相同回复关联不同用户问题复核 | 精确去重完成，未穷尽语义近重复 |
| 集合质量 | 同题关系、切分与污染 | 归一化问题/对话连通分组，开发样例整组保护 | 本轮跨集重叠已核对，不是全部外部污染审计 |

每项都说明对象、规则、动作和证据。未检查不默认通过；工作进度与单条样本的pass/fail/uncertain不同。格式通过、内容待核实可以同时存在，不把各项混成一个平均质量分。

审核决定只记录来源行/hash、消息位置、理由与动作，不追加成新训练样本。发现问题、确认规则、执行处理、复查新版本是四步，不能让一次机器标记直接等于批量删除。

本轮保留普通、思考与工具文本示范；工具相关记录可以合法地有空assistant正文，不能据此删除。来源许可仍有限制，公开可下载不自动代表所有后续发布用途已澄清。


### 4.2 固定完整来源与抽样

#### 4.2.1 来源身份与抽样理由

文件开头的 20 条记录适合调试，但身份自述内容集中，无法代表全量任务分布。因此接下来先取得完整来源、固定身份，再从全体抽样，而不是据此决定“长度 809 足够”或“数据没有工具调用”。

| 项目 | 本轮实际来源 |
| --- | --- |
| 数据仓库 | `jingyaogong/minimind_dataset` |
| revision | `312afb4f76391145c6902f765bb51691c09a12f5` |
| 文件 | `sft_t2t_mini.jsonl` |
| 字节数 | 1,739,201,170 |
| SHA256 | `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518` |

来源与许可限制见[候选检查总览](../experiments/sft_mini_data_inspection_v1/README.md)。目前不宣称已澄清所有原始语料及模型发布的许可范围。

#### 4.2.2 下载与结构扫描的实现

[download_sft_source.ps1](../implementation/scripts/download_sft_source.ps1) 固定来源版本及预期身份，将字节区间分段下载，检查响应范围与长度，合并后再校验整文件 SHA256。已有完整文件时只核验；不会用同名文件存在代替身份检查。

```powershell
& .\scripts\download_sft_source.ps1 -Connections 8 -ChunkMiB 16
```

本轮从普通下载失败转为分段下载，但没有关闭 TLS 校验或改用不明来源。完整文件保存在 `data/raw/minimind/`，原始大文件不进入 Git。

然后由 [inspect_sft_source.py](../implementation/scripts/inspect_sft_source.py) 按行读原始字节，同步计算 hash，解析 JSON、检查基本结构、统计字段并抽样。它不调用模型、不执行样本中的代码或工具，也不改写、排除记录。

抽样使用独立随机数发生器和蓄水池方法：无需把全部原文一次装入内存，也不干扰未来训练的全局 RNG。全体均匀样本用来观察总体；各形态样例用来确保少数形态也进入检查，二者用途不同。

完整可复跑命令见[来源结构扫描的复跑步骤](../experiments/sft_mini_data_inspection_v1/full_structure_v1/README.md#4-复跑步骤)。复跑要使用新的输出版本，不能覆盖本轮样本和报告，否则后续审核会失去原来的参照。

#### 4.2.3 实际扫描结果与边界

905,718 条记录全部通过当前基本结构校验。按原始字段和标记，互斥分组为：

| 形态 | 记录数 | 解释边界 |
| --- | --- | --- |
| 无思考/工具标记 | 547,017 | 不表示模板不会插入空 think 段 |
| 仅思考标记 | 273,869 | 不表示思考正确或任务更难 |
| 仅工具标记 | 84,832 | 不表示工具协议、调用与结果已正确 |
| 两类标记同时出现 | 0 | 只描述本次标记口径 |

固定 seed=20261029，从全体取 1,000 条均匀样本，并从三个非空分组各取 16 条观察样例；合计 1,048 个不同来源位置已逐条核对原行 hash 和内容。抽样包保存 source_line、原行 SHA256、形态与 record，原文文件留在 Git 忽略目录 `data/processed/sft_candidate_inspection_v1/`。

这些样本尚不是训练/验证切分，也不是最终盲测集。结构报告中的文件身份可以复核“看的是哪份数据”，不能据此证明内容可训练。全量编码长度、语义质量、近似重复和最终隔离仍未检查完成。

### 4.3 从样本预检扩展到全量编码

编码预检的目的不是判断答案真伪，而是确认原始记录经过第3章的实际路径后，能够形成哪些输入与监督。先用固定样本校验入口和统计口径，再扩展全量；最终派生文件仍要重新验收。

#### 4.3.1 独立样本预检入口

字段转换接好后，下一件事不是立刻启动 Dataset 训练，而是把相同编码路径接到独立检查入口。训练前先检查数据，Dataset 负责后续读取，运行时检查再作兜底；不能把首次发现坏样本的时机留到昂贵的训练过程中。

本次先检查固定的 1,000 条随机样本，入口为 [inspect_sft_encoding.py](../implementation/scripts/inspect_sft_encoding.py)。这是局部预检与分析，正式实验前仍要对最终实际使用的派生文件全量核验。预检不新增训练规则、不生成训练集，也不把随机样本当作独立最终评测集。

流程如下：

```text
结构报告中的样本 hash -> 核对当前随机样本文件
  -> 基础 validator
  -> prepare_sft_messages：检查字段转换，观察边界字面标记
  -> encode_sft_conversation：复用 Dataset 调用的编码器
  -> 检查标签对齐，统计 labels[1:] 的有效目标
  -> 逐条诊断 + 汇总长度与监督量
```

为什么复用编码器？如果预检自己实现一套模板与 labels，而 Dataset 另用一套，即使报告正常，也无法说明训练实际读取的是什么。预检把同一编码函数提前用于分析，Dataset 仍可在后续消费记录时使用它；两者职责不冲突。

样本预检区分五种诊断：基础结构错误、字段转换错误、模板/编码错误、潜在消息边界标记、shift后零目标。早期诊断只报告风险，不修改原始文件；后续已将源BOS/EOS字面边界拒绝与零目标拒绝接入实际编码器和collator，见第3.8节。预检报告记录当时的观察，不能替代当前运行路径的门禁。

输入身份与 tokenizer 文件在检查前后核对；脚本记录自身、编码器、validator 的 hash。输出目录拒绝覆盖：有标记则保存报告并以 2 退出，无标记以 0 退出。输入身份不符或编码器返回不等长 input_ids/labels 等程序契约错误直接中止，不能为了跑完统计而当普通坏数据忽略。

在 implementation 目录执行：

```powershell
uv run --locked python scripts/inspect_sft_encoding.py `
  --source-report ../experiments/sft_mini_data_inspection_v1/full_structure_v1/structure_report.json `
  --samples data/processed/sft_candidate_inspection_v1/random_sample.jsonl `
  --sample-key random `
  --report-dir ../experiments/sft_mini_data_inspection_v1/encoding_preflight_v1
```

目录已存在时应换新的输出版本，不覆盖旧报告。报告不复制完整原文：report.json 保存身份、总体与分组统计；records.jsonl 保存源行、hash、形态、长度、目标数量和标记。这样能从统计回到具体样本，又不会把审核报告当作一份新增训练集。

本次 1,000 条全部成功编码，未发现上述诊断标记；其中 plain 599、reasoning_only 311、tool_only 90。当前规则下，完整模板长度 P50=499、P95=779.1、P99=983.04、最长 1,662。这里保留完整思考和工具上下文，不截断、不padding。此处统计先于正式长度选择；最终768策略在第4.4节确定。分位数采用线性插值，所以可能有小数；这些只描述样本，不是全量分布。

有效助手目标合计 420,806，所有可预测位置合计 499,121，比例约 84.31%。分母为各记录长度减一后的总和，与 causal shift 口径一致。低目标比例不直接表示质量低，例如工具声明与结果会占输入上下文，但当前不作为预测目标。

检查结果、复跑方法与限制见[随机样本预检报告](../experiments/sft_mini_data_inspection_v1/encoding_preflight_v1/README.md)，脚本验证见 [test_sft_encoding_inspection.py](../implementation/tests/test_sft_encoding_inspection.py)。内容质量与完整工具协议仍须另查，1,000 条无标记不意味着全部合格或可以冻结；本次也没有确定 768、1,024 或其他正式长度。

#### 4.3.2 流式全量编码与明细核验

随机样本的作用是验证入口并估计分布，不是将其他记录默认为合格。随后将诊断扩展到整个 `sft_t2t_mini.jsonl`：905,718 行、1,739,201,170 字节。这个阶段检查完整候选的可编码性，不表示这批原始数据已被全部选作训练输入；最终派生数据生成后仍需重新验收。

先对 1,000 条固定随机样本测量编码速度，预热后单进程约每秒 1,255 条，直接外推约 12 分钟。这只是编码短段，不包含完整诊断、启动和读写成本，也没有假定全量分布与样本完全一致。随后采用 4 个 CPU worker 分块处理，不加载模型或使用 GPU。

新增入口是 [inspect_sft_full_encoding.py](../implementation/scripts/inspect_sft_full_encoding.py)。它复用同一编码器及已有边界/分位数函数；原样本入口和旧报告保持不变，便于核对两个阶段的身份和结果。全量调度有三个需要说明的工程点：

1. **流式读取**：原始字节逐行读取，每行计算 SHA256，读取过程中再次累积整文件 hash，不将 1.74GB 原文全部装入内存。
2. **有限并行**：每块 1,000 行，在途任务不超过 worker 数的两倍；按提交顺序收集结果，使明细的行序与来源一致。Windows 使用 spawn，worker 各自加载 tokenizer，关闭 tokenizer 内部并行以避免线程竞争。
3. **紧凑汇总**：逐条明细及时写盘，内存只保存紧凑的长度/目标数组、计数与少量最长或异常索引，不堆积百万条原文和诊断字典。

在 implementation 目录执行：

```powershell
uv run --locked python scripts/inspect_sft_full_encoding.py `
  --source data/raw/minimind/sft_t2t_mini.jsonl `
  --source-report ../experiments/sft_mini_data_inspection_v1/full_structure_v1/structure_report.json `
  --report-dir ../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1 `
  --details-dir data/audit/sft_full_encoding_v1 `
  --workers 4 --chunk-size 1000
```

两个输出目录都要求不存在，复跑时一起更换版本。来源大小/hash 在启动前先校验；结束前再核对实际读取的 hash、总行数、tokenizer、编码器、检查代码和来源报告身份。若任一身份变化，不生成成功结论。

产物分开存放：实验目录保存 `run_manifest.json` 与小型 `report.json`；大的 `records.jsonl`、`issues.jsonl` 放在被 Git 忽略的 data/audit 目录。前者每个源行一条，后者保存全部触发诊断的记录，均不保存完整原文或 token IDs。汇总报告记录这些明细的路径、字节数和 hash。目录分开是为了避免把约 200MB 诊断明细随小型报告直接推送 Git，不是把明细丢弃。

运行状态与检查结果要分开理解：`run_manifest.status=completed` 表示这次诊断已完成；若找到问题，报告仍会保存、退出码为 2，不能将其误认为程序未跑完。启动前身份错误会拒绝创建输出，运行中捕获到错误才将 manifest 改为 failed；强杀或断电仍可能留下 running，部分输出不能充当全量验收结果。当前处理行序也只是诊断追溯顺序，不等于已经冻结训练时的 sampler 顺序。

入口的定向测试覆盖源行身份、异常类别、shift 口径、汇总、多进程顺序、坏行报告、拒绝覆盖和身份不匹配。另以固定 1,000 条样本逐条对照全量入口的记录级检查，形态、诊断、长度和目标数与原样本报告一致。这样核对扩展前后有没有改变检查口径，而不是仅凭“新脚本也能跑”宣布一致。

最终状态、整体/分组结果、候选长度超限条数及验收范围统一见[完整候选预检报告](../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1/README.md)。本次不截断、不 padding、不筛选、不改身份，也不决定正式长度；即使全量都能编码，也不能取代内容与完整工具协议审核。

本次实际完成 905,718 条，全部成功编码，当前诊断项为 0；CPU 运行约 279 秒。全量长度 P50=495、P95=776、P99=919、最长 3,216，有效助手目标共 377,330,100。超过 768、1,024、1,536、2,048 的记录分别为 49,187、3,395、531、73 条，只表示超限数量，不是已执行的排除或截断策略。

结束后独立同时遍历原始文件与明细，逐条核对全部源行号和 hash，确认 905,718 行无缺失、重排或额外行；重新累计的长度、目标、形态与超限数量和报告一致。原随机样本的 1,000 条结果与全量输出逐条相同。新的 4 项定向测试通过，全项目 406 项通过。完整逐条明细约 172.64 MiB，放在忽略目录；Git 中保留汇总和路径/hash，不把大明细误当小型报告提交。

### 4.4 长度选择与超限处理

有了实际token长度，先分析不同处理方式的损失，再确定策略；不是先选一个方便的数字，再为它寻找理由。

#### 4.4.1 比较整条排除与截断的影响

有了全量长度，仍不能只看 P99 就确定训练上限。下一步要问：如果处理超限记录，会失去多少监督、改变哪些样本类型、是否切断助手回复？本次先比较两种简单候选，不改变实际数据：整条不纳入，或保留编码的前 N 个位置。

分析入口是 [analyze_sft_length_policies.py](../implementation/scripts/analyze_sft_length_policies.py)。它核对来源、全量诊断明细、编码代码与 tokenizer 身份，从全量明细统计整条排除影响，再重新编码超过最小候选 768 的 49,187 条，查看 labels 中每个目标的位置。这样不会假设尾部 token 全都参与监督，也不必把未受影响的其他记录全部再编码一次。

在 implementation 目录运行，已有目录拒绝覆盖，复跑时换新的报告版本：

```powershell
uv run --locked python scripts/analyze_sft_length_policies.py `
  --source data/raw/minimind/sft_t2t_mini.jsonl `
  --encoding-report ../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1/report.json `
  --report-dir ../experiments/sft_mini_data_inspection_v1/length_policy_analysis_v1 `
  --limits 768 1024 1536 2048
```

整条排除的模拟很直接：长度超过上限的记录不计入保留集合，它所有助手目标也随之丢失。注意两个分母不同：排除条数除以 905,718，排除目标数除以 377,330,100。实际影响如下：

| 上限 | 整条排除记录数 | 记录比例 | 丢失助手目标比例 |
| --- | --- | --- | --- |
| 768 | 49,187 | 5.431% | 9.800% |
| 1,024 | 3,395 | 0.375% | 1.037% |
| 1,536 | 531 | 0.059% | 0.236% |
| 2,048 | 73 | 0.0081% | 0.0426% |

这说明长样本数量较少，不代表其监督量同样少。过滤也会改变组成：768 下，思考标记组自身约 11.72% 的记录、17.13% 的目标被排除，明显不同于普通标记组约 2.80% 的记录。这里统计形态，不直接判断知识价值或质量。

前缀切片则模拟 `input_ids[:limit]` 与 `labels[:limit]`，按 `labels[1:limit]` 计算保留目标，不添加 EOS。重点不是“能不能喂进模型”，而是被保留的示范是否仍符合目标。以 1,024 为例，切掉的尾部目标约 82.86 万，比整条排除少；但 3,395 个截止点中，3,319 个位于连续助手监督区间内且没有保留当前回复 EOS，21 个只少了 EOS 后的换行，55 个位于监督区间之外。最后一类也可能丢掉后续回复或工具轨迹，不能直接视为安全。

检查函数 `truncation_effect` 将这三类分别记录，并检查切片后有没有有效目标。源数据原本全都有目标，但 1,024 下有 2 条切片后目标变成 0，768 下有 7 条；这说明长度或轮次处理之后仍必须再次检查，而不是沿用处理前的通过标记。EOS 已保留、仅少换行，与尚未生成 EOS 的不完整助手段也必须分开，避免误判。

本次分析仅比较整条排除与前缀切片，没有实现完整轮次裁切、分块或 packing，也不认定截断一定使训练失败。分析时曾将 1,024 加整条排除列为优先讨论候选，不是已接受配方；随后用户选择沿用官方默认长度 768，并确认超限整条排除，具体执行见下一节。固定 padding 开销、模型约束、显存与吞吐仍需检查。

结果、形态分组影响、零目标源行与验证范围见[长度处理分析报告](../experiments/sft_mini_data_inspection_v1/length_policy_analysis_v1/README.md)。8 项定向测试通过，全项目 414 项通过；本次没有产生处理后的训练数据。以上都基于当前未完成内容筛选、去重和身份修订的候选，后续数据版本变化时要重新统计。

#### 4.4.2 构造768长度候选

影响分析之后，才将讨论变成处理规则。本轮用户确认 `max_length=768`，超过 768 的完整记录整条排除，不截断、不补 EOS。采用官方默认长度是本次选择，不表示不同训练阶段的长度必须相同，也不表示我们的超长策略与官方截断实现相同。

这一阶段只回答“哪些完整记录符合当前编码长度”，不提前回答“哪些内容值得训练”。因此产物命名为 `length_eligible.jsonl`，而不是 `train.jsonl`；超长记录保存在 `overlength_excluded.jsonl`，便于复查，而不是删除。

**实现思路。** [prepare_sft_length_subset.py](../implementation/scripts/prepare_sft_length_subset.py)同时读取原始 JSONL 与全量编码明细：

1. 先核对当前编码路径、tokenizer 与报告身份相同，避免用旧明细筛选新编码规则。
2. 每次读取一条原始记录及其对应明细，检查来源行号与行 hash，再判断 `input_tokens <= 768`。
3. 将原始行字节写入对应临时文件。保留每条完整对话和各输出中的原有相对顺序，不改名字、思考或工具字段。
4. 全部输入的 hash、数量一致后，改为最终文件名，再重新读取输出核对 hash，保存小型报告。失败运行不能仅凭文件存在就算完成。

这里没有重新编码所有记录，因为全量预检已经提供了固定编码版本的逐条长度；复用前提是来源、明细和编码路径全部吻合。后续内容修订会改变输入，届时需要重新编码验收。

在 `implementation` 目录运行以下 PowerShell 命令。本次目录已存在，复练时更换两个输出目录的版本名，不覆盖旧产物：

```powershell
uv run --locked python scripts/prepare_sft_length_subset.py `
  --source data/raw/minimind/sft_t2t_mini.jsonl `
  --encoding-report ../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1/report.json `
  --output-dir data/processed/sft_length_768_v1 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/length_selection_768_v1 `
  --max-length 768
```

实际保留 856,531 条、340,351,475 个有效助手目标；另存超长 49,187 条、36,978,625 个目标。记录、目标与字节数相加均与原始来源一致，与前一节的模拟结果一致。原始数据和保留记录内容未改变。具体身份与边界见[长度筛选报告](../experiments/sft_mini_data_inspection_v1/length_selection_768_v1/README.md)。

**为什么要测试。** [test_sft_length_subset.py](../implementation/tests/test_sft_length_subset.py)用 767、768、769 三种长度验证包含边界，检查写出字节和相对顺序不变；再故意提供错误 hash、错位明细或已有目录，验证入口不会悄悄接受错误输入或覆盖产物。4 项定向测试通过。

### 4.5 精确去重与来源追溯

长度处理和去重的记录可能重叠，不能各自从原始总数简单相减。先定义重复键，再在实际版本上执行并核对映射。

#### 4.5.1 定义重复键并统计规模

长度合格不代表集合组织合格。若重复对话被独立随机切分，可能同时出现在训练集和验证集，验证就不再能可靠衡量对未见对话的拟合情况。因此在正式切分前，先明确重复的定义和规模。

本次先扫描完整原始来源，不是直接对长度候选执行去重。[inspect_sft_duplicates.py](../implementation/scripts/inspect_sft_duplicates.py)逐条校验后，将 `conversations` 按字典键排序并序列化为紧凑 JSON，再计算 hash 作为分组键。它忽略外层排版和顶层元数据，但保留消息顺序、所有消息字段和正文空白；相同问题的不同回答不会因此合并。

扫描只累计每个键的第一条来源行号与次数，并保存少量行号配对供定位，同时核对原始文件大小、数量与 hash。不做近似匹配，也不自动认定某条内容更好。结果为：

| 项目 | 结果 |
| --- | ---: |
| 来源记录 | 905,718 |
| 不同完整对话 | 858,856 |
| 至少出现两次的组 | 8,675 |
| 重复组包含记录数 | 55,537 |
| 若每组留一条，多余记录数 | 46,862 |

运行命令、例子行号与范围见[精确重复报告](../experiments/sft_mini_data_inspection_v1/exact_duplicates_v1/README.md)。4 项定向测试覆盖分组身份、计数及输入错误拒绝。检查没有删除重复记录，也未处理同题多回答、近似重复或同源文档的隔离。

两项处理不能各减各的：某条重复记录也可能超长。后续应在长度候选及内容修订后的实际版本上核算重复，再确定保留规则与分组切分。当前 856,531 是长度候选数量，不是最终训练数量。

#### 4.5.2 保留首条并保存lineage

上一节统计的是原始来源，不能直接拿原始重复数去减长度候选。这一步实际处理 `length_eligible.jsonl` 的 856,531 条记录，采用同一完整对话键，每组保留原始来源中最早出现的一条。保留首条是确定性规则，不是质量排序；同题不同回答、不同思考或工具字段不合并。

**实现与上游的衔接。** 新入口 [deduplicate_sft_subset.py](../implementation/scripts/deduplicate_sft_subset.py)从长度报告取得实际输入文件及 hash，再从全量编码明细依次取出长度合格行。每条候选与明细的原始行 hash 对上之后，才能继续处理。这样，即使中间排除了超长记录，仍能准确定位到原始来源，而不是误用筛选后的行号。

核心分流可以理解为：

```python
key = conversation_sha256(record["conversations"])
if key not in first_seen:
    # 保存首次出现的位置，原样写入保留候选。
    ...
else:
    # 原样另存重复项，并记录它对应哪条保留记录。
    ...
```

具体实现还会为每条输入写 `lineage.jsonl`：输入行号、原始来源行号及 hash、对话键、处理动作、保留输出行号和保留来源行号。训练用 JSONL 不混入这些审核字段，对应表独立保存。与前几周一样，审核或追溯记录不是新的训练样本。

在 `implementation` 目录执行以下 PowerShell 命令；目录已存在时更换版本名，不能覆盖：

```powershell
uv run --locked python scripts/deduplicate_sft_subset.py `
  --length-report ../experiments/sft_mini_data_inspection_v1/length_selection_768_v1/report.json `
  --encoding-report ../experiments/sft_mini_data_inspection_v1/full_encoding_preflight_v1/report.json `
  --output-dir data/processed/sft_dedup_768_v1 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/deduplication_768_v1
```

实际保留 **809,713 条**，另存精确重复 **46,818 条**；两者合计 856,531。保留的有效助手目标为 334,943,065，重复项为 5,408,410，两者合计也与输入一致。没有修改消息或打乱顺序，没有应用内容决定或身份修改。产物身份与边界见[候选精确去重报告](../experiments/sft_mini_data_inspection_v1/deduplication_768_v1/README.md)。

**验证的理由。** 计数相加正确，仍可能把错误的一条记录保留下来或写错来源。因此除 6 项定向测试外，助手又在独立只读过程中遍历完整原始来源，逐条核对长度候选、对应表、保留/重复输出的原始字节和首条分组位置，核验全部输入输出 hash。856,531 条映射全部通过；全项目 428 项通过（69.89 秒）。

在这一去重版本中，原始来源 731,011 的缺代码却编造输出记录、776,360 的未声明工具调用记录，仍分别位于候选第 678,855、721,544 行。它们不因唯一、长度合格就变成好的示范；用户随后确认并执行了原样排除，见第4.6.2节。精确去重也未解决同题多回答的分组隔离，不能据此随机切分并冻结。

### 4.6 工具协议与已确认排除

第3章解决表示兼容，这里核对可作为示范的协议条件。先发现和定位，再按已确认规则分流，最后对实际保留版本重新扫描。

#### 4.6.1 核对调用是否在声明集合内

去重后的记录可能仍然是错误示范。此前来源行 776,360 能解析、能编码，回答中的温度换算也正确，但调用了未声明的天气工具。因此把这个问题拆成一个可自动核查的条件：结构化 `tool_calls` 的函数名，必须属于该对话的声明工具集合。

[audit_sft_tool_declarations.py](../implementation/scripts/audit_sft_tool_declarations.py)先经 [sft_candidate_lineage.py](../implementation/scripts/sft_candidate_lineage.py)验证实际候选与来源对应表，再复用 `prepare_sft_messages` 解析字段。对于每个助手调用，用名字精确查询声明集合；重复声明名和字段适配错误另行标记。它只报告问题，不补造工具、不执行调用、不删除样本。

这一检查连接的是“数据表示正确”和“协议条件一致”，不是所有内容质量。普通正文提到工具名不算结构化调用；助手发现缺少工具并直接澄清，也不能因此判错。合法工具动作的空 content 更不是删除依据。

在 `implementation` 目录使用 PowerShell 运行；已存在目录须改版本名：

```powershell
uv run --locked python scripts/audit_sft_tool_declarations.py `
  --dedup-report ../experiments/sft_mini_data_inspection_v1/deduplication_768_v1/report.json `
  --details-dir data/audit/sft_tool_declarations_768_v1 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/tool_declaration_audit_768_v1
```

实际检查全部 809,713 条，其中 47,584 条含非空工具声明、调用或 tool 消息，结构化调用共 50,820 次。发现 **1,199 条**含未声明函数调用；适配错误和重复声明名均为 0。完整问题定位保存在忽略目录，Git 中的小型报告保存输入身份、计数和少量例子，见[工具声明报告](../experiments/sft_mini_data_inspection_v1/tool_declaration_audit_768_v1/README.md)。

6 项定向测试分别保证可用工具正常通过、未声明及无声明调用被定位、纯文本提及不被误判、空助手工具动作合法、重复声明被单列以及坏参数不被静默忽略。独立复核又直接解析全部原始消息字段重新计算名称差集，与问题清单逐条一致。

这个名称扫描本身不检查完整参数JSON Schema、调用与结果ID对应、工具结果真实性或最终回答质量，不能把无标记记录写成完整协议通过。参数与块闭合由第4.6.4节的门禁补充，内容质量另行审核。扫描标记先作为处理依据，用户确认规则后才在第4.6.3节执行。

#### 4.6.2 应用两条具体审核决定

与全量规则检查并行，用户明确确认排除已讨论的两条原样记录：731,011 的缺代码却编造执行结果，以及 776,360 的未声明工具调用。这是针对两条完整记录的决定，不推广为关键词过滤。

先将确认写入独立的 [decisions.confirmed.jsonl](../experiments/sft_mini_data_inspection_v1/calibration_v1/decisions.confirmed.jsonl)，包括原始文件 hash、来源行号、原行 hash、动作和理由；旧助手建议保留历史状态。再由 [apply_sft_review_decisions.py](../implementation/scripts/apply_sft_review_decisions.py)执行，拒绝未确认、不同来源、内容变化、重复或未命中的决定。

这一层把人工判断与机械执行分开：人决定记录为何排除，脚本只负责精确命中并形成新版本。审核 JSONL 不是新训练样本，不能追加进对话数据。执行时仍逐条核对候选和来源对应表，分别写保留、排除文件及新的来源对应表，不改写剩余记录。

在 `implementation` 目录运行，复练时更换输出版本名：

```powershell
uv run --locked python scripts/apply_sft_review_decisions.py `
  --dedup-report ../experiments/sft_mini_data_inspection_v1/deduplication_768_v1/report.json `
  --decisions ../experiments/sft_mini_data_inspection_v1/calibration_v1/decisions.confirmed.jsonl `
  --output-dir data/processed/sft_reviewed_768_v1 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/confirmed_review_768_v1
```

实际精确排除 2 条，剩余 **809,711 条**。独立逐条比较全部输入、输出和新对应表，确认没有扩大排除、重排或改变内容，hash 均一致。8 项决定应用测试与上节 6 项测试合计新增 14 项；全项目 442 项通过（77.66 秒）。具体产物见[确认决定应用报告](../experiments/sft_mini_data_inspection_v1/confirmed_review_768_v1/README.md)。

原 1,199 条工具标记只有 776,360 被本次具体决定排除，因此该阶段候选仍有 **1,198 条**待处理。随后用户确认统一规则，执行见下一节。记录数不是最终训练数量，剩余记录也没有因此被全部接受。

#### 4.6.3 批量隔离未声明调用

上一节只排除了两条具体记录，新增机器标记还不是可执行决定。用户随后确认：当前候选中未声明结构化函数调用的记录整条隔离，不补造声明、不删除其他工具数据。于是先将类别、动作、原始来源和固定输入报告 hash 写入[规则文件](../experiments/sft_mini_data_inspection_v1/review_rules/01_UNDECLARED_FUNCTION_CALLS.json)，再执行。

**与单条审核的区别。** 单条确认指出某条为何排除；规则确认说明一类可精确检查的问题如何统一处理。机器标记、已接受规则与应用报告是三种产物，不能互相替代。本次不将其他未审核标记自动升级为排除。

[apply_sft_tool_quarantine.py](../implementation/scripts/apply_sft_tool_quarantine.py)核对固定输入，从清单选择授权类别；来源行 776,360 已在上一步排除，核对 hash 后不重复处理。其余 1,198 条转成精确来源行/hash 决定，复用已有执行器分流，再对全部剩余候选重跑检查。这样验证的是实际问题已消失，不只是少了预计条数。

在 `implementation` 目录用 PowerShell 运行；目录已存在时更换输出版本名：

```powershell
uv run --locked python scripts/apply_sft_tool_quarantine.py `
  --candidate-report ../experiments/sft_mini_data_inspection_v1/confirmed_review_768_v1/report.json `
  --audit-report ../experiments/sft_mini_data_inspection_v1/tool_declaration_audit_768_v1/report.json `
  --rule ../experiments/sft_mini_data_inspection_v1/review_rules/01_UNDECLARED_FUNCTION_CALLS.json `
  --output-dir data/processed/sft_tool_quarantine_768_v1 `
  --report-dir ../experiments/sft_mini_data_inspection_v1/tool_quarantine_768_v1
```

实际从 809,711 条隔离 1,198 条，保留 **808,513 条**。新候选仍有 46,385 条工具相关记录、48,961 次结构化调用，重查标记为 0；未覆盖的参数/结果协议和内容质量没有自动通过。原始数据、更早版本与隔离记录均保留，来源对应表和命中清单独立保存，见[规则隔离报告](../experiments/sft_mini_data_inspection_v1/tool_quarantine_768_v1/README.md)。

6 项新测试针对授权范围、版本身份与避免重复排除；独立复核另逐条比较输入、输出和对应表，直接解析原始工具字段，确认排除项命中固定清单、保留项无该问题，数量、字节、顺序和 hash 均一致。全项目 448 项通过（80.97 秒）。

工具声明检查解决的是协议可用性，不证明回答内容正确。接下来将完整上下文纳入内容审核，再按确认规则处理身份自述及其他质量问题。

#### 4.6.4 参数Schema、块闭合与监督门禁

[sft_protocol.py](../implementation/src/minimind_lab/data/sft_protocol.py)消费 prepare_sft_messages 转换后的消息与声明，检查顺序、声明重名、调用名称、参数 Schema 以及调用/结果块是否闭合。Schema 使用 jsonschema 库，不自己手写一套类型验证器；禁止外部引用和联网解析。该函数返回硬问题和有限保证提示，编码器复用同一检查，避免只在分析脚本里检查。

源数据的匿名 tool 消息可以合并多次结果。可以检查它出现在调用之后、后面有助手继续回答，但没有 ID 或可解析的结果结构时，不能证明每项结果都与调用语义匹配。显式 ID 的错配会拒绝，匿名合并保留限制说明，而不是补造 ID 让数据“看起来完整”。

此前 build_assistant_labels 依赖 token 边界查找，若用户正文嵌入真实角色标记，就可能把正文当作助手区域。当前编码器先递归检查源消息与工具字段，发现 BOS/EOS 字面边界即拒绝，不偷偷转义或选中用户正文。编码器与 collator 也检查 shift 后至少有一个有效目标。原来的标签算法仍服务于已经过这些门禁的模板输出，不能脱离前置条件理解它的安全范围。

### 4.7 内容审核、身份处理与精确修订

格式和协议通过之后，才判断“这是不是值得模仿的回答”。检查对象是用户请求、思考、工具结果与最终回答构成的完整上下文，不能只读助手末句。这里按校准标准、组织审核、落实决定三个层次说明；逐条证据保留在[内容复核记录](../experiments/sft_mini_data_inspection_v1/quality_review_v1/README.md)，不将抽样发现率外推为全量准确率。

#### 4.7.1 用典型案例校准标准

| 来源行 | 判断依据 | 对处理规则的启示 |
| --- | --- | --- |
| 99,806 | 集合运算可独立核算，当前回答成立 | 有证据支持时保留；不能因为表达像模板就删除 |
| 731,011 | 用户没有提供代码，助手却另造程序并报告输出 | 输入不足仍编造结果属于内容错误，已确认排除原样 |
| 780,028 | 虽有工具声明，但缺少待处理文本，助手先请求补充 | 不调用工具、要求澄清可能才是正确示范 |
| 693,987 | 将撒哈拉沙漠错误归于印度尼西亚 | 流畅且结构合法也可能事实错误，需要定向核查 |
| 776,360 | 温度换算正确，但调用了未声明工具 | 正文正确与协议正确是两个独立维度 |

这一步不是去重，而是在形成一致的判法：缺输入不能补造输入；工具结果不能支持超出其范围的结论；正确答案也不能掩盖错误解释。校准记录与原文位置见[审核记录](../experiments/sft_mini_data_inspection_v1/calibration_v1/README.md)。

#### 4.7.2 将身份表达放回任务上下文

用户要求相关自述使用训练者yanpeng，并说明MiniMind的上游贡献。执行前仍要区分自我介绍、客观项目介绍、引用和角色扮演：把所有名字全局替换，可能破坏本来正确的任务。原始来源与许可证不改，派生数据的变更必须可追溯。

本轮先用关键词定位，再按高频助手表达分组，回看不同用户问题与思考。它们分别由 `inspect_sft_identity.py`、`group_sft_identity_templates.py` 等诊断入口及其报告组织；检索命中只扩大审核范围，不自动判错。例如，同一句自我介绍可以用于回答“你是谁”，却不能拿来回答“AI有哪些算法”；角色扮演中的人名也不是训练者身份。

最终身份改写限定为：只有一组user/assistant、用户是明确身份问句、回答属于已核对的固定介绍。符合条件的14条改写为真实训练来源，并清空不再适配的旧思考；不是给所有含名字的记录换名。其他未核实语境暂存，包含可能本来合理的引用，不能把暂存数量当成错误总数。

#### 4.7.3 组织辅助审核与固定抽查

高频模式能减少重复阅读，但仍需核对完整语境。本轮让辅助Agent对85条记录提出建议，再由主审核检查；发现其曾误读用户已经提出的请求，因此建议不直接变成批量修改规则。辅助审核是定位和草拟意见，不是新的质量真值。

随后按三种记录形态各取8条，共固定24条进行内容复核，检查相关性、事实/推理、工具解读和思考一致性。发现的9条问题最终处理为2条内容修订、7条暂存。样本固定且分层，适合回看判法与变更，不代表对全量语料作了随机准确率估计。

#### 4.7.4 绑定原文身份后修订与复验

| 问题类型 | 具体观察 | 本轮动作 |
| --- | --- | --- |
| 自述与上下文矛盾 | 10,723虚构吃早餐的经历，思考又虚构系统规则 | 整条暂存，只换名字不能修复 |
| 正确答案伴随错误解释 | 624,568选对音乐社团，但解释混乱 | 修订解释，清空旧思考 |
| 结论超出证据 | 836,051工具只给当前天气，助手推断降雨概率 | 只修最后答复，说明缺少降雨预报，工具上下文不变 |

用户确认音乐题修订保留，并委托助手完成剩余收尾；天气题修订属于受托执行，不写成用户逐字审核。具体替换文本见[最终修订](../experiments/sft_mini_data_inspection_v1/quality_review_v1/README.md#5-典型判法与最终修订)。

策略保存来源行、原行hash、消息索引、原助手消息hash及替换文本。`prepare_sft_recipe.py`先定位再核对身份，复制记录后只替换指定消息：

```python
if digest(raw) != revision["source_line_sha256"]:
    raise ValueError("reviewed source bytes changed")
message = record["conversations"][revision["assistant_message_index"]]
# 再核对消息角色及hash，复制记录并替换目标消息。
```

行号负责定位，hash确认内容没换版；指定修订没有命中必须失败，不能悄悄漏过。修订是在原记录上替换，不把审核表或新答案追加为第二条样本。文本改变后重新编码、核对长度与有效目标、去重，然后才按组切分。第三版从固定的808,513条输入重新构造，完整命令见[准备命令](../experiments/sft_mini_data_inspection_v1/preparation_768_v3/README.md#52-准备命令)。

### 4.8 批量准备与分组切分

前面的检查有了处理依据后，才把规则合并成完整构造流程。运行入口不能一边训练一边临时决定哪些数据该排除。

#### 4.8.1 串联处理与派生文件

前面的步骤分别产生检查证据。这一步由助手按用户委托把它们串成一次完整数据准备：固定输入、执行有界策略、生成文件、分组切分、独立验收，并把约定写成合同。它仍不负责启动训练，更不把没有核实的事实标成正确。

```text
808,513 条固定候选及原始来源对应表
  -> 基础字段、消息顺序、工具声明与参数检查
  -> 有限身份与内容修订 / 问题与不确定语境整条暂存
  -> 同一运行时编码器：边界、长度、有效目标门禁
  -> 修订后精确去重
  -> 同题归一化连通分组 + 开发样例保护
  -> train / validation + 编码索引 + 全部处理动作
  -> 独立核验、实际 Dataset 与 collator
  -> 数据合同候选；用户验收后才讨论最终冻结
```

处理入口为 [prepare_sft_recipe.py](../implementation/scripts/prepare_sft_recipe.py)，当前接受规则集中在[processing_policy.v3.json](../experiments/sft_mini_v1/data_recipe/processing_policy.v3.json)，不散落在聊天记录里。输入候选报告、策略、分组报告、开发样例、tokenizer、源码和依赖锁文件都记录身份；大输出先写 `.partial`，完整处理后再形成正式文件，报告完成也不等于合同获批。

#### 4.8.2 按完整连通组隔离train与validation

按任一归一化 user 正文、或归一化完整对话建立连通组，再以固定 seed 对完整组分配验证预算。归一化只用于分组，不改原文；开发样例所在组强制训练。共享简单问题可能连成大组，这是防止泄漏的成本，需要在报告里观察最大组和实际验证组成，不能声称分组后仍是记录级等概率抽样。

本轮形成659,187个连通组，最大组30,195条；验证2,048条来自1,868组，开发审核涉及的1,197个来源行受训练侧保护。实际组成要与预算一起检查：按组分配不保证恰好获得任意数量或完全相同比例的每种任务。这里的隔离只覆盖已定义的归一化关系，不代表消除了全部语义近重复。

### 4.9 独立验收实际文件与读取链路

[verify_sft_recipe.py](../implementation/scripts/verify_sft_recipe.py)重新核对全部文件 hash、字节数和行数，再逐条对照输入与动作 ledger，验证保留未变、修订只发生在允许的助手字段、暂存没有丢失原文。它还独立检查修订后精确重复、归一化问题/对话跨集重叠、开发样例隔离，重新编码全部验证集，并实际加载两份 SFTDataset，与原 JSONL 编码对照后组批。

为什么必须从实际入口读一次？第3.7节的Arrow故障已经表明，分析脚本能编码不代表Dataset能载入全量异构记录。最终验收必须使用当前真实读取实现，而不是只重复验证诊断脚本。代码更新或修订策略改变后，重新验收新版本，不拿旧报告替新文件背书。

合同按[人类可读说明](../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.md)再到JSON的顺序阅读：先看输入与规则，再看实际train/validation文件身份、tokenizer、监督与padding，最后看接受限制。当前完整准备命令、产物地图与复验范围见[第三版报告](../experiments/sft_mini_data_inspection_v1/preparation_768_v3/README.md)。

准备任务的completed只说明程序完成，不能自动授予数据接受或训练执行权限。先核对实际产物和限制，再进入下一节冻结；模型起点、batch、学习率与时长则留给训练配方。

### 4.10 数据合同与最终冻结

处理完成后先验收、接受，再冻结。数据合同描述这一版记录及编码语义，不负责冻结学习率、batch或模型参数。

#### 4.10.1 规则、合同、接受与冻结标记

数据配方不是单独一份 JSON，而是一组分工明确、互相绑定的文件：

| 文件 | 要回答的问题 |
| --- | --- |
| processing_policy.v3.json | 用什么筛选、修订、暂存、分组与切分规则？ |
| DATA_CONTRACT.candidate.v3.json | 规则实际产生了哪些文件、数量、hash 和编码约定？ |
| acceptance.json | 本轮接受什么限制，由谁确认或受托执行，是否授权训练？ |
| DATA_CONTRACT.json | 最终接受的这一版数据与编码身份是什么？ |
| verification.frozen.json | 最终合同对应的数据与实际读取是否复验通过？ |
| data_freeze.json | 正式合同、接受记录、规则和复验证据是否已经完整提交？ |

正式合同还绑定 tokenizer 全部资产、运行源码和依赖身份。因此，文件名不变但正文、模板或监督规则被改动，也会在身份检查中被拦下。切分 seed 属于此处的数据规则，未来训练 sampler 的 seed 属于训练配方，二者不能混同。

#### 4.10.2 生成冻结并执行只读检查

接受记录与候选精确对应后，在 implementation 目录执行：

```powershell
uv run --locked python scripts/freeze_sft_data_recipe.py `
  --candidate ../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.candidate.v3.json `
  --acceptance ../experiments/sft_mini_v1/data_recipe/acceptance.json `
  --recipe-dir ../experiments/sft_mini_v1/data_recipe `
  --verification-output ../experiments/sft_mini_data_inspection_v1/preparation_768_v3/verification.frozen.json
```

冻结器先生成 pending 中的最终合同内容，独立验证器检查这些最终字节及对应数据；通过后原字节重命名为正式合同，最后才写 data_freeze.json。因此 receipt 的 hash 对应的是最终合同，不是批准前的另一个候选。已有正式输出不能覆盖；失败留下 pending 不代表已经冻结。

冻结后的常规复查不需要重新生成数据：

```powershell
uv run --locked python scripts/freeze_sft_data_recipe.py `
  --check ../experiments/sft_mini_v1/data_recipe/data_freeze.json
```

这个命令只检查合同、证据、数据、代码、tokenizer 资产和包版本是否一致，不重新清洗或训练。当前SFT准备和正式启动检查均已接入这条数据身份核验；它只证明版本一致，不重新判定内容质量。

#### 4.10.3 最终数量与接受限制

| 本轮结果 | 实际观察 |
| --- | --- |
| 保留与切分 | 共 787,425 条，训练 785,377、验证 2,048 |
| 修订 | 身份 14 条、内容 2 条；另存修订后重复 15 条 |
| 暂存 | 21,073 条，其中具体内容问题 7 条，其他为协议或名称语境等 |
| 有效助手目标 | 训练 326,249,045、验证 833,052；最大长度 768，无截断 |
| 隔离复验 | 完整对话精确重复、归一化问题/对话跨集重叠、开发样例进入验证均为 0 |
| 真实读取 | 两份完整 Dataset 可载入，首/中/尾编码对照通过，组批得到 [3,768] |

两条内容修订没有改变验证文件字节，但独立复验仍重新编码了全部2,048条验证，并核对修订后的训练数据与真实读取。验证文件未变只能说明文件身份稳定，不能替代新处理代码和派生训练集的验收。

最终[数据配方](../experiments/sft_mini_v1/data_recipe/DATA_CONTRACT.md)接受的是有限语义审核下的教学版本：事实与推理未全量核实，匿名工具结果未证明逐一语义对应，语义改写/外部污染和全部来源许可未完全解决。这些项目继续列为部分完成，不为冻结而改成通过。`ready_for_training=true` 只表示数据就绪，`training_execution_authorized=false`；接下来在第5章固定比较规则，再在第6章组织模型与训练参数。

### 4.11 核对中间版本与最终数量

| 阶段 | 保留记录 | 本步变化 |
| --- | ---: | --- |
| 原始来源 | 905,718 | 固定原文件，不覆盖 |
| 长度候选 | 856,531 | 另存超长49,187条 |
| 精确去重候选 | 809,713 | 另存重复46,818条 |
| 两条确认决定后 | 809,711 | 排除两条原样错误示范 |
| 工具隔离后 | 808,513 | 另存剩余1,198条未声明调用 |
| 最终接受数据 | 787,425 | 限定修订、暂存21,073条、修订后重复15条 |
| 训练 / 验证 | 785,377 / 2,048 | 按组切分，不是再读Dataset时随机切一次 |

这些数量属于不同输入版本，不能混用。原1,199条未声明调用中，一条已被具体决定排除，所以后续统一隔离的是1,198条，避免重复计数。最终14条身份修订和2条内容修订是替换指定记录中的字段，并非向集合新增16条。

至此，数据配方已交付，但模型是否改善仍没有答案。下一章在更新任何SFT参数之前建立比较依据。


## 5. 训练前评测约定与B0

### 5.1 先定义比较，再改变模型

数据决定模型学什么，评测决定我们用什么证据判断学习是否有用。因此正式训练前先冻结输入、生成方式和判法，采集同一起点B0；训练后只替换为S1权重。不能看到S1输出后再挑有利的问题。

本轮使用自己的第七周预训练最终模型作为B0，不是官方发布的MiniMind成品。评测不恢复预训练优化器，只加载对应架构、tokenizer和原生FP32参数。评测输入与训练/验证做了限定归一化下的精确问题重叠检查，结果为0；尚未证明预训练未见过相关知识，也未做全面语义污染审计。

### 5.2 固定评测对象与执行方式

| 对象 | 本轮约定 | 要验证什么 |
| --- | --- | --- |
| 验证NLL | 全部2,048条，833,052个有效助手目标，batch=4 | 在同一目标分布上的拟合变化 |
| 核心生成 | 日常、指令、给定材料、多轮上下文各8题，共32题 | 是否完成用户任务、遵守约束、利用给定上下文 |
| 工具补充 | 4题，单独报告 | 调用文本或给定工具结果的解读，不执行真实服务 |
| 提示 | 相同system、固定历史、工具声明、模板；预填空think区域 | 不让提示差异或不同滚动历史混入权重对照 |
| 生成 | CUDA FP32，eval/no_grad，greedy，batch=1，最多256个新token，EOS=2，总预算768 | 两边按同一解码与停止条件输出 |
| 数值条件 | 不使用autocast/TF32，固定确定性设置与线程数 | 降低执行条件差异 |

greedy每次取当前logits最大的token，不保证整条序列的联合概率最大。训练长度768是输入与监督的总上下文上限；256是生成新增部分的预算，二者不是同一个量。检查过的最长提示373，加256后629，未超过768。

题目、共用配置、输入检查、接受记录与冻结标记集中在[评测配方](../experiments/sft_mini_v1/evaluation_recipe/README.md)。评分检查点与题号只参与检查和归档，不进入模型提示。具体步骤为：构造题目与判法 → 检查提示及数据重叠 → 检查严格加载 → 执行小规模GPU冒烟 → 接受并冻结 → 完整采集B0。入口及历史执行命令见该目录，不重复维护另一套参数。

### 5.3 保留逐题输出并核对评分

[evaluate_sft.py](../implementation/scripts/evaluate_sft.py)逐题保存输入token、生成token、未清理文本、展示文本、停止原因和耗时，再用同一验证集计算token加权NLL。失败时保留已产出的记录，不将部分结果当作全量结果。

每个检查点记pass/fail/uncertain并附证据；任一fail则该题fail，全部pass才通过。评分不是简单搜索答案关键词：说“不是红色”与说“是红色”可能都有关键词，但含义相反；没有捏造代码也不一定完成了“请求补充代码”的要求。

本轮逐题初评由助手执行，尚非独立人工复核。聚合脚本核对行身份、引用和统计，不能据此宣布语义判断自动正确。原始生成和派生评分分开保存，详见[B0结果目录](../experiments/sft_mini_v1/results/b0_v1/README.md)。

### 5.4 基线结果与评分修订

B0的FP32验证NLL为2.3954561616642405。36条生成均达到256个新token，未遇EOS；生成总计9,216个token。这描述终止行为，不自动将所有题判错。

V1初评核心3/32。复查发现地点题原提示已要求“只回答地点”，旧检查点却漏掉额外内容：B0虽然答对地点，又给出错误时间。经用户确认，保留原题目和原始生成，新增[评分V2补丁](../experiments/sft_mini_v1/evaluation_recipe/scoring_v2_patch.json)，不覆盖V1。按同一份输出重评分后，核心为2/32，工具0/4；后续两边统一采用V2。

这是本次真实的评测合同修订经验：检查点覆盖不足会抬高分数，发现后应保留版本与理由，不能静默改分。当前目标是比B0有初步改善，同时逐条分析回退与重复/结束行为；不另加8/32或EOS16/32门槛，也不把36题诊断扩大解释为通用能力结论。

## 6. 模型起点与训练配方

### 6.1 区分新阶段初始化与中断恢复

SFT沿用第七周架构与tokenizer，加载自己的预训练最终FP32参数，约63,912,192个参数，保留embedding/lm_head共享。新阶段重新建立AdamW与学习率计划，不继承预训练的优化器历史。

而SFT中断恢复要加载本阶段的模型、AdamW、scheduler、sampler、RNG与进度。B0是“从哪里开始”，resume checkpoint是“已走到哪里”；两者不能互相替代。严格加载先检查包清单、代码、配置、tokenizer、权重摘要及共享关系，不是仅凭文件能打开就算起点一致。

### 6.2 确定训练核心与完整计划

| 参数 | 本轮选择 | 作用或边界 |
| --- | --- | --- |
| 更新范围 | 全参数SFT | 不做LoRA或冻结层实验 |
| 数据与长度 | 冻结的785,377条训练记录，最大768 | 训练入口不再次筛选、修订或切分 |
| 轮数 | 完整1个epoch | 每条记录按固定顺序消费一次，尾部不丢弃 |
| batch | micro-batch=4，累积4次 | 普通窗口16条；按有效助手目标数归一化 |
| 学习率 | 1e-5，余弦衰减至10%，无warmup | 调度跨度是完整更新计划，不是短跑步数 |
| AdamW | betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01 | 唯一可训练参数放同一组，foreach/fused=False |
| 梯度 | max_grad_norm=1.0 | 窗口梯度归一化后、参数更新前裁剪 |
| 执行 | CUDA BF16混合精度，num_workers=0 | 不静默降级CPU；与最终FP32评测区分 |
| 随机性 | seed=42，sampler_seed=42 | 固定初始化相关随机性与本轮样本排列 |

完整窗口数为49,086，余1条，因此总更新数为49,087；micro-batch总数196,345。尾部仍按实际有效目标归一化，不能为了凑满16条而重读数据。训练集共326,249,045个有效助手目标，记录数与目标数是不同计数。

这些参数是本轮经讨论采用的起点，不是证明最优的通用SFT配方。本轮只跑一个主配置，不能把结果归因于某个超参数的独立效果。

### 6.3 将组件串成训练入口

```text
配置 -> 检查数据冻结与B0身份 -> 建立完整样本排列和更新计划
     -> SFTDataset / sampler / DataLoader / collator
     -> 一个窗口收集最多4个micro-batch
     -> train_update：forward、loss sum、backward、token归一化、裁剪、step
     -> scheduler更新、提交已成功消费的位置
     -> 日志、周期验证、checkpoint与receipt
```

本轮仍是自己的PyTorch训练链路，不是TRL Trainer。复用已有loss、更新、验证和checkpoint组件，SFT入口只接入对话数据和阶段配置；没有重新实现另一套优化算法。DataLoader提供batch，`train_update`负责一次更新，外层运行代码组织整轮与保存。

早期[train_sft.py](../implementation/scripts/train_sft.py)用于准备检查和有界短跑；后续[sft_formal_runtime.py](../implementation/scripts/sft_formal_runtime.py)补齐正式运行的周期验证、保存和预算保护，由[train_frozen_sft.py](../implementation/scripts/train_frozen_sft.py)校验冻结后调用。两者复用核心组件，不能用“旧入口短跑成功”代替对新增正式编排的验证。

### 6.4 从候选配置到正式运行安排

[sft_mini_candidate.json](../experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json)保存已经讨论并用于短跑的训练核心。它的运行项最初为null，因为保存频率和预算还没有实测依据；这份历史候选不被回写成正式批准状态。

先执行准备检查，核验数据、B0包、评分补丁、完整计划、排列和代码/依赖身份；不建模型、不forward：

```powershell
uv run --locked python -X utf8 scripts/train_sft.py `
  --config ../experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json `
  --output-dir ../experiments/sft_mini_v1/results/entry_prepare_recheck_new `
  --prepare-only
```

真实准备结果见[entry_prepare_recheck](../experiments/sft_mini_v1/results/entry_prepare_recheck/result.json)，状态为prepared_not_trained。上例使用新输出名，避免覆盖旧证据。

短跑与成本测量通过后，用户确认的运行安排写入[training_config.json](../experiments/sft_mini_v1/training_recipe/training_config.json)：引用原候选核心，覆盖正式operations。正式合同以展开后的有效运行项为准，不使用旧候选中的null。具体如何得到这些运行项，见下一章。


## 7. 短跑、恢复与成本验证

### 7.1 每一层验证回答不同问题

| 层次 | 要回答的问题 | 不能替代什么 |
| --- | --- | --- |
| 准备检查 | 配置、数据和来源是否匹配？ | 不证明GPU能运行 |
| 小模型测试 | 更新、尾批、恢复和拒绝分支是否正确？ | 不证明真实模型的资源需求 |
| 真实短跑 | 当前机器上能更新吗，耗时和显存如何？ | 不证明实际恢复、长程稳定或能力提升 |
| 连续/恢复对照 | 保存退出后，是否从同一状态和下一条样本继续？ | 不保证跨设备位级一致 |
| 正式入口门禁 | 新增运行编排是否仍满足恢复和停止约定？ | 不代替整轮最终验收 |

短跑从B0开始，沿用完整49,087步调度，在20步暂停；前64条验证只是工程监控，不是正式全验证或能力评测。复练命令如下，使用新目录，**当前正式训练期间不要并行执行**：

```powershell
uv run --locked python -X utf8 scripts/train_sft.py `
  --config ../experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json `
  --output-dir ../experiments/sft_mini_v1/results/probe20_replay_new `
  --probe-updates 20 --validation-records 64
```

### 7.2 先定位分配失败，再保持配方重试

最初三次真实GPU短跑都在完成更新之前失败；曾分别发生在初始验证或首个训练forward。错误显示还有空闲显存，同时Windows物理内存与系统提交余量偏低，因此没有立即改小batch或重装环境。

有界CUDA分配诊断观察到，显存申请也伴随系统提交量增加。这支持把系统提交额度纳入排查，但没有证明它是唯一原因。用户关闭暂时不用的应用后，第四次保持相同训练核心成功。故障、读数与限制见[短跑故障记录](../experiments/sft_mini_v1/results/PROBE20_STATUS.md)。

这次经验是：文件检查通过、代码单测通过、显存显示有空余，仍不等于当前机器可完成真实更新。失败应保留stderr、退出码和已完成步数；0步失败没有可恢复训练状态，也不能用来估计吞吐。

### 7.3 检查成功短跑，而不只看进度条

第四次完成20次更新、320条记录、128,859个有效助手目标，保存checkpoint并暂停，退出0。去首步窗口中位约0.748秒；峰值allocated约3.77GiB、reserved约3.88GiB，更新后allocated约0.79GiB，在这20步内稳定。

核对日志窗口数、目标计数、学习率、Adam步数、sampler提交位置，以及checkpoint字节与receipt。receipt证明保存文件与当时记录的身份对应；文件存在不代表其中状态已恢复执行。短跑报告与核验方式见[成功短跑](../experiments/sft_mini_v1/results/probe20_v4/README.md)。

前64条BF16工程NLL从2.8312降至2.5992，不能直接与全2,048条FP32基线比较。它只帮助判断更新链路和短程数值行为，不宣布助手能力提升。

### 7.4 用连续运行与中断恢复作对照

对照固定同一初始模型、数据排列和完整调度，建立两条路径：

```text
A：从B0连续更新到20 -> 保存
B：从B0更新到10 -> 保存并退出
                    -> 新进程加载完整checkpoint -> 更新到20 -> 保存
```

逐步比较第11–20步loss、有效目标数、梯度范数、学习率和消费位置；终点比较全部模型参数、AdamW、scheduler、sampler、CPU/CUDA RNG与元数据，而不是只看两条loss接近。候选入口的[实际对照](../experiments/sft_mini_v1/results/resume_comparison_v1.json)通过，本次连最终checkpoint文件SHA也一致。

相同权重只是必要条件之一；未恢复AdamW历史或数据位置，仍可能在下一步分叉。本次同设备、固定条件的短程精确一致，不能推广为任意硬件、不同内核或长时间运行都必然字节相同。

### 7.5 用实测决定保存、验证与预算

| 实测或估计 | 观察 | 用于什么决定 |
| --- | --- | --- |
| 短跑更新窗口 | 中位约0.748秒 | 估计纯训练约10.2小时，仍需长期余量 |
| D盘checkpoint | 约735MiB，保存及receipt约1.54秒 | 估计本地落盘量级 |
| D到E复制并刷盘 | 约22.16秒，复制后SHA一致 | 评估机械硬盘成本；未覆盖完整GPU到CPU序列化 |
| 全量BF16验证 | 2,048条、833,052目标，约40.80秒 | 决定全验证间隔 |
| 整轮文件 | 约50份、35.9GiB | 选择E盘keep_all，并要求初始至少40GiB可用 |

据此确认每1,000步保存、每5,000步全验证，开头和结尾也验证，活动时间预算14小时。保存与验证都计入墙钟耗时；“纯训练10.2小时”单列只是为了看清成本组成，不是把其他开销排除在预算之外。完整推导见[运行方案](../experiments/sft_mini_v1/training_recipe/OPERATIONS_PROPOSAL.md)。

### 7.6 在最终正式代码上补齐门禁

正式编排增加周期保存、验证与预算暂停，必须针对新增行为验证。小模型回归检查尾批照常更新、完整记录/目标计数、周期触发、预算边界保存暂停、无冻结拒绝正式运行及绑定文件篡改拒绝；当时53项相关回归通过，未重跑全项目。

最终代码另做真实GPU连续20步与10步后恢复到20步，仍沿用完整计划。参数、优化器、调度器、样本位置、RNG和逐步指标再次精确一致，见[正式入口恢复对照](../experiments/sft_mini_v1/results/formal_resume_comparison_v1.json)。这种重复有明确原因：验证对象从早期短跑入口变成了最终正式入口，不是单纯增加测试次数。

## 8. 最终训练配方冻结

### 8.1 把已确认选择和证据绑定起来

数据配方固定“读什么、怎样编码和监督”；训练配方固定“从哪里开始、怎样更新、怎样运行、依据哪些验证允许执行”。两者与评测配方放在同一实验目录的三个子目录中：

```text
experiments/sft_mini_v1/
├── data_recipe/         数据与编码身份、处理规则、接受及冻结
├── evaluation_recipe/   固定输入、判法、执行条件及冻结
└── training_recipe/     模型起点、优化计划、运行策略及冻结
```

这里不是要求记住所有文件名，而是顺着引用检查：训练冻结指向训练合同，合同绑定配置、数据冻结、B0模型、评测与代码，再由上游合同展开对应文件身份。小型控制文件进Git；真实数据、权重与运行产物由清单定位和校验，不把大文件塞进文档目录。

| 文件 | 职责 |
| --- | --- |
| sft_mini_candidate.json | 已验证的训练核心与早期候选身份，保留历史状态 |
| training_config.json | 引用候选核心，补齐已确认的正式operations |
| training_acceptance.json | 记录最终运行选择与授权范围 |
| TRAINING_CONTRACT.md | 面向读者解释本轮约定 |
| contract.json | 展开并计算可机器检查的实际身份 |
| training_freeze.json | 绑定合同、接受和必要证据，作为正式入口凭据 |

短跑与恢复报告是“配方为什么可执行”的证据，不是另一份训练超参数。目录说明与上游引用见[训练配方入口](../experiments/sft_mini_v1/training_recipe/README.md)。

### 8.2 一键生成冻结，而不手工拼hash

[freeze_sft_training.py](../implementation/scripts/freeze_sft_training.py)读取正式配置，展开身份，核对接受记录、最终代码上的稳定性/真实恢复证据与通过的测试XML，写合同，最后写冻结标记。本次执行关系为：

```powershell
uv run --locked python -X utf8 scripts/freeze_sft_training.py `
  --recipe-dir ../experiments/sft_mini_v1/training_recipe `
  --comparison ../experiments/sft_mini_v1/results/formal_resume_comparison_v1.json `
  --stability ../experiments/sft_mini_v1/results/formal_gate_cont20_v1/result.json `
  --tests ../experiments/sft_mini_v1/results/final_runtime_tests_v1.xml
```

这些输出已存在，不能重复生成或覆盖。最终合同SHA256为：

```text
3ac716dee78ce33d5d418e7f1f5e7e6b176dcdad6d06c52ac608322172c1abe9
```

模型、tokenizer、数据、配置、样本顺序、代码与依赖、目标快照、评测及V2评分补丁进入绑定或递归复查链。冻结时记录formal_training_started=false，含义是“冻结动作没有开始训练”，不是永远代表当前运行状态。已绑定的目标快照不再回写进度，进度更新README、本文与运行manifest。

### 8.3 启动和恢复时重新检查

```powershell
uv run --locked python -X utf8 scripts/train_frozen_sft.py `
  --freeze ../experiments/sft_mini_v1/training_recipe/training_freeze.json `
  --check-only
```

入口复核冻结绑定、重建训练身份并检查上游数据与评测；只读检查不forward、不恢复或更新参数。实际[冻结预检](../experiments/sft_mini_v1/results/frozen_preflight_v1/result.json)为checked_not_trained。

显式执行或恢复也先走这条检查。检查发现版本变化应失败，不能为继续运行临时更新hash；确需改配方或代码，要按变更范围重新验证并创建新版本。身份校验不重新判断内容真伪，也不替代E盘连接、系统资源余量等运行条件检查。

## 9. 正式启动、观察与恢复

### 9.1 以冻结身份启动新实验

正式SFT从B0、step0开始，工程短跑的checkpoint不能冒充正式起点。此前已按用户确认于2026-10-31北京时间17:43后台启动，11月1日04:26成功退出。下面保留启动方式供理解和后续复练；本轮已完整结束，**不要对原实验目录再次执行**：

```powershell
$processDir = Join-Path $PWD "runs/sft_mini_one_epoch_v1_supervisor"
New-Item -ItemType Directory -Path $processDir -ErrorAction Stop | Out-Null
Start-Process -FilePath powershell.exe -WindowStyle Hidden `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass",
                  "-File", "scripts/run_sft_background.ps1",
                  "-ProcessDirectory", $processDir) `
  -WorkingDirectory $PWD
```

[后台监督脚本](../implementation/scripts/run_sft_background.ps1)启动Python、抑制空闲睡眠，并保存命令、PID、stdout、stderr与退出码。其内部调用的是：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/train_frozen_sft.py `
  --freeze ../experiments/sft_mini_v1/training_recipe/training_freeze.json `
  --run --output-dir runs/sft_mini_one_epoch_v1
```

监督目录与训练输出目录均不得覆盖。固定启动脚本用于本次新实验，不是可传任意参数的通用launcher，也没有直接接收resume参数。启动前正式入口检查CUDA、数据与模型身份、E盘路径及初始空间等条件；不会自动删除已有checkpoint腾空间。具体启动证据见[正式启动记录](../experiments/sft_mini_v1/results/FORMAL_LAUNCH.md)。

### 9.2 用文件分工理解运行状态

| 位置 | 读取什么 |
| --- | --- |
| implementation/runs/sft_mini_one_epoch_v1/run_manifest.json | 当前训练状态、步数与已记录的进度 |
| implementation/runs/sft_mini_one_epoch_v1/ | 训练日志、验证与本次执行产物，实际名称以目录为准 |
| implementation/runs/sft_mini_one_epoch_v1_supervisor/process.json | 监督进程、Python PID、退出码及日志路径 |
| 同一supervisor目录的stdout.log / stderr.log | 正常输出与异常，包括程序内部未及时记录的失败 |
| E:/AI_research_checkpoints/minimind/sft_mini_v1/formal | 正式checkpoint及对应receipt |

只读查看可以从manifest和最近日志开始，不需要重新执行训练脚本或一直附着进程：

```powershell
Get-Content runs/sft_mini_one_epoch_v1/run_manifest.json
Get-Content runs/sft_mini_one_epoch_v1_supervisor/process.json
Get-Content runs/sft_mini_one_epoch_v1_supervisor/stdout.log -Tail 12
Get-Content runs/sft_mini_one_epoch_v1_supervisor/stderr.log -Tail 12
```

manifest为running不保证进程一定活着；被kill或断电时可能来不及更新。综合进程是否存活、步数和日志是否推进、checkpoint时间及错误信息判断。GPU利用率只是旁证，不能单独证明训练正常或停止。

### 9.3 同时观察数值、资源和任务进度

每次更新看loss、有效助手目标、裁剪前grad norm、实际学习率、耗时、已提交记录和显存；周期全验证看相同BF16条件下NLL趋势。观察重点是非有限数值、突发异常、分配持续增长、长时间无推进，以及计数与冻结计划是否对应。

训练中BF16验证与最终FP32配对评测不同，不将两者数字混成一条完全同条件曲线。grad norm略高于1.0也不等于裁剪失败：日志保存的是裁剪前值，更新使用裁剪后的梯度。具体解释见文末QA。

保存、验证与运行检查的耗时也计入活动时间。14小时预算跨恢复累计，在更新边界保护性保存并暂停，不是到点强杀；验证与落盘可能跨过预算检查点，因此不是硬实时上限。

### 9.4 中断后从已提交的状态恢复

先确认旧进程已经停止，再检查最后完整checkpoint与receipt，不能选择仍在写入的文件。模型、优化器、调度器、sampler和RNG一起恢复，调度仍按原49,087步计划；已消耗的活动时间也不能清零。

以下展示正式入口的恢复参数，checkpoint-N.pt是占位示例，不是可以原样执行的真实文件名：

```powershell
uv run --locked python -X utf8 scripts/train_frozen_sft.py `
  --freeze ../experiments/sft_mini_v1/training_recipe/training_freeze.json `
  --run --resume E:/AI_research_checkpoints/minimind/sft_mini_v1/formal/checkpoint-N.pt `
  --output-dir runs/sft_mini_one_epoch_v1_resume_new
```

恢复使用新运行目录保留每次尝试，不创建新实验语义；已有E盘正式保存目录由恢复分支按合同继续使用。本例为前台参数示意，需要后台恢复时应另保存命令、stdout、stderr与退出码，不能误用固定的新训练监督脚本重新从B0开始。

## 10. 训练后评测与结果分析

### 10.1 从运行结束到工程验收

训练进程退出后，先确认发生了什么，再加载模型评测。manifest的completed不是全部证据：要与监督退出码、日志连续性、样本与目标计数、最终验证和checkpoint内部状态交叉核对。若只是预算暂停，不能当成完整一轮。

新增[finalize_sft_run.py](../implementation/scripts/finalize_sft_run.py)串起原冻结检查与验收：逐条检查49,087条更新事件，确认196,345个micro-batch、785,377条记录、326,249,045个有效助手目标；最后一次更新只含1条记录和413个目标，确实没有丢尾批。再核对50次保存与11次全验证，读取最终receipt并完整计算文件SHA，检查模型、AdamW、scheduler、sampler与RNG。

本轮于11月1日04:26:24成功退出，stderr为空。最终参数和AdamW状态有限，90个AdamW状态的步数及scheduler均为49,087，排列与提交位置匹配。50份文件/receipt均存在，但只对最终checkpoint完整计算字节摘要并加载，没有把文件清单齐全写成50份tensor状态全部检查通过。

总墙钟10小时43分17秒，其中更新约10小时12分、保存约22分46秒、全验证约6分53秒，其余为启动和编排等开销，未用满14小时预算。证据和逐项检查理由见[正式训练验收](../experiments/sft_mini_v1/results/final_run_v1/README.md)。

### 10.2 导出最终模型并验证独立加载

checkpoint包含恢复状态；模型包只提供推理所需的配置、FP32权重、tokenizer、运行代码与说明。导出不是再训练，也不是量化，而是把最终参数按可独立交付的形式组织起来。

验收脚本严格加载最终模型，复用第七周的create_bundle与verify_in_fresh_process。包内说明改为SFT阶段后重新生成清单，再用新进程、离线、包内runtime加载。比较完整参数摘要、共享权重，以及两条固定短前缀的输入、logits和8-token预算生成。本轮逐值一致、最大logit差0；这一小探针不证明所有输入或CUDA/BF16都一致。

实际命令如下，输出已经存在，复练需使用新目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/finalize_sft_run.py `
  --experiment-dir ../experiments/sft_mini_v1 `
  --run-dir runs/sft_mini_one_epoch_v1 `
  --supervisor-dir runs/sft_mini_one_epoch_v1_supervisor `
  --report-dir ../experiments/sft_mini_v1/results/final_run_v1 `
  --bundle-dir exports/sft_mini_one_epoch_v1_fp32_v1
```

模型包仍是本地原生格式，不是Transformers AutoModel适配。原始prompt命令行用于续写与加载检查；助手评测必须经过已冻结的chat template，不把原始字符串续写当作对话评测。

### 10.3 复用冻结引擎执行S1评测

实际检查后发现原evaluate_sft.load_weights可以加载同格式模型包，并非硬编码只接受B0。前文曾预判必须修改加载分支，这里纠正：没有修改原引擎，而是新增[evaluate_sft_pair_member.py](../implementation/scripts/evaluate_sft_pair_member.py)作为S1来源适配层。

适配层核对原评测冻结的全部文件与依赖、V2补丁和训练冻结，绑定已验收的最终S1参数，要求S1配置与B0逐项相同，再直接调用原evaluate_sft.run。生成、SFTDataset/collator、全验证与执行条件全部复用；运行前后再次检查来源身份。先固定模型来源，再运行评测，不能只用“同一题号”代替实际输入比较。

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/evaluate_sft_pair_member.py `
  --experiment-dir ../experiments/sft_mini_v1 `
  --bundle exports/sft_mini_one_epoch_v1_fp32_v1 `
  --audit-dir ../experiments/sft_mini_v1/results/final_run_v1 `
  --output-dir ../experiments/sft_mini_v1/results/s1_v1 --run
```

36题生成和全部2,048条FP32验证已完成，实际prompt及input_ids与B0逐条相同；模型参数摘要在评测前后不变。原始输出、NLL、停止原因、执行及适配身份都独立保存，见[S1评测报告](../experiments/sft_mini_v1/results/s1_v1/README.md)。不改B0证据，不因评分变化重跑生成。

### 10.4 将逐题判定绑定到原文

先阅读全文，按原检查点及V2补丁逐项写pass/fail/uncertain、理由和引句。不能只查答案词，也不能把EOS当作正确。助手本轮判断记录在judgments.assistant.v1.json，身份绑定后的完整评分另存，不回写原始generations。

[analyze_sft_pair.py](../implementation/scripts/analyze_sft_pair.py)核对B0评分及S1执行证据、实际输入、生成长度与停止原因，将每项人工语义判断绑定到原始行hash和原检查点，再计算分类、翻转、通过率上下界与NLL变化。它不调用额外模型，也不自动理解答案是否正确。

本次核心题有1项待裁定：grounded-07正确复述钥匙移到顶层，却又复述问题并重复条件句，是否已经回答了最终位置不够明确。暂记uncertain，留给上午复核，不能为了总分选一个有利判法。grounded-06虽然因果信息正确，但两个完整句子不符合原单句概括要求，仍判失败。

### 10.5 分开解释拟合、任务与回复行为

| 证据 | B0 | S1 | 当前结论 |
| --- | --- | --- | --- |
| 全验证FP32 NLL | 2.395456 | 1.721082 | 下降约28.15%，分布拟合改善 |
| V2核心32题 | 2通过、30失败 | 0确定通过、31失败、1待裁定 | 即使待裁定算通过也只有1/32，未超过B0 |
| 工具补充4题 | 0通过 | 0通过 | 不声称工具调用能力改善 |
| 核心配对 | 基准 | 30条都未通过、1条确定回退、1条待裁定 | 无确定正向翻转，不确定项不硬计翻转 |
| EOS / 长度上限 | 0 / 36 | 17 / 19 | 部分结束行为改善，但仍有大量重复与任务失败 |

确定回退grounded-04中，B0虽重复，却指出食品不接收；S1未作出该判断。地点题S1能保留正确时间并结束，但未遵守“只回答地点”，因此仍失败。工具题中，S1甚至用lookup_book查天气：总分同为失败也可能出现新的风险形态。

NLL是在真实上下文下给目标token打分，开放生成要连续使用自己的输出，且任务评分检查整段含义与约束；二者并不等价。本轮只能确认“拟合改善但任务未改善”的现象，不能在没有消融时把原因确定归给数据、训练强度或模型规模。

完整逐题索引、引句、统计和上午核对顺序见[配对分析](../experiments/sft_mini_v1/results/paired_analysis_v1/README.md)。31项收尾/导出/评测/评分定向回归通过，原训练冻结事后只读复查也通过；它们证明工程边界，不自动证明语义评分或能力结论。

### 10.6 归档与待核对事项

当前工程、完整评测和初步分析均已落盘；**任务能力改善目标尚未达到**，评分未独立复核。上午先核对确定回退、唯一待裁定与工具风险，再讨论一个有明确假设的后续问题；不自动增加训练、改解码、换checkpoint或降低评分要求。

本地模型包、数据/训练/评测配方、完整生成与评分均保留；50份checkpoint在E盘未删除。用户确认后已发布独立HF模型仓库，尚未推送Git。发布说明如实区分工程完成、拟合改善和任务失败，不将这次训练包装成成熟助手。

2026-11-01先完成[中英文模型卡草稿与核对记录](../experiments/sft_mini_v1/release/model_card_v1/REVIEW.md)，默认英文、链接切换中文，另提供实际验证过模板和短生成的单轮聊天示例。随后用户确认上传，进入下面的发布步骤。原草稿保留其创建时状态，不再作为最新状态入口。

### 10.7 模型发布与下载核验

训练checkpoint包含恢复状态，不能直接作为面向下载者的完整使用入口。我们复用前面已验证的FP32推理导出包，加入模型卡、聊天入口、来源声明、配方和评测证据，形成新的发布包；权重不再训练或转换，原导出包和冻结合同保持不变。

发布工具为implementation/scripts/publish_sft_bundle.py，依次完成四步：

1. **组包检查。** 核对原生S1清单和最终参数摘要，按白名单复制文件；配方快照去除本机路径、注明原SHA256，训练原文和恢复checkpoint不上传。扫描发布文本中的凭据与本机路径模式。
2. **本地重载。** 用发布包自己的代码、配置和tokenizer，在隔离离线进程中重建模型，核对参数摘要、权重共享、固定logits与短生成；另检查chat.py入口。先排除组包过程破坏可用性的问题。
3. **独立仓库上传。** 确认HF账号，创建minimind-dense-64m-mini-sft，只上传本次白名单，不覆盖预训练仓库。上传结果记录实际提交，不把“请求发出”当作成功。
4. **固定版本复验。** 使用新临时缓存，从上传提交强制下载全部文件，重新核对哈希并执行独立重载。它验证实际公开内容，而不是再次读取原本地文件。

本次47个发布文件共约256.67MB，固定revision为`dc58b3dc61b4a4039894c2a64d82ff8cba910988`。文件身份全部匹配，CPU FP32参数摘要、权重共享、两条固定输入的logits和短贪心生成完全一致，最大logits差为0；聊天入口短生成通过。这是有边界的发布一致性检查，不代表重新通过了任务能力验收，也不保证跨设备逐位一致。

入口为[HF模型仓库](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft)，模型卡顶部可切换中英文。详细文件范围、执行过程和核验报告见[发布记录](../experiments/sft_mini_v1/release/public_v2/README.md)。10项模型卡/发布定向测试通过；本步没有重跑全项目测试，也没有自动推送Git。

### 10.8 执行效率与后续实验

#### 10.8.1 优化目标

完成模型评测后，下一步不一定是增加数据或更换模型，也可以先问：**在相同模型与训练目标下，能否用更少时间和显存完成训练？** 本轮手写组件便于理解计算与验证正确性，但这不意味着它们已经达到当前硬件的性能上限。

因此，后续实验需要分开评价两类改进：能力实验检查模型是否更好地完成任务；执行效率实验检查相同工作量的耗时、吞吐与资源占用。速度提高不能解释成本轮S1能力已经改善，模型能力未改善也不代表工程效率没有优化价值。两条结论应分别保留证据。

#### 10.8.2 对照方法

本次先在服务器探针中观察到SDPA提速，随后回到本机RTX 4060 Ti 8GB做独立SFT对照。原实现显式构造Attention分数和权重；候选调用PyTorch的`F.scaled_dot_product_attention`，仍保留显式的因果与padding可见性规则，由运行环境选择可用后端。我们只在测速进程中临时替换调用，没有修改历史模型与训练代码。

对照使用同一份预训练初始权重、tokenizer和固定抽取的1,024条已处理训练样本。两条路径均采用微批4、累积4、有效batch16、长度768、FP32参数/BF16计算，以及相同优化器、确定性和TF32设置。固定小批次数值对比先检查loss与梯度；性能部分先预热，再分别测缓存编码后的更新，以及包含逐条编码和组批的更新，避免把仅有GPU计算的速度误认为完整输入链路速度。

入口是implementation/scripts/benchmark_attention_backends.py；具体实现、环境、服务器与本机证据统一见[容量与吞吐探针](../experiments/full_data_capacity_probe_v1/README.md)。该探针不是新的正式训练，也不是完整版数据已经准备完成的证明。

#### 10.8.3 实测结果

| 本机SFT指标 | 原手写Attention | SDPA自动后端 |
| --- | --- | --- |
| 缓存编码后的吞吐 | 22.19条/秒 | 38.95条/秒 |
| 包含编码、组批的吞吐 | 21.86条/秒 | 37.36条/秒 |
| 含编码更新的中位耗时 | 0.732秒 | 0.428秒 |
| 训练窗口峰值allocated | 约3.77GiB | 约2.86GiB |

本次含编码吞吐约提升71%，对应每次更新耗时约减少42%；不能把“吞吐增加71%”写成“训练时间减少71%”。缓存计时窗口各约15秒，另测3次含编码更新，尚未进行整轮或长时稳定性验证，也不包含完整数据随机读取、周期保存、评测和发布等全部开销。

固定4条SFT样本的原NLL为2.8488555，SDPA为2.8482838，差约5.72e-4；梯度有限，但全局L2相对差约1.54%，真实位置logits平均绝对差约0.01116、最大差0.234375。改变后端会改变浮点运算顺序与舍入，不能称为逐位一致。这里的小批次loss用于数值诊断，不是新的能力评分。

性能对照按等时窗口运行，原实现完成21次缓存测量更新，SDPA完成37次。因此，两条测速路径训练后的NLL不能拿来证明某个后端“学得更好”：它们已经接受了不同数量的更新。若要比较学习效果，需要另外固定更新数、样本顺序与评测条件。

本机强制Flash候选未跑通，当前Windows版PyTorch明确报告未编译FlashAttention内核。成功结果来自可用的SDPA自动后端，不应记录成FlashAttention已启用；也没有为跑通候选而关闭确定性。完整本机数值见[原始结果](../experiments/full_data_capacity_probe_v1/local_attention_backends_v1/summary.json)。本机预训练后端提速尚未单独实测，不能直接套用SFT的71%。

随后用户授权新建独立WSL环境，Linux版同版本PyTorch实际跑通了强制Flash前向、反向及两阶段六组短测。同一WSL环境内，预训练含编码吞吐为原实现24.27、SDPA自动42.18、强制Flash45.59条/秒；SFT分别为23.69、37.99、42.85条/秒。强制Flash峰值allocated约2.82GiB，原实现约3.77GiB。这补充了Linux路径证据，不改变上面的Windows构建限制。

该Flash适配依赖严格右padding、因果约束及padding目标不监督，省略了对真实查询冗余的padding key掩码；不是任意mask的通用替换。小批次梯度有限，但与手写路径仍有浮点差异，尚未通过正式恢复或能力验收。本次跳过重复checkpoint保存，E盘另有文件迁移并发，故不将短测直接当作整轮墙钟预算。完整设置、数值差异和原始结果见[WSL测试记录](../experiments/full_data_capacity_probe_v1/README.md#8-wsl强制flash测试)。

#### 10.8.4 采用条件

本次可以将“Attention执行后端”列为后续工程优化方向，但**有短测收益不等于可以直接替换正式实现**。采用前依次完成：

1. **语义与数值。** 在相同输入和初始权重下，覆盖因果约束、padding、助手监督、loss及梯度，明确允许的浮点误差；若使用省略右padding掩码等其他快速路径，另行证明其适用条件，不能与本次保留显式mask的结果混为一谈。
2. **稳定性与恢复。** 固定工作量补足较长运行，核对吞吐、峰值显存、非有限数值和保存/验证开销；候选后端内部重新完成连续运行与中断恢复对照，不要求它与旧后端逐位相同，但要验证自身恢复合同。
3. **配方与结果。** 在新实验中绑定后端、依赖版本及验收证据，再执行训练与同合同评测。旧实验继续使用原冻结实现，不把效率候选用于旧checkpoint的隐式跨后端续训，也不覆盖已发布权重与原结果。

这样，执行效率就成为一项有假设、有对照、有边界的实验，而不是凭显存剩余或更换显卡来猜测速度。它的直接价值是降低后续实验成本、增加可承受的验证次数；是否同时影响模型质量，仍需独立证据。


## 11. 学习 QA

### 11.1 预训练与 SFT 只是 mask 不同吗？

在我们当前的 causal-LM 计算中，两者复用相同骨干和 next-token 交叉熵，监督位置是直接可见的差异。但阶段目标、数据分布、对话模板、起点、训练强度与评测也发生变化。不能只改 mask 就宣称完成助手训练。

### 11.2 用户不计 loss，为什么还要保留？

因为助手预测需要条件上下文。-100 屏蔽的是某个目标位置的 loss，而不是删除输入 token 或切断它参与后续计算的梯度路径。模型仍可能通过助手目标学习如何使用用户信息。

### 11.3 assistant 是普通 token，还要不要监督？

按当前规则不监督，因为它属于调用方提供的角色头；不是因为普通 token 天生不该监督。反过来，EOS 是特殊 token，当前却属于需要学习的结束目标。判断依据是任务约定与位置。

### 11.4 回复结束后的换行为什么也被监督？

当前参考路径将 `<|im_end|>\n` 作为完整结束序列。训练时保留换行，可以学习模板串行格式；推理时仍可以在 EOS 处立即停止，并不要求继续生成换行。本轮已将该换行纳入监督约定，不能在编码和评测之间默默改变。

### 11.5 reasoning_content 与 thinking 有什么区别？

前者是记录中的文本字段，模板可能把它包装进 think 标签；thinking 也可能指某种生成模式或行为约定。已有思考文本只是训练示范，同样需要检查内容质量及与最终回答的一致性，不能当作真实思维或自动正确的证明。

### 11.6 为什么搜索边界要加 1，break 又只退出一层？

`range` 右端不包含，查找长 M 的序列需要检查最后起点 N-M，所以右端写 N-M+1。内层 break 表示这次回复的结束已经找到，外层还要寻找下一轮；若两层都停止，多轮样本只剩第一轮监督。二者解决不同问题。

### 11.7 repr 和单 token 解码结果能说明什么？

`repr` 帮助看清换行、空白等字符串表示，不修改原文。字节级分词的单个片段可能不足以组成完整字符，单独显示异常时，要同时核对 token ID、完整解码和模板文本，再判断是否真的编码错误。

### 11.8 结构检查通过，为什么还有明显错误？

基本 validator 只能证明字段与类型满足已实现的规则。缺少用户代码却编造输出的样本，JSON 结构可以完全合法；调用不存在工具的样本，也可能通过仅检查角色和字符串类型的规则。需要内容、协议和集合层面的独立检查，不能用一个“通过数”代替全部结论。

### 11.9 抽样审核决定会不会造成训练数据重复？

不会自动加入。审核记录是控制信息，通过源行与 hash 指向原样本，供后续保留、修订、排除或暂缓处理。只有正式构造派生数据时才决定输出哪些对话；不能把“样本原文”和“审核表中的同一原文”各追加一次。

### 11.10 为什么先检查编码，再选择长度与短跑？

工具转换、思考保留与监督边界都会改变最终 token 长度和目标数量。先确认当前处理对象，再统计相应分布，才能让长度选择与资源估计有依据。前缀 20 条、随机 1,000 条和全量扫描的结论范围也不同，不能互相冒充。

### 11.11 内容没有全部核实，为什么可以冻结？

冻结是固定“在什么目标和已接受限制下，实际用了哪版数据和处理规则”，不是宣布所有清洗项目均已完成。本轮明确作为教学实践，限定审核范围、暂存未审语境，并记录事实与污染等缺口后接受这个版本。若用于风险更高的真实产品或要作更强能力结论，必须重新定义标准并补足证据，不能沿用这个冻结标记冒充质量认证。

### 11.12 数据就绪为什么还不能直接开训练？

数据配方回答“读什么、怎样编码和监督”；训练配方还要回答“从哪份权重开始、更新多少、用什么优化和调度、怎样验证与比较、预算和停止条件是什么”。二者需要分别确认，再以引用绑定。因此数据就绪与启动授权是两个独立字段；本轮已分别完成冻结，不代表以后可以省略其中一步。

### 11.13 NLL到多少才算好？

没有跨数据、tokenizer、监督范围与上下文都通用的合格线。先比较同一验证集、编码和精度下的B0与S1，再看实际任务输出。验证NLL按总loss sum除以有效助手目标数，不是按记录数或各batch平均值再平均。

本轮训练中的BF16全验证应与它自己的step0比较；第5章B0的FP32结果留给最终FP32对照。NLL下降说明目标分布拟合改善，不单独证明回答正确、遵循指令或正常终止。PPL=exp(NLL)只是同一损失的变换，也不是准确率。

### 11.14 grad norm在1出头，意味着信息太多或没有拟合吗？

不能这样推断。当前日志在一个累积窗口的梯度按有效目标数归一化后，记录裁剪前的全局范数，再以1.0裁剪。若记录为1.2，裁剪因子约为1/1.2；并非“裁剪以后仍是1.2”。阈值是更新前的保护规则，不是要求日志收敛到1以下的学习目标。

范数受监督位置、样本难度、梯度方向抵消、模型状态和batch等共同影响，不能反推出单条样本的信息量，也不直接等于AdamW参数更新幅度。预训练有效batch256、本轮16，数据和监督也不同，因此两条grad norm曲线不能直接比较高低。结合loss、验证趋势、非有限值和突发尖峰判断，不因稳定地略高于阈值就认定训练异常。
