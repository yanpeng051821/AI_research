# SFT 评测配方与基线

> 更新：2026-11-01。B0/S1已按同一冻结FP32设置与[地点题V2补丁](scoring_v2_patch.json)完成配对评测。B0核心2/32；S1助手初评0确定通过、31失败、1待裁定，工具均0/4。输入不变，评分尚未独立复核，见[配对分析](../results/paired_analysis_v1/README.md)。下文草案、未运行和V1分数是各自历史阶段证据，不作为当前执行状态。

## 1. 当前结论

原32条核心题、4条工具题、CUDA/FP32规则及逐题检查点已冻结；发现地点题检查遗漏后，用户在训练前确认仅追加V2补丁，B0按原输出重评分为2/32。SFT已完整结束，[S1评测](../results/s1_v1/README.md)按同一规则采集36题及全验证；FP32 NLL从2.395456降为1.721082，但任务初评未改善。正式身份仍以原合同、冻结标记及单独V2补丁为准，没有覆盖V1或回写冻结配置。

本轮比较自己的约 64M 预训练模型 B0 与从同一起点得到的 SFT 模型 S1，观察助手行为与回答质量的变化。不是通用能力榜单，也不把 36 条人工题称为独立盲测集。之后若根据这些题调配置或提示，应将其视为开发诊断，不能继续声称是未接触的最终测试。

E 盘最终 checkpoint-4943.pt 已重新可见，SHA256 核对为 8d3cb52867e0c74b35c5931bfb9c90de3a7b6589fd9466fd68d00c9593431e38，与预训练导出来源记录一致。B0 实际加载对应的原生 FP32 导出参数，核对完整状态摘要，不恢复 optimizer 或预训练进度；后续 SFT 是从同一模型参数开始的新实验。

## 2. 两类证据

1. **验证 NLL。** 使用数据合同冻结的全部 2,048 条验证记录，复用 evaluate_nll，统计 loss_sum / 有效助手目标数；不将各 batch 的平均 loss 直接平均。B0 和 S1 都用同一编码、batch 与精度。
2. **固定生成。** 使用本目录的题目，检查任务完成、约束、角色/重复异常和终止方式。核心 32 条与工具 4 条分开报告；工具题不调用真实服务，只看调用文本或给定结果的解读。

设备、精度和范围固定；用户在[目标快照](../EXPERIMENT_OBJECTIVES.md)中确认比B0有初步改善，不设置额外固定分数/EOS门槛。当前没有自动语义能力通过标准，NLL下降不能抵消生成退步；初评及歧义需人工核对，不能事后放宽规则。

## 3. 题目与评分

### 3.1 文件分工

- [probes.jsonl](probes.jsonl)：冻结的每题固定历史、最后输入、工具声明及人工检查点；与草案逐字节一致。
- [evaluation_config.json](evaluation_config.json)：正式共用 system、解码和 NLL 设置、评分与执行授权。
- [inspection.frozen.json](inspection.frozen.json)：正式输入的 hash、长度、重叠和实际提示，不是模型输出；其中未批准状态为最后写冻结标记之前的检查快照，不回写成运行结果。
- [evaluation_acceptance.json](evaluation_acceptance.json)：用户执行授权与草案来源身份；旧 .draft 文件保留，不覆盖。
- 本页：解释为什么这样出题、怎样判定、还没有确认什么。

评分说明与输入可以保存在同一记录中，但构造提示时**只读取 messages、tools 和共用 system**，不读 checks、类别、题号或标准答案。inspect_sft_eval_draft.py 的 render_probe 与两项定向测试验证了这条隔离边界；未来生成入口仍须复用或验证同一构造方式。

### 3.2 判分口径

每个检查点分别记 pass / fail / uncertain，并写简短证据。单题任一检查点 fail 则任务 fail；全部 pass 才为 pass；没有 fail 但仍有 uncertain 时，整题为 uncertain。开放回答接受语义等价表达，不要求照抄检查点。

报告确定通过数、失败数和不确定数；通过率以固定 32 或 4 为分母，同时给出确定通过率及把全部不确定视为通过的上界，不删除难判样本来抬分。对照列出双方都对、B0 对而 S1 错、B0 错而 S1 对、双方都错；不确定对单列，不能硬算成翻转。

输出质量与终止分别记录。达到 max_new_tokens 不自动等于任务失败；遇到 EOS 也不自动等于回答正确。每题保留完整生成 token、未清理文本、展示文本、停止原因和人工评分，不能只保留最终总分。B0原始运行文件与派生初评分别归档，见第7.3节。

### 3.3 完整题目表

类别依次为日常问答、指令约束、给定材料、多轮上下文、补充工具。下表只展示最后一条消息和首要检查点；多轮历史、工具定义和全部评分要求以 JSONL 为准。

| 题号 | 最后一条输入 | 首要检查点 |
| --- | --- | --- |
| everyday-01 | 小禾把同样湿的两块毛巾分别摊开和揉成一团晾晒。在其他条件相同的情况下，为什么摊开的通常干得更快？ | 解释摊开通常增加与空气接触的面积，有利于水分蒸发。 |
| everyday-02 | 小远在晴天发现自己站到树荫里之后，地上看不见完整的人影了。请简单解释影子形成需要什么条件。 | 说明光线被物体遮挡形成较暗区域，通常需要光源、遮挡物和承接影子的表面。 |
| everyday-03 | 把一块普通冰块放进水里，它通常浮在水面上。请用容易理解的话解释原因。 | 指出普通冰的密度小于液态水，浮力可以支撑冰块。 |
| everyday-04 | 小竹准备清理电脑文件。她问：备份文件和直接删除文件有什么区别？请用日常语言回答。 | 区分备份是另存可供恢复的副本，删除是移除原有文件。 |
| everyday-05 | 小岚把一张纸质地图当成了真实道路，认为地图上画得短的路走起来一定更省时间。这个想法哪里不完整？ | 指出地图表达或距离不能单独决定耗时，还受比例尺、路况或出行方式等影响。 |
| everyday-06 | 小园有20个橘子，每盒放4个，所有橘子都装完需要多少盒？请给出答案和一句解释。 | 答案为5盒。 |
| everyday-07 | 小帆上午读了12页书，下午读了7页，没有重复阅读。今天一共读了多少页？请简短回答。 | 答案为19页。 |
| everyday-08 | 小青说：我已经看懂了示范，所以不用自己做也一定能做好。请温和地说明理解示范与独立完成之间的区别。 | 区分看懂过程与能独立执行、排错，两者不能直接等同。 |
| instruction-01 | 只输出一个合法JSON对象，键为city和count，值分别为“青禾镇”和3，不要解释。 | 去掉首尾空白后整个输出能解析为JSON对象，不含代码围栏或额外解释。 |
| instruction-02 | 请恰好写三行，依次为“备份文件”“核对副本”“清理原件”。不要加序号或其他文字。 | 忽略最外层空白后恰好三行。 |
| instruction-03 | 将以下数字按从小到大排列，只输出用英文逗号分隔的数字：17、4、11。 | 去掉首尾空白后输出恰好为4,11,17。 |
| instruction-04 | 请判断“9大于6”是否成立。只输出“是”或“否”中的一个字。 | 去掉首尾空白后输出恰好为是。 |
| instruction-05 | 把“周二下午整理图书”中的“周二”替换成“周四”，其他内容不变，只输出修改后的句子。 | 去掉首尾空白后输出恰好为周四下午整理图书。 |
| instruction-06 | 请给“先关闭水龙头，再擦干桌面”这两步编号。只输出两行，格式分别为“1. …”和“2. …”。 | 恰好两条编号步骤，先关闭水龙头，后擦干桌面。 |
| instruction-07 | 下面三个词中，哪个表示颜色：石头、紫色、台灯？只输出这个词。 | 去掉首尾空白后输出恰好为紫色。 |
| instruction-08 | 请只输出下面句子的前四个汉字，不要标点或解释：午后读书很安静。 | 去掉首尾空白后输出恰好为午后读书。 |
| grounded-01 | 记录：青禾图书角周三开放，开始时间14:20，结束时间17:10，地点是二楼东侧。根据这份记录，只回答开放地点。 | 答案包含二楼东侧。 |
| grounded-02 | 便条写着：阿澄负责带蓝色文件夹，阿沐负责带白板笔，阿桥负责带计时器。根据便条，谁负责带白板笔？ | 回答阿沐。 |
| grounded-03 | 库存记录：铅笔8支，橡皮5块，直尺2把。只根据这份记录回答：橡皮和直尺合计多少件？ | 回答7件，计算为5加2。 |
| grounded-04 | 通知：本次交换会只接收图书和文具，不接收食品。小叶想带饼干来交换。根据通知，饼干是否在接收范围内？ | 明确不在接收范围内。 |
| grounded-05 | 材料：星桥小组共有三个人，本周计划修好一盏台灯。材料没有记录任何人的年龄。根据材料，组员的平均年龄是多少？ | 明确无法根据材料确定平均年龄。 |
| grounded-06 | 请把下面的活动记录概括成一句话，保留人物和结果：阿岚试着修理旧收音机，发现是电池没电。换上新电池后，收音机重新响了。 | 保留阿岚、更换电池、收音机恢复工作的核心信息。 |
| grounded-07 | 记录一：周一，小许把钥匙放进绿色盒子。记录二：周二，小许把同一把钥匙移到书架顶层。只根据这两条记录，钥匙最后放在哪里？ | 回答书架顶层。 |
| grounded-08 | 配送单：订单A装有2本练习册，订单B装有6本练习册。请只根据配送单，指出装有6本练习册的订单编号。 | 回答订单B或B。 |
| context-01 | 把时间改成周日下午两点，活动地点不变。只写修改后的通知。 | 通知的会议时间为周日下午两点，地点为三楼讨论室。 |
| context-02 | 预算改为40元，购物数量不变。请重新列出这两个条件，不要推荐商品。 | 列出预算40元、购买两本笔记本。 |
| context-03 | 那我刚才排除的是哪种颜色？ | 回答红色。 |
| context-04 | 把“他”换成名字，其他内容不变。 | 输出小林会在周五归还这本书，或等价保留全部信息的句子。 |
| context-05 | 现在请判断上面那段代码会输出什么。 | 指出目前没有收到代码，需要用户提供代码。 |
| context-06 | 把刚才的要点改成两条编号列表，内容不变。 | 两条编号列表分别保留先核对数量、再确认包装完整。 |
| context-07 | 做沙拉时，刚才说的两种配料里应该留下哪一种？ | 保留黄瓜，排除用户不想要的花生。 |
| context-08 | 补充一下：不是6个人，而是4个人。每人仍领2张卡片，一共要准备多少张？ | 按最新人数4乘2，回答8张。 |
| tool-01 | 请使用提供的工具计算23加18。 | 输出可解析的<tool_call>块，函数名add_numbers，arguments包含整数a=23、b=18；顺序无关。 |
| tool-02 | 请调用工具查询星桥镇此刻的天气。 | 说明现有工具不能查询天气，不编造实时天气。 |
| tool-03 | {"place":"松石镇","temperature_c":18,"condition":"多云"} | 报告给定结果中的18摄氏度、多云，不改成其他天气。 |
| tool-04 | {"sku":"BK-42","available":0} | 说明可用库存为0，不能按现有库存发出1件。 |

## 4. 生成约定

正式采用贪心解码：每一步对 logits 做 argmax，等价于选择当前概率最大的 token，不进行随机采样。它不保证整条输出序列的联合概率最大。temperature、top_p 等采样参数不参与这条路径，不靠调温度让两边产生不同设置。

两边均固定：

- 相同 system、历史、工具声明和 tokenizer 模板；多轮历史是人工固定的输入，不分别采用两边之前生成的不同回复。
- add_generation_prompt=true、open_thinking=false，当前模板预填空 think 区域，模型接着生成正文。主对照不测主动展开思考的质量，工具题同样如此。
- max_new_tokens=256、总长度预算 768、EOS ID=2；greedy、batch=1、model.eval()、关闭梯度。
- 同一本机 CUDA、FP32、关闭 autocast/TF32、确定性算法及相同模型代码；CPU threads=1。
- 只改变模型权重，不给 S1 额外提示、重试机会或不同停止规则。

现有 greedy_generate 作为解码核心，不替调用者设置 model.eval()、验证总长度或生成完整评测记录。[evaluate_sft.py](../../../implementation/scripts/evaluate_sft.py)已补这些外层职责，并复用 evaluate_nll；不改变数据合同或原解码实现。正式运行分支尚未用完整 B0 验证，不能仅凭入口存在就宣称评测完成。

## 5. 输入检查

检查入口为 [inspect_sft_eval_draft.py](../../../implementation/scripts/inspect_sft_eval_draft.py)。它不加载模型、不计算 NLL、不生成回答。为了检查消息协议，会在副本末尾附一条占位助手消息使交互闭合，然后删除占位后才渲染真实生成提示；占位文本不进入评测输入。

在 implementation 目录运行；已有报告不能覆盖，复查需换 output 名：

```powershell
uv run --locked python scripts/inspect_sft_eval_draft.py `
  --cases ../experiments/sft_mini_v1/evaluation_recipe/probes.draft.jsonl `
  --config ../experiments/sft_mini_v1/evaluation_recipe/evaluation_config.draft.json `
  --output ../experiments/sft_mini_v1/evaluation_recipe/inspection.rerun.json
```

实际扫描训练 785,377 条、验证 2,048 条，边扫描边核对冻结文件 hash、行数和字节数；将每题当前及历史 user 正文与这两份数据按 NFKC/casefold/空白合并规则比较，精确重叠均为 0。未做语义相似检索、预训练语料污染或外部 benchmark 清查，不能把“精确不重合”说成模型一定没见过题型或知识。

实际最长提示 373 token，预留 256 后最长总预算 629，小于 768；所有题目协议检查通过。已检查的输入 hash 只标识这版草案，不是最终冻结或批准。

```powershell
uv run --locked pytest tests/test_sft_eval_draft.py -q
```

2 项定向测试通过：固定多轮和工具上下文下，修改题号/评分检查点不改变提示与 token，原记录不变，协议占位不进入提示，空 think 生成前缀正确。没有用测试数代替题目或评分质量审核，本次未重跑全项目回归。

### 5.1 评测入口与只读加载检查

入口先验证输入检查报告与当前草案的 hash，核对冻结数据，再验证原生导出包。它使用当前且与导出包一致的模型源码，按包内配置构造模型，严格加载 FP32 参数并核对完整状态摘要和 embedding/lm_head 共享；不恢复 optimizer、scheduler 或预训练进度。包内 tokenizer 与数据配方也必须一致。

上述运行条件随后已写入正式评测合同：生成 batch=1，NLL batch=4、全 2,048 条。若 CUDA 不可用，入口报错，不悄悄切换 CPU。下面命令和报告保留冻结前的只读检查阶段，不是正式 B0 运行。

在 implementation 目录执行只读检查；复查要给新 output-dir，不覆盖已有目录：

```powershell
uv run --locked python scripts/evaluate_sft.py --check-only `
  --config ../experiments/sft_mini_v1/evaluation_recipe/evaluation_config.draft.json `
  --cases ../experiments/sft_mini_v1/evaluation_recipe/probes.draft.jsonl `
  --inspection ../experiments/sft_mini_v1/evaluation_recipe/inspection.draft.json `
  --bundle exports/pretrain_mini_one_epoch_v1_fp32_v2 `
  --device cuda `
  --output-dir ../experiments/sft_mini_v1/evaluation_recipe/readiness_cuda_fp32_rerun
```

实际[加载检查结果](readiness_cuda_fp32_v1/result.json)通过：RTX 4060 Ti，63,912,192 个参数，完整状态摘要匹配，36 条输入预算通过，未生成答案、未计算 NLL。[manifest](readiness_cuda_fp32_v1/run_manifest.json)保存运行模式、输入/代码 hash 和最终状态。

首次在测试与检查并行期间，第三方依赖导入报 MemoryError，发生在模型加载和 manifest 创建之前，不是显存 OOM。定向测试结束后串行重试成功，没有改依赖、重装环境或关闭其他用户进程。导入期错误目前由本页人工记录；正式后台启动仍需外层保存 stderr 和退出码，不能假定进程内部 manifest 能覆盖导入之前的失败。

### 5.2 生成记录与验证边界

正式运行分支会逐题渲染输入，检查 prompt+生成预算，调用既有 greedy_generate；额外检查参与选词的 logits 有限。每题保存 input_ids、generated_ids、未清理文本、展示文本、EOS/length_limit 原因和耗时，评分状态为 pending_manual_review。达到 256 不自动失败，也不按展示文本是否有标记判断 EOS。

之后复用 SFTDataset、SFTCollator 和 evaluate_nll 计算完整验证 NLL，核对 833,052 个有效目标，检查模型状态未变。文件分别为 generations.jsonl、validation.json、result.json 和 run_manifest.json。失败会保留已有逐题记录并标记 failed，不将部分输出当作全量结果。

旧草案即使传 --run 也会被拒绝，不能把只读检查成功当成运行授权。正式 B0 使用新的冻结文件，第 7 节记录冻结与启动方式，不修改旧草案来伪装已批准。

定向检查命令为：

```powershell
uv run --locked pytest tests/test_sft_evaluation.py tests/test_sft_eval_draft.py -q
```

6 项通过：既有两项提示隔离测试，以及新入口的 EOS/长度上限记录、非有限 logits 与超预算拒绝、模式恢复、草案运行拒绝。测试用小模型，不把其输出当作 B0 能力结果；完整 --run 分支仍需正式基线验收，本次未重跑全项目测试。

另做[真实 GPU 两 token 冒烟](gpu_smoke_v1.json)：采用不属于 36 条题目的独立运行检查输入，在 FP32 下实际 forward/贪心生成，生成 2 个 token 后按预算停止，完整模型状态未变。只验证 GPU 执行链路，不评分、不计算 NLL，不代表 256 token 长回复的耗时或质量已验收。

## 6. 验收草案

### 6.1 工程完成与能力改善

实验完成不等于模型变好。工程完成要求：按之后确认的训练预算执行，数据/模型/代码身份匹配，训练和验证无未处理的非有限数值，checkpoint 与交付检查通过，B0/S1 在相同规则下完整评测，原始输出和评分可追溯。即使能力没有改善，也应如实完成分析，不把训练跑完写成效果成功。

能力判断采用两条分开的证据，不合成加权总分：

| 证据 | 怎样比较 | 能说明什么 |
| --- | --- | --- |
| 验证 NLL | 相同有效助手目标上比较 S1 与 B0；同时记录绝对差和相对变化 | 是否更贴近本轮验证数据的目标分布，不单独证明助手能力 |
| 32 条核心生成 | 统计双方通过数、正向翻转 G、负向翻转 R，以及净变化 G-R | 本批固定诊断题的任务完成变化，不推广为通用能力提升 |
| 4 条工具观察 | 单独逐题检查调用与给定结果解读 | 仅补充工具文本行为，不抵消核心题退步，也不是端到端 Agent 成功率 |

仅在双方判定明确的配对中统计 G/R，不把 uncertain 强行当错或当对；同时保留全部固定分母的通过率上下界。若所有核心题均有明确判定，S1 通过数减去 B0 通过数等于 G-R。

### 6.2 结果解释与回退分析

NLL 下降、核心任务也净改善，才具有两方面同向的证据；是否达到本轮目标，还须达到训练前确定的提升幅度和可接受回退上限。NLL 下降但生成不改善，只能说明拟合改善；生成改善但 NLL 上升，应拆开解释，不能称为两项都改善。两者都无改善则记录目标未达成，不补跑到出现好结果才汇报。

每一条负向翻转都保留原始两份输出和理由。尤其单列缺少材料却编造答案、无工具却编造调用/结果、改动用户已给事实等问题；即使净通过数增加，也不能用总分覆盖这些退步。终止、重复和格式异常分别报告，遇到 EOS 不自动获得任务通过。

工程故障首先修复实现并按新身份重验；效果未达标则先分析错误，再决定新的实验。不在原实验里偷偷更换提示、生成预算、评分规则或挑选更有利的 checkpoint。任何后续筛选必须披露，不能仍称为原定最终 checkpoint 的无选择评测。

### 6.3 基线校准与冻结顺序

当前还不知道 B0 在 32 条核心题上能过几条，因此不先凭空指定“必须提高 10 个百分点”。固定题目下的上限也不同：若 B0 已过 30 条，最多只能增加 2 条。

1. 先审定并固定题目、逐题评分、提示、生成规则、模型起点和运行条件。
2. 用上述条件完整运行 B0，记录 NLL、通过数、各类别表现和耗时，不据此把题改得更有利。
3. 在尚未开始训练、更未查看 S1 的前提下，结合基线剩余空间和预算，明确最低净改善题数、允许的负向翻转上限与 NLL 判定边界，并写入训练实验的最终验收合同。
4. 再确定训练强度、短跑与正式执行；S1 必须沿用 B0 的评测输入与运行规则。

这里分开的是“评测输入/执行规则固定”和“结合 B0 制定训练前目标”，不是允许看到 S1 后移动标准。若 B0 暴露题目歧义、评分错误或入口故障，应明确修订版本并重跑受影响的基线；不能悄悄只修 S1。

评测输入、评分和执行规则已冻结，数值能力目标仍待 B0 校准，数据合同不变。B0 启动不代表 SFT 已开始或助手能力已提升。

## 7. 正式冻结与 B0 启动

### 7.1 冻结与检查

用户授权后，由 freeze_sft_evaluation.py 生成正式配置、原字节复制题目、重新检查正式输入，再写接受记录和最后的 evaluation_freeze.json。题目与检查点没有改变；新增明确的是执行条件及评测授权，不含训练授权。

工作目录为 implementation；以下为本轮实际冻结命令，已有输出不可覆盖：

```powershell
uv run --locked python scripts/freeze_sft_evaluation.py `
  --recipe-dir ../experiments/sft_mini_v1/evaluation_recipe `
  --bundle exports/pretrain_mini_one_epoch_v1_fp32_v2 `
  --authorize-evaluation
```

启动前核对冻结输入、执行源码、依赖版本和 B0 原始包身份：

```powershell
uv run --locked python scripts/freeze_sft_evaluation.py `
  --check ../experiments/sft_mini_v1/evaluation_recipe/evaluation_freeze.json
```

本轮冻结 SHA256 为 f3b5bc996a4f465ae967565b1d54317b5c86888992a3406aeb6410a401cbe876。这个清单固定 B0 来源供追溯，后续 S1 仍允许权重不同，但共享架构和评测规则不得改变。

### 7.2 核心运行命令与外层日志

冻结检查通过后，核心等价命令为：

```powershell
uv run --locked python scripts/evaluate_sft.py --run `
  --config ../experiments/sft_mini_v1/evaluation_recipe/evaluation_config.json `
  --cases ../experiments/sft_mini_v1/evaluation_recipe/probes.jsonl `
  --inspection ../experiments/sft_mini_v1/evaluation_recipe/inspection.frozen.json `
  --bundle exports/pretrain_mini_one_epoch_v1_fp32_v2 `
  --device cuda `
  --output-dir ../experiments/sft_mini_v1/results/b0_v1
```

实际外层 PowerShell 使用当前 .venv Python、-X utf8 -u，顺序执行冻结检查与核心入口，并捕获 stdout、stderr 和退出码。启动记录保存在[launch.json](../results/b0_v1_launcher/launch.json)，结束时写 exit.json；因此即使导入期来不及建立内部 manifest，也有外层证据。已有结果路径拒绝覆盖，不直接重跑这个目录。

生成先逐题落 generations.jsonl，然后计算完整 NLL。只有 run_manifest 完成、外层退出码成功、数量与身份检查通过，才称为完整生成/NLL 基线；人工评分另列状态，不因程序结束就自动得到通过率。

### 7.3 实际结果

B0 正常结束、退出码0，36条记录完整。验证 NLL=2.3954561616642405，2,048条、833,052个有效目标；生成耗时274.13秒，NLL约60.38秒，含启动检查的墙钟约359.51秒。36条均达到256新token上限，EOS提前停止为0。原始输入token与冻结检查逐题一致、参数未变门禁和数据/评测事后身份检查均通过，详见[结果报告](../results/b0_v1/README.md)与[完整性复核](../results/b0_v1/verification.json)。

原始程序状态 evaluation_completed_scoring_pending 是采集结束时的快照，保留不改。随后另存[逐题初评](../results/b0_v1/scoring_v1/README.md)：核心3/32、工具0/4；通过的三条均重复，grounded-01还暴露了未覆盖的错误时间，不能把检查点通过解释为整体回答正确。评分只由助手完成，尚未独立复核。后续如需修补，应保留V1、明确新版本，并对B0/S1一致执行。

运行入口定向8项测试通过（8.53秒）；本轮评分聚合定向2项测试通过，检查来源、冻结检查点、证据引句与汇总一致性，不自动验证语义判分。本次未重跑全项目。
