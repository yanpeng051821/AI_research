# SFT正式训练配方与入口

> 更新：2026-11-01。正式SFT已于北京时间04:26完整结束，49,087步、785,377条和326,249,045个有效目标均对齐。工程验收与独立导出见[收尾报告](../results/final_run_v1/README.md)，能力判断见[配对分析](../results/paired_analysis_v1/README.md)。下文准备、短跑及冻结状态按各自执行阶段阅读。

## 1. 当前状态

数据、模型、评测和训练配方均已具备冻结证据。候选入口完成20步短跑后，正式入口补齐周期保存、全量验证与14小时累计活动预算保护，并在最终代码身份上重做20步及10+恢复对照，见[最终对照](../results/formal_resume_comparison_v1.json)。[启动前只读检查](../results/frozen_preflight_v1/result.json)通过后，经用户确认启动并完成整轮；[训练后只读复查](../results/postrun_frozen_check_v1/result.json)仍通过，未更改冻结身份。早期短跑与故障证据保留。

| 项目 | 当前值或状态 |
| --- | --- |
| 起点 | 自己的预训练B0，加载原生FP32参数，不继承预训练优化器 |
| 更新范围 | 全参数，唯一可训练参数放同一AdamW组 |
| 正式训练量 | 完整一轮785,377条，纯训练粗估10.2小时，活动预算14小时 |
| 精度与batch | CUDA BF16混合精度，micro-batch=4，累积4次，有效batch=16 |
| 学习率 | 1e-5，按optimizer更新余弦衰减至10%，无warmup |
| AdamW | betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，foreach/fused均False |
| 梯度 | 整个窗口按有效助手目标总数归一化，再以1.0裁剪 |
| 计划更新数 | 49,087；最后一次仅剩1条记录，不丢尾批 |
| 种子 | seed=42，sampler_seed=42，已确认 |
| 保存与验证 | 每1000步保存、每5000步全验证；开始/结束也验证，受控暂停保存 |
| 存储与保留 | E盘专用目录，keep_all，约50份/35.9GiB，初始至少40GiB可用空间 |

已验证的核心配置保留在[sft_mini_candidate.json](sft_mini_candidate.json)，原候选中的operations=null作为历史快照不改；[training_config.json](training_config.json)引用该核心并提供已批准的正式运行策略。正式入口以training_config为入口、以contract.json的identity.operations为有效运行策略，不读取原候选中的null来决定长跑。两者作用不同，不是两份并行维护的训练参数副本。配置相对路径仍相对于implementation根目录解析。

正式产物已生成：[人读合同](TRAINING_CONTRACT.md)、[展开身份合同](contract.json)、[用户接受记录](training_acceptance.json)、[最终冻结标记](training_freeze.json)。正式合同SHA256为3ac716dee78ce33d5d418e7f1f5e7e6b176dcdad6d06c52ac608322172c1abe9。目标文档作为冻结前快照绑定，之后进度只更新本README和实践文档，不修改已绑定文件。权重、日志与短跑证据继续位于results或E盘，不混入配方目录。

## 2. 入口如何接线

[train_sft.py](../../../implementation/scripts/train_sft.py)承担SFT专用编排，不改动已有预训练入口或已冻结的数据处理代码：

1. **读取与校验候选。** 检查配置结构、数值和候选状态，核对数据冻结标记、合同与V2评分补丁的SHA256，再调用既有数据冻结检查。后者核对训练/验证文件、派生产物、tokenizer与编码运行代码。
2. **绑定B0与计划。** 校验导出包的文件清单，绑定B0参数摘要和架构配置；用sampler_seed生成样本排列及摘要，计算完整一轮计划更新数，同时记录源码和依赖身份。
3. **区分准备与运行。** prepare-only到此结束，不创建模型或执行forward。若显式进入工程短跑，再复用B0评测入口的严格权重加载检查：核对当前模型代码、tokenizer、实际参数摘要和embedding/lm_head权重共享。
4. **组建SFT输入。** 创建SFTDataset、SFTCollator、StatefulSampler和DataLoader。每条记录编码为input_ids与助手labels，Collator补齐到合同长度768，并构造attention_mask；不使用PretrainDataset，不重新筛选数据。
5. **执行参数更新。** 新建AdamW和调度器。每次从DataLoader收集4个micro-batch，调用已有train_update；参数更新成功才commit样本位置。日志记录loss、有效目标数、梯度范数、实际学习率、耗时和显存。
6. **边界保存与恢复。** 工程短跑在指定边界做验证并保存完整checkpoint及receipt。恢复时加载SFT训练状态，检查合同、原计划、样本顺序与提交位置；不是仅重载B0，也不按短跑步数重新压缩学习率调度。

代码同时保留两条身份：B0代表新阶段的初始参数，resume代表已经开始的SFT训练状态。启动短跑时重新核对B0来源不等于覆盖已恢复的SFT参数；恢复加载发生在B0初始化之后。

## 3. 仅准备检查

在implementation目录执行下列命令。已有输出目录拒绝覆盖，再次检查应使用新目录名：

```powershell
uv run --locked python -X utf8 scripts/train_sft.py `
  --config ../experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json `
  --output-dir ../experiments/sft_mini_v1/results/entry_prepare_recheck `
  --prepare-only
```

此前已从旧配置路径执行准备检查，实际[结果](../results/entry_prepare_v1/result.json)为prepared_not_trained，training_started=false，候选计划49,087次更新，formal_training_authorized=false。上例使用迁移后的配置位置与新输出名，便于再次核验。[候选身份快照](../results/entry_prepare_v1/candidate_contract.json)包含配置、数据冻结引用、B0包、样本顺序、代码与依赖摘要；[manifest](../results/entry_prepare_v1/run_manifest.json)记录该次准备状态。快照只绑定当次文件，不自动适配后续代码修改。

准备检查验证的是文件、配方和来源，不会计算真实batch的loss，也没有验证真实GPU显存、BF16更新稳定性或训练速度。实际参数加载校验安排在运行分支，不能将包清单通过说成已执行模型训练。

迁移后已使用新位置再次执行准备检查，[新结果](../results/entry_prepare_recheck/result.json)通过，候选身份仍为e403c68bc3433e8071a6439179d24b6a2f14cdd9af5966ab9f5367bfaa32d914，与成功短跑一致。这验证了此次只是目录归拢，不改变训练语义；上面的输出目录现已存在，再次执行需另取新名。迁移后的入口5项定向测试通过，未重跑全项目。

## 4. 工程短跑的范围

短跑需要另外显式传入probe-updates和validation-records；前三次分配失败，第4次20步/64条验证短跑成功。入口拒绝省略运行模式，拒绝将短跑终点设为完整一轮或更远，且不会静默从CUDA退到CPU。

短跑使用候选完整一轮的49,087步调度，仅在较早边界暂停；它不是把学习率在几步内衰减完。短跑验证范围是明确指定的验证集前缀，batch=4，使用训练精度监控；这是工程诊断设置，不是已批准的正式周期验证策略，也不替代最终全2,048条FP32配对评测。

候选短跑仅在边界保存一次，checkpoint在短跑输出目录。正式入口另按已批准策略周期保存到E盘，两种模式明确分开。工程checkpoint不能作为正式续训起点；正式从B0、step0开始。

## 5. 验证与后续

入口测试使用临时微型对话与小模型，在CPU上验证连续更新和中断恢复一致，以及候选不能跑完整计划、验证子集越界拒绝、准备模式不运行模型、准备失败写failed与traceback。不把小模型测试当成真实B0或真实数据短跑，也不以测试数证明训练有效。

入口接线阶段执行`uv run --locked pytest tests/test_sft_training_entry.py tests/test_training.py tests/test_checkpoint.py tests/test_sft_data.py tests/test_pretrain_experiment.py -q`，46项通过（25.85秒），其中新增入口5项；没有重跑全项目。这是小模型测试记录，后续真实GPU短跑已另外完成。连续/恢复测试逐项核对模型、AdamW状态、调度器、样本顺序/提交位置和CPU RNG，而不只比较loss。

真实短跑中位窗口约0.748秒、峰值allocated约3.77GiB，D盘保存约1.54秒；E盘复制刷盘约22.16秒，全2048条BF16验证约40.80秒。最终正式代码的GPU连续20步和10+恢复到20步再次精确一致；另53项相关回归通过（18.65秒），覆盖尾批、周期验证/保存、超时暂停、冻结前运行拒绝及冻结文件篡改拒绝。没有重跑全项目，工程NLL变化不代表任务评测已改善。

## 6. 最终冻结与启动检查

[freeze_sft_training.py](../../../implementation/scripts/freeze_sft_training.py)核对用户接受与正式策略一致、最终代码身份上的真实恢复/稳定性证据、通过的测试XML，再绑定配置、核心候选、合同、目标快照、依赖、评测与评分补丁等，最后写training_freeze.json。文件已存在则拒绝覆盖。修改已绑定配置或代码后，应重新验证并创建新版本，不直接编辑冻结文件继续训练。

从implementation目录只读复查：

```powershell
uv run --locked python -X utf8 scripts/train_frozen_sft.py `
  --freeze ../experiments/sft_mini_v1/training_recipe/training_freeze.json `
  --check-only
```

启动前检查另存[frozen_preflight_v1](../results/frozen_preflight_v1/result.json)，状态checked_not_trained；该检查不forward、不恢复或更新参数。之后用户授权的显式--run已完整执行，结果见页首。正式输出目录必须是新目录；首次E盘checkpoint专用目录须为空，不删除用户文件。14小时是跨恢复累计的活动时间预算，在更新边界暂停，不是硬实时kill。本轮实际墙钟约10小时43分，未触发预算暂停；不得对原目录重新启动。
