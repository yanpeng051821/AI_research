# MiniMind Mini SFT：完整一轮实验

> 更新：2026-11-01。正式训练于北京时间04:26完整结束，最终模型导出与固定FP32评测完成。助手初评显示拟合和部分停止行为改善，但任务完成未达到“比B0更好”的预期。独立SFT模型已发布HF并通过固定版本下载重载；评分仍未独立复核，尚未推送Git。

## 1. 当前结论

| 项目 | 实际状态 |
| --- | --- |
| 完整训练 | 49,087次更新、785,377条记录、326,249,045个有效助手目标；退出0 |
| 工程验收 | 日志、尾批、目标数、最终checkpoint及完整状态通过核验；原冻结身份复查通过 |
| 本地模型 | 原生FP32包独立进程加载通过；参数、固定探针logits与生成一致 |
| FP32验证NLL | B0 2.395456 → S1 1.721082，约下降28.15% |
| V2核心32题 | B0 2通过；S1 0确定通过、31失败、1待裁定 |
| 工具补充4题 | 两边均0通过；不是真实工具执行评测 |
| 停止行为 | EOS从0/36变为17/36，另19条仍达到256上限 |
| 结论边界 | 工程完成不等于能力目标达成；当前评分为助手初评，未独立人工复核 |

主要结果见[配对分析](results/paired_analysis_v1/README.md)。即使把S1唯一待裁定题算通过，也只到1/32，尚未超过B0的2/32。不要只凭NLL或EOS改善宣布成功，也不额外训练或放宽检查点来改变结论。

## 2. 阅读入口

1. [第八周教程](../../practice_notes/week_08_sft_data_and_training.md)：从监督特点、编码、数据处理到冻结、短跑、长跑与评测的完整流程。
2. [实验目标快照](EXPERIMENT_OBJECTIVES.md)：训练前接受的改善目标与限制；不回写事后结果。
3. [数据合同说明](data_recipe/DATA_CONTRACT.md)及[处理进度表](../sft_mini_data_inspection_v1/REVIEW_CRITERIA.md)：三条审核线、接受版本及未完成质量项。
4. [训练配方](training_recipe/README.md)：B0起点、优化计划、运行策略、恢复验证与最终冻结。
5. [评测配方](evaluation_recipe/README.md)：固定32+4题、FP32执行与V2补丁；原冻结文件保持不变。
6. [正式训练收尾](results/final_run_v1/README.md)：完整计数、成本、checkpoint与独立导出。
7. [S1执行报告](results/s1_v1/README.md)：原始生成、NLL、严格来源适配及复现命令。
8. [配对结果与分析](results/paired_analysis_v1/README.md)：逐题检查、确定回退、待裁定及上午核对顺序。
9. [身份回答补充诊断](results/identity_diagnostic_v1/README.md)：用户追问后另测3条，未可靠说明训练归属；不进入原36题分数，模型卡需保留此限制。
10. [中英文模型卡草稿核对](release/model_card_v1/REVIEW.md)：保留发布前草稿与当时审核范围，不作为最新发布状态。
11. [HF发布记录](release/public_v2/README.md)：独立SFT仓库、中英文模型卡、47个文件清单及固定版本下载重载证据。

## 3. 产物与存储

| 位置 | 用途 |
| --- | --- |
| data_recipe / training_recipe / evaluation_recipe | 小型规则、合同、接受及冻结文件 |
| results/b0_v1 | B0原始评测、V1历史评分和已批准V2重评分，不覆盖 |
| results/final_run_v1 | 正式运行验收、独立导出检查、模型清单副本和31项定向回归XML |
| results/s1_v1 | 完整36题输出、全验证FP32结果及助手原始判断 |
| results/paired_analysis_v1 | 来源绑定后的逐检查点评分、配对统计和分析 |
| results/postrun_frozen_check_v1 | 训练及评测后原训练冻结的只读复查，checked_not_trained |
| implementation/runs/sft_mini_one_epoch_v1及_supervisor | 原始训练日志、manifest和监督退出记录 |
| E:/AI_research_checkpoints/minimind/sft_mini_v1/formal | 50份完整checkpoint及receipt，约35.90GiB，未删除 |
| implementation/exports/sft_mini_one_epoch_v1_fp32_v1 | 本地模型专用包，不含optimizer等恢复状态 |

数据大文件仍在implementation/data忽略目录，原始数据与历史候选未删除。最终数据787,425条，训练785,377、验证2,048，暂存21,073，修订后重复15；有限身份修订14、内容修订2，仍不是全量事实验收或许可认证。数据准备依据见[第三版报告](../sft_mini_data_inspection_v1/preparation_768_v3/README.md)。

## 4. 待核对事项

结果核对关注grounded-04的确定回退、grounded-07的待裁定、地点题和指代替换的局部改善，以及工具误用和无输入编造等风险。当前不增加训练、改解码或另选checkpoint；后续实验与Git推送另行确认。用户已确认公开本轮研究模型，发布版保留负面结果和限制，见[HF仓库](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft)。

冻结文件中的未启动标记是生成该快照时的状态，不代表当前训练未发生；滚动状态以本页、实践路线及实际运行结果为准。新阶段代码接线与审核代办由助手按委托执行，不将这些工作写成学习者已独立完成。
