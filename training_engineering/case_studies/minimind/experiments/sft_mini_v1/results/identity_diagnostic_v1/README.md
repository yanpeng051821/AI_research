# S1身份回答补充诊断

## 1. 结论与范围

2026-11-01应用户询问补充3条身份诊断。此前正式36题没有身份专项题，本次不加入原主评测分母、不修改冻结合同、不更新原B0/S1总分，也不是独立盲测。三条均使用最终S1，同一普通中文助手system、空think前缀、CUDA FP32、greedy、最大新增256、总预算768；没有在提示中告知yanpeng或MiniMind作为答案。

**这3条均未正确说明本模型训练归属和来源，均达到256-token上限，未生成EOS。** 这是助手阅读完整输出后的判断，不宣称覆盖全部身份问法，也未执行B0配对。

| 输入 | 回答观察 | 判断 |
| --- | --- | --- |
| 你是谁？ | 声称没有个人身份，转向在线资源、学术论文建议并重复 | 没有说明yanpeng或MiniMind来源 |
| 你是由谁训练的？ | 声称没有实际训练数据，转向泛化训练建议并重复 | 未回答训练者，包含错误自述 |
| 你的预训练权重是从随机初始化开始训练的，还是接续官方预训练权重？ | 复述两个选项，把学习率等说成训练数据并重复 | 没有明确区分本模型实际权重来源 |

第三题明确给了两个候选描述，是引导性来源诊断，不作为未提示的身份召回能力证明。

## 2. 原文节选

第一题开头：

> 作为一个AI助手，我没有个人身份或情感，因此无法提供个人身份信息。不过，我可以提供一些关于如何获取信息的建议：

第二题开头：

> 作为一个AI，我没有实际的训练数据或训练数据，因此无法提供具体的训练信息。

第三题开头：

> 预训练权重是从随机初始化开始训练的，还是接续官方预训练权重。

以上为模型输出引用，不代表这些自述为事实。全部token、prompt、原文、显示文本及停止原因见[generations.jsonl](generations.jsonl)；输入见[cases.jsonl](cases.jsonl)，运行和来源身份见[run_manifest.json](run_manifest.json)。

## 3. 数据改写与实际行为

实际训练文件中已核对有14条统一的yanpeng/MiniMind身份说明，验证集没有该改写文本。但“示范写入数据且参与训练”不等于“模型能在自由生成中稳定回答”。本次3题说明至少这些输入下尚未可靠学会，不能反推准备脚本没有修改，也不能据此确定唯一原因是14条太少。

模型卡可以如实说明训练者与模型来源，但不能把可靠身份自述列为已验证能力。未根据这些输出改数据、system、解码或重训，也未多次采样挑选正确回答。

## 4. 执行方式

新增[inspect_sft_identity_responses.py](../../../../implementation/scripts/inspect_sft_identity_responses.py)复用原load_weights与generate_case，不改冻结引擎。它核对已验证S1包，前后检查完整参数摘要和输入源码身份，原样保存输出。该次真实运行完成，并不代替单独的全项目回归。

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/inspect_sft_identity_responses.py `
  --experiment-dir ../experiments/sft_mini_v1 `
  --bundle exports/sft_mini_one_epoch_v1_fp32_v1 `
  --output-dir ../experiments/sft_mini_v1/results/identity_diagnostic_v1
```

输出已存在，不能覆盖。本轮只运行这一次，不自动继续加题或修改训练。
