# Mini SFT 配对评测合同

## 1. 范围

用户授权冻结已讨论的评测输入与执行规则并运行 B0，不授权 SFT。B0 为自己的最终 mini 预训练模型；S1 必须保持相同架构、tokenizer、输入及执行规则，仅模型参数改变。本批题用于教学诊断，不是独立盲测或通用能力榜单。

## 2. 输入与执行

probes.jsonl 固定 32 条核心题、4 条补充工具题及各检查点，不改变草案题目内容。system、模板、固定历史、工具声明及思考模式统一；评分字段绝不进入提示。不执行真实工具。

本机 CUDA、FP32 参数与计算，不使用 autocast，关闭 TF32，启用确定性算法，CPU threads=1，CUBLAS_WORKSPACE_CONFIG=:4096:8。生成 batch=1、greedy、open_thinking=false、max_new_tokens=256、总预算768、EOS ID=2。无额外采样、重试或候选择优。CUDA 不可用时报错，不切换 CPU。

NLL 使用数据合同冻结的全部 2,048 条验证记录，batch=4、无 shuffle、num_workers=0，有效助手目标应为833,052，以 total_loss_sum / total_valid_tokens 汇总。两边均使用同一 FP32 路径。

## 3. 评分

各检查点 pass/fail/uncertain；任一 fail 则单题失败，全部 pass 才通过，否则 uncertain。不因 EOS 就判对，不因达到长度上限就自动判错。原始 token、文本和停止原因保留。核心与工具题分开汇报，固定分母，报告不确定数及通过率上下界；配对不确定项单列，不能硬算翻转。开放题接受语义等价表达。

人工评分在生成后进行，尚未实现自动能力判定。题目及评分不依据 B0 或 S1 结果偷偷修改；确有歧义时另立版本并重建受影响基线。

## 4. 模型与产物

B0 使用 pretrain_mini_one_epoch_v1 最终 step4943 的 FP32 导出包，原生权重与完整参数摘要按清单核验，只加载模型参数，不恢复训练状态。模型架构代码与包内版本匹配，embedding/lm_head 共享关系和 tokenizer 必须一致。评测结束参数摘要应不变。

evaluation_freeze.json 绑定配置、题目、输入检查、授权、合同、运行代码、依赖及 B0 来源。用于建立 B0 的源包在清单中固定；后续 S1 的权重允许不同，但共享评测条件不得改变，并须明确其单独来源。

正式运行保存 generations.jsonl、validation.json、result.json、run_manifest.json；外层另保存 stdout、stderr、退出码和启动记录。中断/失败产物不称为完整评测，不补挑成功样本。

## 5. 接受边界

本次冻结评测输入、评分和执行规则，不冻结训练参数或能力提升数值门槛。B0 完成并人工评分后，在任何 SFT 训练与 S1 观察之前确定最低净改善、允许退步及 NLL 判定边界，写入训练实验合同。现阶段不得宣称模型能力改善。

当前精确重叠检查未发现题目 user 文本与 SFT train/validation 重合；没有证明语义近重复、预训练污染或全部外部污染已清除。失败和异常如实记录，任何变更须版本化。
