# Training Engineering Playbooks

这个目录保存已经跨案例验证过的训练工程方法，不直接保存某次实验的超参数和结果。

当前第一个真实案例仍在正式 S1 前，因此暂不把案例中的做法提前包装成通用标准。待
`qwen3-0.6b-openr1-math-sft` 完成正式训练、同合同配对评测和事故收尾后，优先提炼：

1. 数据审计、人工复核与 artifact 冻结。
2. 服务器环境和运行身份预飞行。
3. 最坏样本容量与 CUDA allocator telemetry。
4. smoke、pilot、checkpoint 和恢复验证。
5. B0/S1 同合同评测与逐样本错误分析。

每份 playbook 必须说明适用前提、不可复用的实验特定参数、最低证据包和失败门禁。

