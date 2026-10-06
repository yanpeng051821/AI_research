# MiniMind 第一轮理论笔记

这是第一轮围绕 MiniMind 整理的 LLM 从 0 到 1 理论笔记，也保留了当时的历史实验记录与实践待办。

当前进入第二轮实践：自己实现核心组件、训练工程和各阶段数据处理，并亲自运行实验。
本轮的周安排、验收与状态统一维护在 [实践路线](../PRACTICE_ROADMAP.md)，下方旧周文档作为理论参考。

## 核心入口

- [实践路线](../PRACTICE_ROADMAP.md)：第二轮实践主线、18 周安排、数据合同与独立验收。
- [`minimind_learning_plan.md`](minimind_learning_plan.md)：第一轮学习计划与知识体系。
- [`minimind_12_week_roadmap.md`](minimind_12_week_roadmap.md)：第一轮 12 周路线图与历史进度。
- [`experiments/`](experiments/README.md)：实验记录与模板。

## 周学习文档

| 阶段 | 主题 | 文档 |
|---|---|---|
| Week 01 | 项目地图 | [`week_01_project_map.md`](week_01_project_map.md) |
| Week 02 | Tokenizer 与 Dataset | [`week_02_tokenizer_dataset.md`](week_02_tokenizer_dataset.md) |
| Week 03 | 模型前向传播 | [`week_03_model_forward.md`](week_03_model_forward.md) |
| Week 04 | Attention、RoPE 与 KV Cache | [`week_04_attention_rope_kvcache.md`](week_04_attention_rope_kvcache.md) |
| Week 05 | Pretrain 训练循环 | [`week_05_pretrain_training_loop.md`](week_05_pretrain_training_loop.md) |
| Week 05.5 | Pretrain 实践 | [`week_05_5_pretrain_practice.md`](week_05_5_pretrain_practice.md) |
| Week 06 | SFT 训练循环 | [`week_06_sft_training_loop.md`](week_06_sft_training_loop.md) |
| Week 06.5 | 训练代码拆解 | [`week_06_5_training_code_dissection.md`](week_06_5_training_code_dissection.md) |
| Week 07 | 推理与采样 | [`week_07_inference_sampling.md`](week_07_inference_sampling.md) |
| Week 07.5 | 实验工程 | [`week_07_5_experiment_engineering.md`](week_07_5_experiment_engineering.md) |
| Week 08 | LoRA | [`week_08_lora.md`](week_08_lora.md) |
| Week 09 | 知识蒸馏 | [`week_09_distillation.md`](week_09_distillation.md) |
| Week 10 | DPO | [`week_10_dpo.md`](week_10_dpo.md) |
| Week 11 | GRPO | [`week_11_grpo.md`](week_11_grpo.md) |
| Week 11 补充 | PPO 前置知识 | [`week_11_ppo_prerequisites.md`](week_11_ppo_prerequisites.md) |
| Week 12 | Agentic RL | [`week_12_agentic_rl.md`](week_12_agentic_rl.md) |
| Week 12 实践 | Agentic RL 待办 | [`week_12_agentic_rl_practice_todo.md`](week_12_agentic_rl_practice_todo.md) |

学习主线是：

```text
数据 -> tokenizer -> model -> loss -> training loop -> checkpoint -> inference -> evaluation
```
