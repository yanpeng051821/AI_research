# IFM K2 Horizon：案例总览

> 记录日期：2026-09-12
> 研究类型：前沿观察，不是复现实验
> 当前结论：高价值开放生命周期案例，等待可运行训练材料进一步公开

## 1. 案例定位

IFM 将 K2 Horizon 描述为一个包含 375B-A23B、36B-A4B、32B、7B、3.7B 和
0.9B 的模型家族，覆盖预训练、reasoning 与 agentic post-training。它对我们的主要
价值不是立刻复现，而是观察完整模型生命周期应公开哪些证据。

## 2. 官方开放声明

官方发布材料声明将为各尺寸模型提供：

- 训练数据或数据构造 recipe 与 mixture；
- 模型架构、训练代码和配置；
- 中间 checkpoint；
- 细粒度训练日志；
- 通用与专项评测结果；
- 最终模型权重。

多尺寸模型还共享核心架构、词表、训练方法、接口、评测基础设施和部署工具。0.9B、
3.7B 与 7B 因接近个人可研究的小模型范围，值得后续重点观察。

## 3. 当前证据边界

本次查阅时，`ifm-ai/xllm` 与 `ifm-ai/horizon-post-train` 仓库中的可见实现仍不足以
支持“已经可以独立运行完整训练链路”的结论。因此必须区分：

| 判断 | 当前状态 |
| --- | --- |
| K2 Horizon 模型家族与开放目标 | 有官方材料支持其声明 |
| 完整训练代码已经可审查、可运行 | 尚未验证 |
| 0.9B 的能力增益来自哪个训练阶段 | 尚未验证 |
| 可以直接复现官方 benchmark | 尚未证明 |

官方宣布“将开放”与仓库中已经存在“固定版本、可运行、能对应数据和 checkpoint 的
实现”是两个不同事实。

## 4. 复查条件

满足以下任一条件时重新检查：

1. 官方训练仓库出现可运行的固定 commit。
2. 配置、数据版本、checkpoint、日志和评测脚本能够互相对应。
3. 0.9B 某一阶段可以在个人预算下完成行为或数值验证。
4. 当前 Qwen3-0.6B SFT 工程闭环已经完成，且我们形成了针对 K2 的明确假设。

## 5. 官方来源

- [IFM K2 Horizon 官方发布文章](https://ifm.ai/blog/k2/)
- [IFM K2 Horizon 0.9B 模型卡](https://huggingface.co/IFM/K2-Horizon-0.9B)
- [IFM xllm 官方仓库](https://github.com/ifm-ai/xllm)
- [IFM horizon-post-train 官方仓库](https://github.com/ifm-ai/horizon-post-train)
