# 原生 FP32 模型导出与本地加载验证

> 当前导出 v2，2026-10-26：最终 checkpoint 已导出，本机新进程 CPU FP32 对照和独立续写命令通过。未上传 HF，不是 AutoModel 适配，也不是完整恢复备份。

## 1. 入口与位置

完整目的、数据流、命令、共享参数处理和验证理由见[第七周教程第 11.4 节](../../../../practice_notes/week_07_pretrain_experiment.md#114-模型导出与本地加载验证)。

- [export_manifest.json](export_manifest.json)：模型包清单的原始副本，关联来源 checkpoint、合同、导出工具及包内文件身份。
- [verification.json](verification.json)：新进程参数、输出和独立续写入口的验证结果。
- 实际包：`implementation/exports/pretrain_mini_one_epoch_v1_fp32_v2/`，整个目录被 Git 忽略，尚未上传。

报告不是权重本身；实际 model.safetensors、配置、tokenizer 和运行代码必须一并保留。

## 2. 本次核验

源文件是 E 盘 checkpoint-4943.pt，SHA256 为 `8d3cb52867e0c74b35c5931bfb9c90de3a7b6589fd9466fd68d00c9593431e38`，导出前后不变。FP32 权重文件为 255,659,304 字节，整个包为 256,196,322 字节。

两侧全部状态摘要一致，独立参数数目为 63,912,192，embedding/lm_head 仍为同一 Parameter。两条右侧 padding 的短前缀产生相同的 token、mask 和 logits；一次最多 8 token 的贪心续写也相同。独立命令正常退出，UTF-8 输出可解码。

条件为本机同一第三方依赖环境、CPU、FP32、单线程、无 autocast。新进程离线加载 tokenizer，并核实使用包内模型代码；这不是干净安装环境、跨设备或跨精度验证。没有重算完整验证 NLL，也不代表聊天或其他下游能力已评测。

新增导出测试 9 项通过，本次全项目回归 341 项通过。实现、测试和执行由助手按委托完成，不记作学习者独立实现。

## 3. 版本与保存边界

[历史 v1](../export_fp32_v1/README.md) 数值验证通过，但独立打印续写时遇到 Windows GBK 编码错误。v2 在加载入口中显式配置 UTF-8，并把独立命令检查加入验收；保留 v1 原报告，不覆盖证据。修正前后逻辑模型状态摘要相同，但不要求两次序列化的文件字节或清单哈希相同。

公开审查、许可与再分发核对、模型卡、HF 上传和指定 revision 下载加载仍待完成。完整恢复状态不在此包中，E 盘原件须保留；独立恢复备份未验证。final_run_v1 里的 pending 表示当时状态，不改写历史证据，最新进度以实践路线和本报告为准。
