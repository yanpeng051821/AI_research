# 原生 FP32 模型导出与本地加载验证

> 历史导出 v1：数值对照通过，但随后单独运行续写命令发现 Windows GBK 标准输出无法编码部分生成字符。该版本不作为当前交付；修正后的模型包和证据见 [export_fp32_v2](../export_fp32_v2/README.md)。原 JSON 保留当时检查结果，不改写为后来的状态。

## 1. 入口与位置

完整目的、数据流、命令、共享参数处理和验证理由见[第七周教程第 11.4 节](../../../../practice_notes/week_07_pretrain_experiment.md#114-模型导出与本地加载验证)。

- [export_manifest.json](export_manifest.json)：实际模型包清单的原始副本，关联来源 checkpoint、合同、导出工具及包内每个文件的大小和哈希。
- [verification.json](verification.json)：新进程加载对照结果，记录条件、状态摘要、输入形状、输出一致性和未覆盖范围。
- 实际模型包：`implementation/exports/pretrain_mini_one_epoch_v1_fp32/`，整个目录被 Git 忽略，当前仍仅保存在本机。

只有这两份 JSON 无法恢复模型，实际 model.safetensors 及配置、tokenizer、运行代码必须一并保留。清单是身份核对依据，不是权重本身。

## 2. 本次核验

源文件是 E 盘 checkpoint-4943.pt，SHA256 为 `8d3cb52867e0c74b35c5931bfb9c90de3a7b6589fd9466fd68d00c9593431e38`。导出前后源文件哈希一致。模型权重为 255,659,304 字节，整个包为 256,196,147 字节。

两侧全部状态摘要一致，63,912,192 个独立参数，embedding/lm_head 仍为同一 Parameter。两条右侧 padding 的短前缀产生相同的 token、mask 和 logits；一次最多 8 token 的贪心续写也相同。

条件是同一本机环境、CPU、FP32、单线程、无 autocast。新进程使用包内模型代码且离线加载 tokenizer，不依赖项目中的训练模块；不是干净安装环境或跨设备验证。没有重算完整验证 NLL，也不代表聊天或其他下游能力已评测。

新增导出测试 8 项通过，全项目本次回归 340 项通过。实现、测试和执行由助手按委托完成，不记作学习者独立实现。

## 3. 保存边界

模型卡、许可与再分发审查、HF 上传以及指定 revision 的下载加载检查仍待完成。本地包未提交 Git 或上传 HF；Git 仅保存工具、教程和小型证据。

此包省略了 optimizer、scheduler、sampler 与 RNG，不能代替完整续训 checkpoint。E 盘原恢复文件须保留，另一份独立恢复备份尚未验证。此前 final_run_v1 中的 pending 表示当时状态，不改写历史证据；最新进度以实践路线和本报告为准。
