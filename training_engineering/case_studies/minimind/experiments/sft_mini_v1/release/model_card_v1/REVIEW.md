# SFT模型卡草稿核对

## 1. 当前产物

2026-11-01已完成英文默认卡[README.md](README.md)、[中文卡](README_zh.md)和单轮聊天辅助脚本[chat.py](chat.py)。本目录是模型卡与示例草稿，不是已组装的完整上传包；没有复制权重、改原模型包、创建HF仓库、上传或推送Git。

模型卡按“概况 → 数据与监督 → 训练配方与成本 → 评测 → 使用 → 导出与复现 → 用途限制 → 来源许可”组织。相同事实在中英文保持一致，切换链接只改变文档语言，不改变模型行为或承诺英文能力。

## 2. 已核对内容

- 模型起点、参数量、架构、训练/验证数量、有效目标、优化参数和最终状态依据既有合同及真实产物，不使用官方模型成绩。
- 核心题明确为0确定通过、31失败、1待裁定；工具0/4，NLL下降和EOS改善不被改写为任务成功。语义评分仍是助手初评。
- 身份示范14条的有限改写与3条身份补测失败分别说明，不宣称模型会正确自我介绍。
- YAML使用真实已发布B0作为base_model、finetune关系及zh语言；未自动选择整体/权重license或冒充Transformers适配。格式依据[HF模型卡文档](https://huggingface.co/docs/hub/model-cards)。
- 发布草稿只链接现有上游、B0及Git项目入口，不编造S1地址或未公开的固定提交。SFT源码/证据的公开固定链接留到授权推送后再填。
- 现有数据集卡的混合许可和此前来源审查的不确定性保留，不将“研究用”当作补足再分发授权。

## 3. 用法验证

新增8项定向测试通过，见[local_tests.xml](local_tests.xml)：核对双语metadata、实际NLL/状态摘要、聊天模板与已有身份诊断input_ids逐项一致，以及上下文边界和非法预算拒绝。未重跑全项目。

在已有本地依赖环境中，独立CPU进程实测：

```powershell
# 从implementation目录执行；模型包保持原样，显式指定它的位置。
.\.venv\Scripts\python.exe -I -B `
  ../experiments/sft_mini_v1/release/model_card_v1/chat.py `
  --bundle-dir exports/sft_mini_one_epoch_v1_fp32_v1 `
  --prompt '你是谁？' --max-new-tokens 2
```

退出0，输出“作为一个”。这是加载、模板和短生成冒烟，不是正确身份回答或256-token能力评测。模型卡给出的包内默认路径需要在未来组装时将chat.py与原生包放在同一目录，不能在本草稿目录省略bundle-dir直接运行。

原生S1包的22项清单hash全部仍匹配，export_manifest SHA256仍为22097e902d69e9b86adea833f96db6e8c9d61fc861ea96ca40d0f757e338e354。新增脚本和文档没有被写进旧清单冒充原包内容。

## 4. 发布边界

本目录未来拟上传的仅README.md、README_zh.md、chat.py；REVIEW.md、draft_manifest.json、测试XML和缓存属于本地控制文件，不应直接整目录上传。公开发布时另组装带适用来源声明、配方及证据的payload，并生成新清单，不能覆盖已核验的原始导出包或沿用旧清单。

仍需核对：中英文措辞、待裁定评分、来源/权重再分发及原创条款、独立S1仓库名、实际可访问Git提交。获得上传确认并解决相应发布项后，再上传与固定revision下载重载。当前不宣称这些步骤已完成。

草稿与证据摘要见[draft_manifest.json](draft_manifest.json)，仅作为本地一致性记录，不是法律审查、模型能力认证或数字签名。
