# MiniMind Mini SFT：HF发布记录

> 2026-11-01完成上传及固定revision下载重载。此次只发布推理模型和有边界的实验证据，没有额外训练，也未提交或推送Git。

## 1. 发布入口

| 项目 | 实际值 |
| --- | --- |
| 模型仓库 | [yanpeng051821/minimind-dense-64m-mini-sft](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft) |
| 固定版本 | [dc58b3dc61b4a4039894c2a64d82ff8cba910988](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft/tree/dc58b3dc61b4a4039894c2a64d82ff8cba910988) |
| 英文模型卡 | [README.md](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft/blob/dc58b3dc61b4a4039894c2a64d82ff8cba910988/README.md) |
| 中文模型卡 | [README_zh.md](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-sft/blob/dc58b3dc61b4a4039894c2a64d82ff8cba910988/README_zh.md) |
| 发布文件 | 47个白名单文件，共256,668,187字节，约256.67MB |
| 发布清单SHA256 | 777f70145a9800600d136e3cd33cd6e796b8fc212057d34634ff1c0b46aab117 |

这是独立SFT仓库，未覆盖原预训练模型。默认模型卡为英文，顶部链接切换中文。卡片保留任务能力目标未达成、身份补测失败、评分未独立复核及许可范围不完全明确等限制，不以公开发布代替能力或授权认证。

## 2. 组包过程

发布工具为implementation/scripts/publish_sft_bundle.py。它先只读核验已评测原生S1导出包的清单和文件，复制相同FP32权重、配置、tokenizer、独立运行代码和加载入口，再加入中英文模型卡、chat.py、来源声明、上游许可证及配方/评测证据。原生导出包及冻结训练、数据、评测文件均未修改。

recipes/中的文件明确标记为去除本机路径的历史快照，并附原文件SHA256，不冒充字节一致的原合同，也不是下载后即可一键训练的完整数据包。历史快照中的未启动/待授权标记属于当时的生成状态；最终执行以本次训练验收与模型卡为准。evidence/保留固定诊断输入、B0/S1完整生成token与原文、逐点评分、验证NLL及独立身份补测。

没有上传原始/派生训练JSONL、完整恢复checkpoint、凭据、个人画像或本机绝对路径日志。对46个文本文件的凭据/路径模式扫描未发现匹配；这不是全量PII或模型记忆泄漏认证。

最初public_v1组包在上游许可证字节检查处停止：本机检出的许可证受换行差异影响，不等于已核对的原始字节。public_v2改用固定上游提交的许可证原始内容并核对既有SHA256，才完成正式组包；public_v1没有上传。

## 3. 上传与重载

先在本地新进程中验证发布包，再通过已授权.env中的凭据确认HF账号，创建此前不存在的新模型仓库，只上传白名单内文件。工具拒绝覆盖无本地创建记录的既有仓库；上传记录与下载核验分开保存，避免把上传成功误认为模型已验证。

上传后使用本次固定revision、新临时缓存和强制下载重新取回47个文件，逐个核对大小和SHA256。随后在隔离、离线的CPU FP32新进程中构造模型，验证参数摘要、embedding/LM head权重共享、两条固定输入的token/mask/logits及短贪心生成。最大logits绝对差为0，独立聊天入口短生成也通过。检查范围不包括重新执行完整CUDA评测或跨设备逐位一致性。

| 证据 | 用途 |
| --- | --- |
| [preparation.json](preparation.json) | 文件数、体积、文本扫描和清单身份 |
| [local_verification.json](local_verification.json) | 上传前独立加载与聊天入口核验 |
| [upload.json](upload.json) | 实际仓库、提交和上传白名单；其状态为上传时的历史记录 |
| [download_verification.json](download_verification.json) | 最终下载重载通过、逐文件哈希和模型一致性 |
| [publication_tests.xml](publication_tests.xml) | 10项模型卡/发布定向测试通过；未重跑全项目测试 |

因此最终状态是“已上传且固定版本下载重载通过”，不应仅凭upload.json中的download_verification_pending历史字段判断当前结果。

## 4. 使用与边界

```powershell
hf download yanpeng051821/minimind-dense-64m-mini-sft `
  --revision dc58b3dc61b4a4039894c2a64d82ff8cba910988 `
  --local-dir minimind-mini-sft
cd minimind-mini-sft
# 在合适的Python虚拟环境中安装requirements.txt所列依赖，然后执行：
python -I -B chat.py --prompt "你好，请简单介绍一下你自己。" --max-new-tokens 256
```

chat.py默认CPU FP32，使用与评测一致的中性中文system和空think前缀，提供一次助手回复；load_model.py的原始prompt接口仍用于纯续写。当前不是Transformers AutoModel适配或自动在线推理服务。模型卡中的身份说明来自实验记录，模型本身未可靠学会回答身份。

后续若更新卡片或代码，需要新发布清单及新版本检查；不要覆盖本次固定revision。完整训练checkpoint继续保存在E盘，推理包不能替代其optimizer、scheduler、sampler和RNG恢复状态。Git推送与后续实验另行确认。
