# 发布前审查与模型卡

> 用户于 2026-10-27 确认发布研究 checkpoint，HF 公开上传及固定 revision 下载验证已完成，见[实际发布记录](public_v1/README.md)。许可不确定性保留，不标为授权范围已全部澄清；原 v2 包、冻结配方和原始运行证据保持不变。

## 1. 当前结论

训练、分析、归档及本地模型导出验证已经完成。现在需要把这些事实转成对外可理解的模型说明，同时明确哪些材料可以分发、哪些结论不能承诺。工程通过和发布授权是两回事，后者仍有未决项。

| 项目 | 结果 |
| --- | --- |
| 模型卡 | [MODEL_CARD_DRAFT.md](MODEL_CARD_DRAFT.md)，已写阶段、配方、结果、加载方法、限制及用户亲自运行的三条续写 |
| 实际模型 | 沿用本机原生 FP32 v2 包，未改写、未上传 |
| 上游代码 | 固定 MiniMind 提交，根许可证为 Apache-2.0 |
| tokenizer | 固定提交的 JSON 内容相同，仅换行格式有差异 |
| 训练数据许可 | 数据集同时标注 Apache-2.0、CC-BY-NC-2.0，mini 文件及权重再分发条件尚未明确 |
| 本项目许可 | GitHub 未识别根许可证；未替用户选择原创代码或权重许可 |
| GitHub | 核心实现、工具及配方已推送到 34a26c3；实际发布记录随后回写 |
| HF | [研究模型](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain) 已发布，revision f764fe9f62c864caa5510fc73a5e8cd5af6cef07 下载加载核对通过 |

这些是公开声明与本地证据核对，不是法律意见；许可不明不能直接改写成“全部 Apache-2.0”，非商用标签也不是自动授权。

## 2. 审查方法

本步骤读取已有导出清单和验证报告，不重新训练或导出。入口为 [audit_pretrain_release.py](../../../implementation/scripts/audit_pretrain_release.py)，从 implementation 执行：

```powershell
.\.venv\Scripts\python.exe scripts/audit_pretrain_release.py `
  --bundle-dir exports/pretrain_mini_one_epoch_v1_fp32_v2 `
  --experiment-dir ../experiments/pretrain_mini_one_epoch_v1 `
  --card ../experiments/pretrain_mini_one_epoch_v1/release/MODEL_CARD_DRAFT.md `
  --output ../experiments/pretrain_mini_one_epoch_v1/release/release_review_v2.json
```

已有报告拒绝覆盖，重新审查应使用新版本。脚本先核对包内文件哈希及 v2 验证报告，再扫描候选包和模型卡中的常见凭据模式、本机绝对路径；之后通过公开 API 检查固定代码许可证、tokenizer 内容、固定数据集卡和本项目仓库可见性。报告保存输入、来源和工具身份，不打印匹配到的秘密值。

此扫描没有覆盖整个 Git 历史、原始训练文本、模型记忆或全部个人信息，也没有完成将要上传的其他附件审查。原始合同和日志保留历史路径，不能把它们不经检查整目录上传；需要时另做公开副本，并保留与原件哈希的对应关系。

实际检查 22 个包内文件身份和 23 份候选文本，未发现所检查的模式。新增扫描测试 2 项通过，全项目本次回归 343 项通过；通过测试不表示许可或公开发布决策已经确认。

## 3. 来源与证据

- [MiniMind 固定许可证](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE)：本次核对到 Apache-2.0，不能自动替代权重许可决策。
- [固定 tokenizer.json](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/tokenizer.json) 与 [tokenizer_config.json](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/tokenizer_config.json)：本机文件转换 CRLF 为 LF 后与源文件一致，解析 JSON 内容相同；保留实际训练哈希，不修改冻结资产来追求字节一致。
- [固定数据集卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/312afb4f76391145c6902f765bb51691c09a12f5/README.md)：数据集级声明不能直接推导为每条数据的单一许可。
- [本轮导出检查](../results/export_fp32_v2/README.md)：A/B 加载对照已通过，但不等于 HF 下载验证或发布授权通过。
- [当前机器审查报告](release_review_v2.json)：核对补入人工抽查后的模型卡；[历史 v1](release_review_v1.json) 保留当时内容身份。自动报告不替代后续用户的发布决定，也不代表许可不确定性已消失。

## 4. 发布边界

初查提出的事项及本次实际处理：

1. 许可范围仍未完全确立；用户要求以披露不确定性的研究 checkpoint 公开，不选择统一权重许可或宣称无条件商用。
2. 新建公开包，保留上游许可、来源及换行差异说明，生成独立清单，不改已验证 v2。
3. 已将工具、报告与教程推送，并把公开模型卡链接固定到实际 GitHub commit。
4. 已整理公开配方与证据，候选文本通过有限扫描；未上传原始数据、完整恢复状态或凭据。
5. HF 仓库已创建，上传和固定 revision 下载、哈希、加载及推理对照通过；最终证据见 public_v1。

用户随后已明确要求发布。实际执行使用已授权本地凭据所对应的账户，另建公开包并保留第三方声明；不自动添加宽松的整体或权重许可，也不把用户发布决定写成授权范围已完全确立。模型包约 256 MB；E 盘完整恢复文件仍须保留，HF 推理包不包含完整续训状态。
