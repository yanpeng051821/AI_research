# HF 公开发布与下载验证

> 2026-10-27：用户确认发布研究 checkpoint，公开上传及固定 revision 下载加载验证完成。数据和权重许可范围仍未全部确立，不宣称可无条件商用；完整恢复文件的独立备份仍未验证。

后续仅将模型卡改为英文首页并提供中文链接，见[中英文更新记录](../bilingual_v1/README.md)。本页继续记录原始发布 revision 与当时的完整下载加载验收，不改写历史清单。

## 1. 发布入口

| 项目 | 实际版本 |
| --- | --- |
| 模型仓库与模型卡 | [yanpeng051821/minimind-dense-64m-mini-pretrain](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain) |
| 已验证 HF revision | f764fe9f62c864caa5510fc73a5e8cd5af6cef07 |
| 固定发布文件 | [该 revision 的文件目录](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain/tree/f764fe9f62c864caa5510fc73a5e8cd5af6cef07) |
| 对应核心代码、工具与配方 | [GitHub 34a26c3](https://github.com/yanpeng051821/AI_research/tree/34a26c3b33ed91486343ac11964eac317e8ed72b) |
| 权重 | 与本地原生 FP32 v2 完全相同，SHA256 为 3f1d1e6f585d516149d62d33670ef20cb29caa11c1759786586879e19fa3cc22 |
| 公开包 | 33 个清单内文件，含 manifest；HF 另有仓库自动管理的 .gitattributes |

模型卡第 4.3 节记录学习者亲自运行的三条原始续写，标注 CPU FP32、贪心、32-token 上限和用户报告来源；不把它们写成独立测试或准确率。

## 2. 发布内容与证据

公开包由已验证 v2 的权重、配置、tokenizer 和运行代码构成，另加入正式模型说明、训练配置、核心合同、无本机路径的数据配方摘要、验证历史、学习曲线、成本、人工抽查及上游许可说明。原始训练 JSONL、完整 optimizer/scheduler/RNG checkpoint、凭据与私人画像不在上传白名单中。

入口为 [publish_pretrain_bundle.py](../../../../implementation/scripts/publish_pretrain_bundle.py)。凭据只从用户授权的本地文件读入内存，不写进配置、公开包或报告；按实际文件白名单上传，拒绝覆盖已有模型仓库。发布副本单独位于 implementation/exports/pretrain_mini_one_epoch_v1_public_v1，不改写旧 v2。

- [preparation.json](preparation.json)：准备时的输入身份、扫描与用户授权；not_yet_performed 是该阶段的历史状态。
- [export_manifest.json](export_manifest.json)：公开包清单，与 HF 上的清单字节相同。
- [upload.json](upload.json)：实际仓库、提交 revision 和上传结果；其中下载待验证状态为当时状态。
- [download_verification.json](download_verification.json)：最终下载、文件核验与新进程加载结果。

用户批准的是带已披露限制的研究发布，不是确认所有第三方权利已完全澄清。HF 未设置虚构的统一权重 license，附带的 MiniMind Apache-2.0 文本仅说明上游代码来源，不能冒充本权重的整体许可。

## 3. 回下载验证

上传后以 f764fe9f62c864caa5510fc73a5e8cd5af6cef07 为固定 revision，从新的临时缓存强制下载 33 个公开文件，再执行：

1. 逐项核对公开清单、文件大小和 SHA256，不能只看到文件名就算成功。
2. 在新进程离线使用下载包的 runtime、配置、tokenizer 与权重，不借用当前项目中的模型对象。
3. 与已验证本地 v2 比较全部状态摘要、共享 Parameter、两条短前缀的输入和 logits，以及一次最多 8 token 的贪心续写。
4. 独立执行下载包续写命令，检查正常退出和 UTF-8 输出。

结果通过，logits 最大绝对差为 0。另以无账户凭据方式确认仓库公开可访问，远端模型卡与本地发布副本一致。整个过程仍使用本机同一依赖环境的 CPU FP32，不是另一台机器的干净安装、跨设备等价或能力基准。

发布前全项目回归 347 项通过。代码与证据已推送，原始 E 盘恢复文件未改写；HF 推理包不是完整断点恢复备份。

## 4. 复练命令

在 implementation 中下载固定版本，不需要传入账户 token：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yanpeng051821/minimind-dense-64m-mini-pretrain', revision='f764fe9f62c864caa5510fc73a5e8cd5af6cef07', local_dir='exports/hf_minimind_pretrain', token=False)"
```

再从下载包推理：

```powershell
.\.venv\Scripts\python.exe -I -B exports/hf_minimind_pretrain/load_model.py `
  --prompt "春天的风吹过田野，" --max-new-tokens 32
```

这条复练命令是供使用者重做的入口；本次真正执行的强制回下载与对照结果以 download_verification.json 为准。
