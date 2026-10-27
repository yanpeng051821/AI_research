# 中英文模型卡更新

> 2026-10-27：模型卡默认显示英文，顶部链接可查看中文，中文版也可返回英文。本次仅更新说明文件和清单，不改变训练、权重、配置、tokenizer 或运行代码。

## 1. 页面入口

- [English：默认模型卡](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain)。
- [简体中文：README_zh.md](https://huggingface.co/yanpeng051821/minimind-dense-64m-mini-pretrain/blob/main/README_zh.md)。
- 文档更新后的 HF revision：`9e268f1bf9d74c087168ded4f6c461c281d6413b`。
- 首发且完成全部下载与推理核对的父 revision：`f764fe9f62c864caa5510fc73a5e8cd5af6cef07`。

这里使用 Markdown 导航链接，不是新增 HF 原生语言设置。英文说明保留三条中文输入输出原文，观察结论译为英文，不把翻译结果冒充模型生成的英文。

## 2. 本次修改

只向现有模型仓库提交三个文件：

1. README.md：英文首页，完整保留配方、指标、限制、来源和许可不确定性。
2. README_zh.md：首发中文版增加返回英文的导航，其余原文保留。
3. export_manifest.json：记录两份模型卡的新身份与文档更新来源，避免模型卡改了仍沿用旧文件哈希。

英文源稿见 [model_cards/README.md](../model_cards/README.md)。实际提交副本与清单位于本目录 payload/，[upload.json](upload.json) 记录提交，[verification.json](verification.json) 记录远端核对。首次发布记录和旧清单不改写。

## 3. 验证范围

执行入口为 [update_model_card_languages.py](../../../../implementation/scripts/update_model_card_languages.py)。更新前确认远端 main 与记录的父 revision 一致，提交时使用 parent_commit 防止覆盖并发改动。英文与中文 metadata 相同，关键数值、四个 SHA256、运行命令及三条原始生成输出均核对保留。

更新后匿名重新下载三个改动文件，字节与本地一致；比较前后远端的 Git blob、大小及 LFS SHA256，其他 32 个已有文件身份未变，只新增 README_zh.md。模型权重 SHA256 仍为 `3f1d1e6f585d516149d62d33670ef20cb29caa11c1759786586879e19fa3cc22`。

两个网页均实际返回 HTTP 200，能看到中英文导航及 Markdown 内容。新增 3 项文档更新测试通过，全项目本次回归 350 项通过。此次没有重新训练、转换权重、完整下载大权重或新增推理评测；原父版本的模型下载加载证据继续适用于未变的模型文件，不把文档验证冒充新的能力验证。
