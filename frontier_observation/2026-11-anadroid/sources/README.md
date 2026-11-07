# AnaDroid 原始资料

首次核对：2026-11-02。更新：2026-11-03。

## 论文

- 标题：Automated Classification, Root Cause Analysis, and Repair Recommendations for Failed Mobile Testing by Specialized LLM。
- 作者：Chun Li、Fei Wang、Minxue Pan、Zhong Li、Mengliang Zeng、Bin Zhang、Xuejiao Yu、Boyun Wang、Kaijian Hua、Xuandong Li。
- 作者主页：[论文入口](https://pkun.life/papers/issta26/)。
- 下载来源：[作者提供的 PDF](https://pkun.life/files/issta26.pdf)。
- 本地文件：[AnaDroid-ISSTA2026.pdf](AnaDroid-ISSTA2026.pdf)。
- DOI：`10.1145/3832097`。
- PDF 首页标注：Proc. ACM Softw. Eng., Vol. 3, No. ISSTA, Article ISSTA006；Publication date: October 2026。
- 下载日期：2026-11-02。
- 页数：25。文件大小：3,656,036 bytes。
- SHA-256：`2c585fb4c5156ba0edc177dcb013caef4ef241a623192c87d15fb8a2ca6aede0`。

PDF 原样保存，已通过 pdfinfo 和首页文本提取核对标题、作者及文件可解析性。
版本身份以上述来源和 hash 为准，不把下载日期当作发表日期。

## 代码与数据入口

- 官方仓库：[pppppkun/AnaDroid](https://github.com/pppppkun/AnaDroid)，由作者主页链接。
- 已读取：[main 分支 README](https://github.com/pppppkun/AnaDroid/blob/main/README.md)。
- README 说明 `Code` 包含所用训练框架及相关代码，`config` 包含各模型训练脚本；
  `parse.py`、`utility.py`、`preprocess.py` 分别涉及日志和其他预处理、数据集生成。
- README 说明 `dataset/meta.xlsx` 提供应用与脚本信息，`dataset/example.xlsx` 提供
  脱敏案例示例。这里没有确认完整工业训练语料可获得。
- README 明确说明因企业缺陷数据限制，不公开专门训练后的模型权重。

以上为首次 README 核对记录。2026-11-03 已克隆并完成以下文件检查，尚未运行训练
或验证完整复现。

## 本地源码与示例核查

- 本地仓库：[AnaDroid](AnaDroid/README.md)，使用 `git clone --depth 1` 下载。
- Remote：`https://github.com/pppppkun/AnaDroid.git`。
- HEAD：`00b2b1d65fd6738bea392ebab5addcba01cc9e13`，提交日期 2026-06-26，提交说明 `init`。
- 克隆日期：2026-11-03。源码工作区干净；案例 `.gitignore` 排除该嵌套仓库，主资料库
  仅记录来源与版本。后续机器需要按该版本重建本地源码目录。
- 实际目录名是 `code/`，实际数据处理文件是 `code/preprocess_wf.py`，与 README
  中 `Code`、`preprocess.py` 的写法不同。另已定位 `code/parse.py`、`code/utility.py`、
  `code/LLaMA-Factory/` 和 `config/` 下各模型阶段脚本，尚未审计其逻辑。

### 示例数据

[example.xlsx](AnaDroid/dataset/example.xlsx) 的 `All` 工作表包含一行表头和 10 条
非空案例，共 12 列：`TestID`、`OrderID`、`Application`、`TestScriptID`、
`TestScriptName`、`PreCondition`、`Steps`、`Expect Result`、`Stack trace`、
`Classification`、`Root Cause`、`Repair Recommendation`。

已只读核对表头、记录数量和前两条案例。表中没有独立的原始日志或截图列，也没有
嵌入图片或嵌入文件；`dataset/` 当前仅有两个 Excel 文件，没有另附日志、截图文件。
这 10 条可以帮助理解记录结构与人工分析结果，但不能当作论文完整多模态输入或
284,552 条训练记录。表格原件保持不变。

[meta.xlsx](AnaDroid/dataset/meta.xlsx) 已核对表头为 `TestID`、`OrderID`、
`AnonymouslyApplication`、`AnonymouslyTestScriptID`。未统计其有效记录总数；
工作表声明的最大行数不能直接视为实际数据量。

[rag.xlsx](AnaDroid/data/rag.xlsx) 为 RAG 结果表，已确认文件可读，不是原始训练案例。
