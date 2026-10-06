# 第 1 周：数据入口与审计实践教程

本教程带你从 Python 工程开始，逐步实现一个 JSONL 审计工具：读取原始 Pretrain/SFT 数据，
检查结构，分流接受与拒绝记录，输出报告，并通过命令行返回退出码。

本轮已有结果：41 个测试通过，官方 Pretrain、SFT 各 20 条记录通过当前结构校验。
这些结果证明数据入口可用，尚不涉及 tokenizer、训练 batch 或模型训练。

## 阅读方式

首次实践按第 1–10 节推进，每节先看目的和函数合同，自己实现、测试，再对照源码。
复习时可以单独重做其中一步。本文按依赖顺序整理带练过程，命令按当前工程重建，
不是聊天逐字稿；补充的复习题不算已完成的验收。

这层审计是我们增加的工程练习。数据格式参照 MiniMind 冻结提交
`f659b55761b754d306bd140573493a6543cafd7f`，不代表官方入口自带这些审计功能。

| 顺序 | 实践内容 | 完成标志 |
| --- | --- | --- |
| 1 | 工程与环境 | 包可导入，测试可运行，张量可反向传播 |
| 2 | 单行解析 | 正常、空行、非法 JSON 有明确输出 |
| 3 | 文件读取 | 保留物理行号、路径与原文 |
| 4 | 记录校验 | 两类数据返回稳定错误码 |
| 5 | 审计流水线 | 解析与校验正确衔接 |
| 6 | 产物写入 | 接受文件、拒绝文件、统计报告一致 |
| 7 | 命令行 | 参数进入流水线，退出码返回操作系统 |
| 8 | 故障验证 | 正常、拒绝、缺失文件三条路径可观察 |
| 9 | 真实数据 | 固定版本的官方样本走完整链路 |
| 10 | 来源归档 | 样本可核验，Git 边界清楚 |

## 1. 工程与环境

### 目的

建立可导入、可测试、可执行命令的 Python 包。后续数据、模型、训练模块持续放在这套工程里。
已有工程直接进入 implementation；从零复练时使用独立练习目录，避免覆盖现有实现。

### 操作

在 implementation 下准备以下目录，两个 __init__.py 可以为空：

~~~text
implementation/
  pyproject.toml
  src/minimind_lab/
    __init__.py
    data/
      __init__.py
  tests/
~~~

pyproject.toml 的最小打包配置为：

~~~toml
[project]
name = "minimind-lab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
~~~

执行：

~~~bash
uv python pin 3.12
uv add --dev pytest
uv add numpy torch
uv sync
uv run python -c "import sys; print(sys.version); print(sys.executable)"
uv run python -c "import minimind_lab; print(minimind_lab.__file__)"
~~~

最后一条应指向当前工程的 src/minimind_lab/__init__.py。src 布局通过安装项目让解释器找到包，
不需要在测试中手工修改 sys.path。现有依赖见 [pyproject.toml](../implementation/pyproject.toml)；
已有工程重建环境优先使用 `uv sync --locked`，沿用 [uv.lock](../implementation/uv.lock)。

在 tests/test_environment.py 临时写一个函数，断言 1+1==2，执行 `uv run pytest -q`。
pytest 收集 test_*.py 中的 test_* 函数，不需要 if main。这只是环境练习，
不属于后面 41 个业务测试；确认环境后可移除这个临时测试。

再在 Python 交互环境或临时脚本中验证自动求导：

~~~python
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(device, loss.item(), x.grad.cpu().tolist())
# loss 为 5.0，梯度为 [2.0, 4.0]
~~~

### 检查

解释器来自项目 .venv，包可导入，测试可发现，张量可反向传播。本周数据审计不依赖 GPU。
MPS 用于 Mac 小实验，服务器可选择 CUDA；后续通过设备配置使用同一套实现。

## 2. 单行解析

### 目的

JSONL 每个物理行是一份独立 JSON。先处理一行，再扩展到文件，便于定位哪一行出了什么问题。

在 src/minimind_lab/data/reader.py 中实现：

~~~python
def parse_json_line(line: str, line_number: int) -> dict:
    ...
~~~

输出固定包含 line_number、record、error_code：

| 输入 | record | error_code |
| --- | --- | --- |
| '{"text": "hello"}' | {"text": "hello"} | None |
| 空行或纯空白 | None | blank_line |
| '{"text":}' | None | invalid_json |

### 实现

先用 line.strip() 判断是否空白。非空行交给 json.loads，捕获 json.JSONDecodeError，
把解析结果或错误码写入返回字典。

这里不检查 text 是否存在：[]、123、null 都是合法 JSON，但未必是合法训练记录。
JSON 语法与记录结构由不同层处理。

### 测试

创建 tests/test_reader.py，先分别测试正常、空白、坏 JSON。例如：

~~~python
from minimind_lab.data.reader import parse_json_line

def test_parse_blank_line():
    assert parse_json_line(" ", 1) == {
        "line_number": 1,
        "record": None,
        "error_code": "blank_line",
    }
~~~

执行 `uv run pytest -q tests/test_reader.py`，通过后再扩展读取。
参考：[reader 实现](../implementation/src/minimind_lab/data/reader.py)。

## 3. 文件读取

### 目的

复用单行解析器，遍历文件并保留追查错误需要的来源信息。继续在 reader.py 实现：

~~~python
def iter_jsonl(path: str | Path) -> Iterator[dict]:
    ...
~~~

需要导入 pathlib.Path 和 collections.abc.Iterator。

### 实现

用 with 打开 UTF-8 文件，通过 enumerate(file, 1) 取得物理行号。
逐条调用 parse_json_line，再 yield 一个包含以下字段的字典：

~~~python
{
    "source_path": "train.jsonl",
    "raw_line": '{"text": "hello"}\n',
    "line_number": 1,
    "record": {"text": "hello"},
    "error_code": None,
}
~~~

空行也输出，不能跳过，否则输出行号不能对应原文件。raw_line 保留原文及原本的换行。
yield 让下游逐条消费，不必把整个文件一次性放入列表。

### 测试

用 pytest 的 tmp_path 创建四行文件：合法 JSON、空行、坏 JSON、另一条合法 JSON。
检查输出长度为 4，行号为 [1,2,3,4]，原文及错误码正确。最后一行没有换行也应能读取。

tmp_path 是 pathlib.Path 类型的临时目录，用 path.write_text(..., encoding="utf-8")
准备输入即可，不需要维护固定的磁盘路径。

缺失文件测试应真正迭代生成器：

~~~python
with pytest.raises(FileNotFoundError):
    list(iter_jsonl(tmp_path / "missing.jsonl"))
~~~

只调用 iter_jsonl(...) 通常还没有打开文件。测试中转 list 便于断言，
实际流水线仍逐条消费。参考：[reader 测试](../implementation/tests/test_reader.py)。

## 4. 记录校验

### 目的

解析成功后，检查 Python 对象是否符合当前训练数据格式。创建 data/validator.py：

~~~python
def validate_pretrain_record(record: object) -> list[str]:
    ...

def validate_sft_record(record: object) -> list[str]:
    ...
~~~

返回错误码列表，空列表表示通过。不修改输入，不读写文件。

### Pretrain

按外层到内层检查，前提不成立就直接返回：

| 条件 | 错误码 |
| --- | --- |
| record 不是字典 | record_not_object |
| 缺少 text | missing_text |
| text 不是字符串 | text_not_string |
| text 为空或全为空白 | empty_text |

额外字段保留，例如 {"text":"hello","source":"wiki"} 仍通过。

### SFT

先检查外层，再遍历 conversations 中的消息：

| 条件 | 错误码 |
| --- | --- |
| record 不是字典 | record_not_object |
| 缺少 conversations | missing_conversations |
| conversations 不是列表 / 为空 | conversations_not_list / empty_conversations |
| message 不是字典 | message_not_object |
| role 缺失 / 类型不对 / 非允许角色 | missing_role / role_not_string / invalid_role |
| content 缺失 / 类型不对 | missing_content / content_not_string |
| 整条记录没有 assistant 角色 | missing_assistant |

允许角色为 system、user、assistant、tool。非字典 message 记录问题后继续下一条消息。
用 has_assistant 记录是否遇到 assistant。相同错误码只保留一次，并保持首次出现的顺序：
本轮使用 list(dict.fromkeys(issues))。

当前允许空 content，因为工具调用可能将内容放在 tool_calls。
但尚未检查工具调用合法性、对话顺序、reasoning_content 等可选字段类型；
“允许结构进入”不等于已经证明存在有效训练监督。

### 测试

在 tests/test_validator.py 使用参数化测试：

~~~python
@pytest.mark.parametrize("record, expected", [
    ({}, ["missing_text"]),
    ({"text": " "}, ["empty_text"]),
    ({"text": "hello"}, []),
])
def test_pretrain_cases(record, expected):
    assert validate_pretrain_record(record) == expected
~~~

每组参数会生成独立测试。补上重复错误码去重，以及 deepcopy 保存原输入后检查“未修改输入”的测试。
运行 `uv run pytest -q tests/test_validator.py`。

参考：[validator](../implementation/src/minimind_lab/data/validator.py)、
[validator 测试](../implementation/tests/test_validator.py)。

## 5. 审计流水线

### 目的

把 reader 和 validator 接起来，统一回答“这行接受吗，为什么”。
创建 data/pipeline.py，定义数据类型到函数的映射：

~~~python
VALIDATORS = {
    "pretrain": validate_pretrain_record,
    "sft": validate_sft_record,
}
~~~

### 单行审计

实现 audit_parsed_line(parsed_line: dict, dataset_type: str) -> dict：

1. 未知 dataset_type 抛出 ValueError。
2. error_code 非空：放入 issues，直接拒绝，不调用 validator。
3. 解析成功：按数据类型调用 validator。
4. issues 为空则 accepted=True，并保留来源、原文和行号。

第二行为空时应输出：

~~~python
{
    "source_path": "train.jsonl",
    "line_number": 2,
    "raw_line": "\n",
    "record": None,
    "issues": ["blank_line"],
    "accepted": False,
}
~~~

空行已经有明确根因；再传 None 给 validator，会产生无助于定位的后续结构错误。

### 文件审计

实现 iter_audited_jsonl(path, dataset_type)：遍历 iter_jsonl，
逐条调用 audit_parsed_line，然后 yield 结果。这一步尚不负责落盘。

### 测试

在 tests/test_pipeline.py 检查正常记录、缺失字段、未知类型，以及解析失败的短路行为。
用 monkeypatch 把 validator 临时替换成“调用就报错”的函数：

~~~python
def validator_must_not_run(record):
    raise AssertionError("validator should not run")

monkeypatch.setitem(pipeline.VALIDATORS, "pretrain", validator_must_not_run)
~~~

再输入空行解析结果，测试仍应通过，证明 validator 确实没有调用。
monkeypatch 在测试结束后自动撤销替换。

参考：[pipeline](../implementation/src/minimind_lab/data/pipeline.py)、
[pipeline 测试](../implementation/tests/test_pipeline.py)。

## 6. 产物写入

### 目的

把逐行结论保存下来，供后续程序读取、人工追查。
在 pipeline.py 实现 audit_jsonl(input_path, dataset_type, output_dir) -> dict。

### 顺序

1. 验证 dataset_type，先打开输入文件确认可读。
2. 创建输出目录，以 w 覆盖模式打开 accepted.jsonl 和 rejected.jsonl。
3. 遍历审计结果，分流记录，更新接受数、拒绝数及 errors_by_code。
4. 写 report.json，并返回同一份报告。

接受文件只写原始 record；拒绝文件写 source_path、line_number、raw_line、record、issues。
不必再写 accepted，因为该文件全部是拒绝记录。
用 json.dumps(..., ensure_ascii=False) 保留可读中文，每条记录后补换行。

### 样例

测试输入为四个物理行，第三行为空：

~~~text
{"text": "valid"}
{"source": "missing text"}

{"text": }
~~~

accepted 只有第一条，rejected 包含第 2、3、4 行。报告还包含来源与类型，其核心统计为：

~~~json
{
  "total_lines": 4,
  "accepted_lines": 1,
  "rejected_lines": 3,
  "errors_by_code": {
    "blank_line": 1,
    "invalid_json": 1,
    "missing_text": 1
  }
}
~~~

rejected_lines 按记录计数，errors_by_code 按问题类型计数。
一条记录可能有多个问题，所以错误计数之和可以大于拒绝记录数。

### 测试

读回三个文件检查内容，不能只判断文件存在。再验证两个边界：

- 输入不存在或类型不支持：抛错，输出目录不创建。
- 相同目录运行两次：旧产物被覆盖，统计不累加。

本轮曾补上“先检查输入，再创建目录”，防止缺失输入留下空产物。
这不保证中途 I/O 失败时原子回滚；当前实现仍可能留下部分输出。

运行 `uv run pytest -q tests/test_pipeline.py`，通过后进入命令行。

## 7. 命令行入口

### 目的

将审计函数变成终端命令。创建 src/minimind_lab/audit_cli.py，分三个函数实现：

| 函数 | 职责 |
| --- | --- |
| build_parser() | 定义 input_path、--dataset-type、--output-dir |
| main(argv=None) | 解析参数、调用 audit_jsonl、输出结果、返回整数 |
| entrypoint() | 用 raise SystemExit(main()) 将退出码交给操作系统 |

input_path 是位置参数，另两个选项必填；路径类型用 Path，
dataset-type 的 choices 为 pretrain、sft。main 支持传 argv 列表，方便直接测试。

### 状态

| 条件 | 输出 | 退出码 |
| --- | --- | --- |
| 完成且无拒绝行 | stdout 打印 JSON 报告 | 0 |
| 完成但有拒绝行 | stdout 打印 JSON 报告 | 1 |
| audit_jsonl 抛出 OSError 或 ValueError | stderr 打印错误 | 2 |

参数缺失或 choices 不合法由 argparse 自己报错并退出 2。
return 2 只是函数返回值，SystemExit 才将它变为进程退出码。

### 注册

在 pyproject.toml 加入：

~~~toml
[project.scripts]
minimind-audit = "minimind_lab.audit_cli:entrypoint"
~~~

执行 uv sync，然后运行 `uv run minimind-audit --help` 验证参数。
命令通过模块路径找到 entrypoint，与 shell 当前目录是否和 cli 文件同层无关。
审计入口命名为 audit_cli；后续同一个包可以另加训练、评测入口。

### 测试

在 tests/test_audit_cli.py 准备临时输入，直接调用：

~~~python
exit_code = main([
    str(input_path),
    "--dataset-type", "pretrain",
    "--output-dir", str(output_dir),
])
captured = capsys.readouterr()
~~~

分别检查 0/1/2、captured.out 和 captured.err。capsys 捕获标准输出和标准错误。
当前三个自动测试调用 main；真实终端运行再验证注册入口及进程退出码。

参考：[audit_cli](../implementation/src/minimind_lab/audit_cli.py)、
[CLI 测试](../implementation/tests/test_audit_cli.py)。

## 8. 端到端验证

### 操作

在 implementation 下手工准备 data/manual_checks/valid.jsonl，仅含 {"text":"hello"}。
再准备 mixed.jsonl，使用第 6 节的四行样例，第三行留空。这些是复练路径，不表示仓库已有对应文件。

~~~bash
uv run minimind-audit data/manual_checks/valid.jsonl --dataset-type pretrain --output-dir data/audit/manual-valid
echo $?

uv run minimind-audit data/manual_checks/mixed.jsonl --dataset-type pretrain --output-dir data/audit/manual-mixed
echo $?

uv run minimind-audit data/manual_checks/does-not-exist.jsonl --dataset-type pretrain --output-dir data/audit/manual-missing
echo $?
~~~

第三条路径应确实不存在。预期退出码依次是 0、1、2。
echo $? 紧接目标命令，否则读到的是其他命令的退出状态。

最后运行 uv run pytest -q。本轮记录为 41 passed；临时环境测试或后续新增测试会改变总数。

### 排错

| 现象 | 检查方向 |
| --- | --- |
| 无法导入 minimind_lab | src 结构、打包配置、是否执行 uv sync |
| 找不到 minimind-audit | project.scripts、函数名、是否同步安装 |
| pytest 没发现测试 | 文件和函数命名是否符合 test_* |
| 缺失文件测试不报错 | 是否真正消费了生成器 |
| 拒绝行数与错误计数不同 | 一条记录是否包含多个问题 |
| 第二次运行数量翻倍 | 是否误用追加模式 a |
| 输入不存在却创建了目录 | 输入检查与创建输出的先后顺序 |

## 9. 真实样本

### 获取

人工 fixture 通过后，运行官方真实数据。当前只需少量样本。
在 implementation 下执行以下命令，下载固定版本前 1MiB：

~~~bash
mkdir -p data/source_samples data/download_chunks

REV=312afb4f76391145c6902f765bb51691c09a12f5
BASE=https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/$REV

curl -fL --range 0-1048575 --max-filesize 2097152 "$BASE/pretrain_t2t_mini.jsonl" -o data/download_chunks/pretrain.part
curl -fL --range 0-1048575 --max-filesize 2097152 "$BASE/sft_t2t_mini.jsonl" -o data/download_chunks/sft.part

wc -c data/download_chunks/*.part

head -n 20 data/download_chunks/pretrain.part > data/source_samples/pretrain_sample.jsonl
head -n 20 data/download_chunks/sft.part > data/source_samples/sft_sample.jsonl

wc -l data/source_samples/*.jsonl
~~~

先确认下载成功、每个片段为 1048576 字节，再提取。
max-filesize 是本教程增加的下载限制，防止忽略 Range 的服务器直接返回整包。
本轮已验证两个片段均含超过 20 条完整记录；换文件后不能直接沿用这一结论。

Range 按字节截取，片段尾部可能只有半个 JSON 或半个 UTF-8 字符。
本次探测就遇到过整块解码的 UnicodeDecodeError。选取完整行后再解析，
不要使用 errors="ignore" 掩盖字符损坏。

### 审计

~~~bash
uv run minimind-audit data/source_samples/pretrain_sample.jsonl --dataset-type pretrain --output-dir data/audit/pretrain
uv run minimind-audit data/source_samples/sft_sample.jsonl --dataset-type sft --output-dir data/audit/sft
~~~

本次两份报告均为 total=20、accepted=20、rejected=0、errors_by_code={}。
这表明现有结构校验接受了这些记录，不证明事实正确、无重复、无泄漏。

### 字段观察

打开 Pretrain 的一条记录，确认 text 是字符串；打开 SFT 记录，沿 conversations 逐个看 message。
一个多轮对话仍是一条 JSONL 记录，每个 message 不是独立样本。

| 观察项 | 本轮结果 |
| --- | --- |
| Pretrain | 20 条记录均含 text |
| SFT 消息数 | 每条含 4–16 个 message |
| SFT 角色 | user 95 个、assistant 95 个 |
| 可选字段 | 34 个 message 带 reasoning_content |
| 工具调用 | 当前 20 条没有 Tool Call |

第一条 SFT 记录有 14 个 message，部分 assistant 消息带 reasoning_content。
role 表示说话者，content 保存消息内容，reasoning_content 单独保存推理文本。
可选字段当前原样保留，如何进入模板与监督目标将在 SFT 阶段实现。

前 20 条是顺序截取，只验证格式与链路，不能代表全量分布或工具调用兼容性。

## 10. 来源与归档

### 清单

创建 data/source_samples/manifest.json，完整内容参考
[本次来源清单](../implementation/data/source_samples/manifest.json)，至少记录：

- source_repo、source_revision：来源仓库与版本。
- byte_range、selection：下载片段和抽取方法。
- upstream_file、record_count：原文件与样本条数。
- sha256：本地文件的字节身份。

在 Mac 终端核验：

~~~bash
shasum -a 256 data/source_samples/pretrain_sample.jsonl
shasum -a 256 data/source_samples/sft_sample.jsonl
~~~

本轮 Pretrain 为 a2aa375582bfd2c0bd87be9b031e1b6e187fa48b3b034ac246b177c8319b9db8，
SFT 为 525420d6d82b484ddc8e8cb88032b89acefc060e51d31be004b8e64716007d18。
重新格式化或改变换行符也会改变文件哈希；文件名相同不代表内容相同。

### Git 边界

案例根目录的 .gitignore 统一管理 implementation，相关规则如下：

~~~gitignore
.venv/
__pycache__/
.pytest_cache/
implementation/data/download_chunks/
implementation/data/audit/
implementation/data/source_samples/*.jsonl
~~~

在 implementation 下验证：

~~~bash
git check-ignore -v data/source_samples/pretrain_sample.jsonl
git check-ignore -v data/source_samples/manifest.json
git status --short
~~~

JSONL 应被忽略，manifest 不应被忽略。第二条没有输出、退出码为 1，表示没有命中忽略规则。
代码、测试、依赖锁文件和 manifest 纳入版本管理，样本与输出留在本地。
保留 manifest 不等于保留数据本身；复现仍依赖固定版本的上游可获取或本地副本。

## 完整回放

运行 minimind-audit 后，完整调用顺序为：

~~~text
project.scripts 找到 entrypoint
  -> main 解析参数
  -> audit_jsonl 验证输入、打开输出
     -> iter_audited_jsonl 开始迭代
        -> iter_jsonl 读一个物理行
           -> parse_json_line 返回解析结果
        -> audit_parsed_line
           -> 解析失败：直接生成 issues
           -> 解析成功：对应 validator 检查字段
     -> 写 accepted/rejected，更新统计
     -> 迭代结束，保存并返回 report
  -> main 输出报告并返回 0/1；文件级异常返回 2
  -> entrypoint 用 SystemExit 设置进程退出码
~~~

新增数据类型主要增加 validator 和映射，reader 仍负责通用 JSONL 读取。
下游 tokenizer 接收 accepted 中的原始 record，而不是 rejected 中的审计包装。

## 复练与衔接

合上参考实现，解释空行为什么保留行号、解析失败为什么不进入 validator、
错误计数为什么可以大于拒绝行数，以及 return 与 SystemExit 的差别。

再独立构造一份 SFT 记录，同时触发 missing_content 和 missing_assistant，
用测试与 CLI 验证“一条拒绝记录、两个错误计数”。
进一步独立增加一条校验规则及测试，作为迁移验收；这项尚未记录为完成。

本轮代码、测试和真实样本命令由学习者在指导下完成；助手提供讲解、审查、
排错提示、下载命令、manifest 内容及运行复核。本篇教程由助手整理。

第二周继续实现 text → token IDs → labels → batch。
当前任务与官方对应关系统一维护在 [实践路线](../PRACTICE_ROADMAP.md)。
