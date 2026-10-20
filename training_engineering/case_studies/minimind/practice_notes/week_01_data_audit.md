# 第 1 周：数据入口与审计实践教程

> 教程整理：2026-10-19。目标是完成结构审计，不是训练模型；QA 统一放在末尾。

## 1. 本周任务与准备

工程的起点是一份能被解释和追查的数据文件。本周从单行 JSON 解析开始，逐层增加记录校验、文件分流和命令行，最后交付下一周要读取的 accepted.jsonl。

~~~text
原始 JSONL → reader → validator → audit → accepted.jsonl
                                     ├→ rejected.jsonl
                                     └→ report.json
~~~

完成标准：每一行都有接受或拒绝结论；坏数据能追查到原行；报告与文件内容一致；进程退出码表达成功、数据拒绝和执行失败。结构通过不证明事实正确、已经去重或没有泄漏。

### 1.1 环境与复练方式

以下命令默认在 implementation 目录运行，使用 PowerShell。已有仓库先执行 uv sync --locked，复用锁定环境。核心模块在独立练习副本中重写，不删除后续周数依赖的现有文件。本周只需要 CPU。

正文先说明操作，再提供参考实现；同一文件中的函数依次追加，导入放到文件顶部。参考代码用于核对，不要求一开始就复制全部答案。

分步测试约定：文中的测试命令适用于已有仓库复练。从空目录重写时，先完成当前步骤的函数和最小验证，再运行对应测试；不要在 reader.py 尚未创建时运行 test_reader.py。pytest 的筛选选项不会跳过测试文件的导入，因此缺少后续模块时，应先用本节的小例子检查，相关依赖完成后再执行仓库测试。

### 1.2 实践顺序

| 步骤 | 交付物 | 进入下一步的条件 |
| --- | --- | --- |
| 2. 环境 | 可导入的包 | pytest 与 Python 正常 |
| 3–4. 读取 | reader.py | 行号、原文、解析结论正确 |
| 5. 校验 | validator.py | 两类格式有稳定错误码 |
| 6–7. 审计与写入 | pipeline.py | 三份产物相互对应 |
| 8–9. CLI 与故障路径 | audit_cli.py | 退出码与文件内容正确 |
| 10–11. 真实数据与归档 | 样本、报告、manifest | 来源与字节可核验 |
| 12. 验收 | 回归与调用链回顾 | 可独立重跑审计 |

数据格式参照 MiniMind 冻结提交 f659b55761b754d306bd140573493a6543cafd7f；这套审计器是教学工程新增内容，不宣称官方自带。

## 2. 工程与环境

### 2.1 目的

建立可导入、可测试、可执行命令的 Python 包。后续数据、模型、训练模块持续放在这套工程里。
已有工程直接进入 implementation；从零复练时使用独立练习目录，避免覆盖现有实现。

### 2.2 操作

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

已有仓库执行：

~~~powershell
uv sync --locked
uv run --locked python -c "import sys, minimind_lab; print(sys.executable); print(minimind_lab.__file__)"
uv run --locked pytest tests/test_reader.py -q
~~~

独立从零建包时，在最小 pyproject 基础上执行 uv add --dev pytest 和 uv add numpy torch。已有仓库不重复添加依赖，以 [pyproject.toml](../implementation/pyproject.toml) 和 [uv.lock](../implementation/uv.lock) 为准。包路径应指向本工程 src/minimind_lab，不手工修改 sys.path。

在 tests/test_environment.py 临时写一个函数，断言 1+1==2，执行 `uv run pytest -q`。
pytest 收集 test_*.py 中的 test_* 函数，不需要 if main。这只是环境练习，
不属于后面 41 个业务测试；确认环境后可移除这个临时测试。

再在 Python 交互环境或临时脚本中验证自动求导：

~~~python
import torch

device = "cpu"
x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
loss = (x ** 2).sum()
loss.backward()
print(device, loss.item(), x.grad.cpu().tolist())
# loss 为 5.0，梯度为 [2.0, 4.0]
~~~

### 2.3 检查

解释器来自项目 .venv，包可导入，测试可发现，张量可反向传播。本周数据审计不依赖 GPU。
MPS 用于 Mac 小实验，服务器可选择 CUDA；后续通过设备配置使用同一套实现。

## 3. 单行解析

### 3.1 目的

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

### 3.2 实现

先用 line.strip() 判断是否空白。非空行交给 json.loads，捕获 json.JSONDecodeError，
把解析结果或错误码写入返回字典。

这里不检查 text 是否存在：[]、123、null 都是合法 JSON，但未必是合法训练记录。
JSON 语法与记录结构由不同层处理。

### 3.3 参考实现

下面代码按本节描述写入对应文件；追加函数时不要重复覆盖之前实现：

~~~python
import json

def parse_json_line(line: str, line_number: int) -> dict:
    record = None
    error_code = None

    if line.strip() != "":
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            error_code = "invalid_json"
    else:
        error_code = "blank_line"

    return {
        "line_number": line_number,
        "record": record,
        "error_code": error_code,
    }
~~~

### 3.4 测试

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

## 4. 文件读取

### 4.1 目的

复用单行解析器，遍历文件并保留追查错误需要的来源信息。继续在 reader.py 实现：

~~~python
def iter_jsonl(path: str | Path) -> Iterator[dict]:
    ...
~~~

需要导入 pathlib.Path 和 collections.abc.Iterator。

### 4.2 实现

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

### 4.3 参考实现

下面代码按本节描述写入对应文件；追加函数时不要重复覆盖之前实现：

~~~python
from pathlib import Path
from collections.abc import Iterator

def iter_jsonl(path: str | Path)-> Iterator[dict]:
    jsonl_path = Path(path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            result = parse_json_line(line, line_number)

            yield {
                "source_path": str(jsonl_path),
                "raw_line": line,
                "line_number": line_number,
                "record": result["record"],
                "error_code": result["error_code"],
            }
~~~

### 4.4 测试

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

## 5. 记录校验

### 5.1 目的

解析成功后，检查 Python 对象是否符合当前训练数据格式。创建 data/validator.py：

~~~python
def validate_pretrain_record(record: object) -> list[str]:
    ...

def validate_sft_record(record: object) -> list[str]:
    ...
~~~

返回错误码列表，空列表表示通过。不修改输入，不读写文件。

### 5.2 Pretrain

按外层到内层检查，前提不成立就直接返回：

| 条件 | 错误码 |
| --- | --- |
| record 不是字典 | record_not_object |
| 缺少 text | missing_text |
| text 不是字符串 | text_not_string |
| text 为空或全为空白 | empty_text |

额外字段保留，例如 {"text":"hello","source":"wiki"} 仍通过。

### 5.3 SFT

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

### 5.4 参考实现

下面代码按本节描述写入对应文件；追加函数时不要重复覆盖之前实现：

~~~python
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def validate_pretrain_record(record: object)-> list[str]:

    if not isinstance(record, dict):
        return ["record_not_object"]

    if "text" not in record:
        return ["missing_text"]

    text = record["text"]

    if not isinstance(text, str):
        return ["text_not_string"]

    if text.strip() == "":
        return ["empty_text"]

    return []


def validate_sft_record(record: object)-> list[str]:
    issues = []
    if not isinstance(record, dict):
        return ["record_not_object"]

    if "conversations" not in record:
        return ["missing_conversations"]

    conversation = record["conversations"]

    if not isinstance(conversation, list):
        return ["conversations_not_list"]

    if len(conversation) == 0:
        return ["empty_conversations"]

    has_assistant = False
    for message in conversation:
        if not isinstance(message, dict):
            issues.append("message_not_object")
            continue

        # 检查role
        role = message.get("role")
        if "role" not in message:
            issues.append("missing_role")
        elif not isinstance(role, str):
            issues.append("role_not_string")
        elif role not in ALLOWED_ROLES:
            issues.append("invalid_role")
        elif role == "assistant":
            has_assistant = True

        # 检查content
        content = message.get("content")
        if "content" not in message:
            issues.append("missing_content")
        elif not isinstance(content, str):
            issues.append("content_not_string")

    if not has_assistant:
            issues.append("missing_assistant")

    return list(dict.fromkeys(issues))
~~~

### 5.5 测试

在 tests/test_validator.py 使用参数化测试：

~~~python
import pytest
from minimind_lab.data.validator import validate_pretrain_record

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

## 6. 审计流水线

### 6.1 目的

把 reader 和 validator 接起来，统一回答“这行接受吗，为什么”。
创建 data/pipeline.py，定义数据类型到函数的映射：

~~~python
VALIDATORS = {
    "pretrain": validate_pretrain_record,
    "sft": validate_sft_record,
}
~~~

### 6.2 单行审计

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

### 6.3 文件审计

实现 iter_audited_jsonl(path, dataset_type)：遍历 iter_jsonl，
逐条调用 audit_parsed_line，然后 yield 结果。这一步尚不负责落盘。

### 6.4 参考实现

下面代码按本节描述写入对应文件；追加函数时不要重复覆盖之前实现：

~~~python
from pathlib import Path
from collections.abc import Iterator
from minimind_lab.data.reader import iter_jsonl
from minimind_lab.data.validator import validate_pretrain_record, validate_sft_record

def audit_parsed_line(parsed_line: dict, dataset_type: str) -> dict:
    if dataset_type not in VALIDATORS:
        raise ValueError(f"unsupported dataset type: {dataset_type}")

    if parsed_line["error_code"] is not None:
        return {
            "source_path": parsed_line["source_path"],
            "line_number": parsed_line["line_number"],
            "raw_line": parsed_line["raw_line"],
            "record": parsed_line["record"],
            "issues": [parsed_line["error_code"]],
            "accepted": False,
        }

    validator = VALIDATORS[dataset_type]
    issues = validator(parsed_line["record"])

    accepted = len(issues) == 0

    return {
        "source_path": parsed_line["source_path"],
        "line_number": parsed_line["line_number"],
        "raw_line": parsed_line["raw_line"],
        "record": parsed_line["record"],
        "issues": issues,
        "accepted": accepted,
    }

def iter_audited_jsonl(path:str | Path, dataset_type: str)-> Iterator[dict]:

    for parsed_line in iter_jsonl(path):
        yield audit_parsed_line(parsed_line, dataset_type)
~~~

### 6.5 测试

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

## 7. 产物写入

### 7.1 目的

把逐行结论保存下来，供后续程序读取、人工追查。
在 pipeline.py 实现 audit_jsonl(input_path, dataset_type, output_dir) -> dict。

### 7.2 顺序

1. 验证 dataset_type，先打开输入文件确认可读。
2. 创建输出目录，以 w 覆盖模式打开 accepted.jsonl 和 rejected.jsonl。
3. 遍历审计结果，分流记录，更新接受数、拒绝数及 errors_by_code。
4. 写 report.json，并返回同一份报告。

接受文件只写原始 record；拒绝文件写 source_path、line_number、raw_line、record、issues。
不必再写 accepted，因为该文件全部是拒绝记录。
用 json.dumps(..., ensure_ascii=False) 保留可读中文，每条记录后补换行。

### 7.3 样例

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

### 7.4 参考实现

下面代码按本节描述写入对应文件；追加函数时不要重复覆盖之前实现：

~~~python
import json

def audit_jsonl(input_path: str | Path, dataset_type: str, output_dir: str | Path)->dict:
    # 检查数据类型
    if dataset_type not in VALIDATORS:
        raise ValueError(f"unsupported dataset type: {dataset_type}")

    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8"):
        pass

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "report.json"

    accepted_counts = 0
    rejected_counts = 0

    errors_by_code = {}

    with (
        accepted_path.open("w", encoding="utf-8") as accepted_file,
        rejected_path.open("w", encoding="utf-8") as rejected_file,
    ):
        for audited_line in iter_audited_jsonl(input_path, dataset_type):
            output_record = {
                key: value
                for key , value in audited_line.items()
                if key != "accepted"
            }

            serialized_record = json.dumps(output_record, ensure_ascii=False)

            if audited_line["accepted"]:
                accepted_file.write(json.dumps(audited_line["record"], ensure_ascii=False) + "\n")
                accepted_counts += 1
            else:
                rejected_file.write(serialized_record + "\n")
                rejected_counts += 1

                for error_code in audited_line["issues"]:
                    errors_by_code[error_code] = errors_by_code.get(error_code, 0) + 1

    report = {
                "source_path": str(input_path),
                "dataset_type": dataset_type,
                "total_lines": accepted_counts + rejected_counts,
                "accepted_lines": accepted_counts,
                "rejected_lines": rejected_counts,
                "errors_by_code": dict(sorted(errors_by_code.items())),
            }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report
~~~

### 7.5 测试

读回三个文件检查内容，不能只判断文件存在。再验证两个边界：

- 输入不存在或类型不支持：抛错，输出目录不创建。
- 相同目录运行两次：旧产物被覆盖，统计不累加。

本轮曾补上“先检查输入，再创建目录”，防止缺失输入留下空产物。
这不保证中途 I/O 失败时原子回滚；当前实现仍可能留下部分输出。

运行 `uv run pytest -q tests/test_pipeline.py`，通过后进入命令行。

## 8. 命令行入口

### 8.1 目的

将审计函数变成终端命令。创建 src/minimind_lab/audit_cli.py，分三个函数实现：

| 函数 | 职责 |
| --- | --- |
| build_parser() | 定义 input_path、--dataset-type、--output-dir |
| main(argv=None) | 解析参数、调用 audit_jsonl、输出结果、返回整数 |
| entrypoint() | 用 raise SystemExit(main()) 将退出码交给操作系统 |

input_path 是位置参数，另两个选项必填；路径类型用 Path，
dataset-type 的 choices 为 pretrain、sft。main 支持传 argv 列表，方便直接测试。

### 8.2 状态

| 条件 | 输出 | 退出码 |
| --- | --- | --- |
| 完成且无拒绝行 | stdout 打印 JSON 报告 | 0 |
| 完成但有拒绝行 | stdout 打印 JSON 报告 | 1 |
| audit_jsonl 抛出 OSError 或 ValueError | stderr 打印错误 | 2 |

参数缺失或 choices 不合法由 argparse 自己报错并退出 2。
return 2 只是函数返回值，SystemExit 才将它变为进程退出码。

### 8.3 参考实现

下面是 audit_cli.py 完整接线。它不增加审计算法，只连接命令行和库函数：

~~~python
import argparse
import json
import sys

from collections.abc import Sequence
from pathlib import Path

from minimind_lab.data.pipeline import audit_jsonl


def build_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimind-audit",
        description="Audit pretrain or sft jsonl datasets."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input jsonl file."
    )

    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=["pretrain", "sft"],
        help="Type of dataset to audit."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for accepted.jsonl, rejected.jsonl, and report.json."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = audit_jsonl(
            args.input_path,
            args.dataset_type,
            args.output_dir
        )
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    if report["rejected_lines"] > 0:
        return 1

    return 0


def entrypoint()-> None:
    raise SystemExit(main())
~~~

### 8.4 注册

在 pyproject.toml 加入：

~~~toml
[project.scripts]
minimind-audit = "minimind_lab.audit_cli:entrypoint"
~~~

执行 uv sync，然后运行 `uv run minimind-audit --help` 验证参数。
命令通过模块路径找到 entrypoint，与 shell 当前目录是否和 cli 文件同层无关。
审计入口命名为 audit_cli；后续同一个包可以另加训练、评测入口。

### 8.5 测试

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

## 9. 端到端验证

### 9.1 操作

在 implementation 下手工准备 data/manual_checks/valid.jsonl，仅含 {"text":"hello"}。
再准备 mixed.jsonl，使用第 7 节的四行样例，第三行留空。这些是复练路径，不表示仓库已有对应文件。

~~~powershell
uv run minimind-audit data/manual_checks/valid.jsonl --dataset-type pretrain --output-dir data/audit/manual-valid
$LASTEXITCODE

uv run minimind-audit data/manual_checks/mixed.jsonl --dataset-type pretrain --output-dir data/audit/manual-mixed
$LASTEXITCODE

uv run minimind-audit data/manual_checks/does-not-exist.jsonl --dataset-type pretrain --output-dir data/audit/manual-missing
$LASTEXITCODE
~~~

第三条路径应确实不存在。预期退出码依次是 0、1、2。
$LASTEXITCODE 紧接目标命令，否则读到的是其他命令的退出状态。

最后运行相关测试。历史第一周全仓 41 passed，后续测试已增加；不以匹配旧数量作为验收。

### 9.2 排错

| 现象 | 检查方向 |
| --- | --- |
| 无法导入 minimind_lab | src 结构、打包配置、是否执行 uv sync |
| 找不到 minimind-audit | project.scripts、函数名、是否同步安装 |
| pytest 没发现测试 | 文件和函数命名是否符合 test_* |
| 缺失文件测试不报错 | 是否真正消费了生成器 |
| 拒绝行数与错误计数不同 | 一条记录是否包含多个问题 |
| 第二次运行数量翻倍 | 是否误用追加模式 a |
| 输入不存在却创建了目录 | 输入检查与创建输出的先后顺序 |

## 10. 真实样本

### 10.1 固定样本恢复

人工输入验证错误路径，真实样本检查上游格式是否能接入。在当前 Windows 仓库运行已有恢复脚本：

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_source_samples.ps1 -DatasetType pretrain
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore_source_samples.ps1 -DatasetType sft
~~~

脚本读取 [manifest](../implementation/data/source_samples/manifest.json)，下载固定版本片段，提取前二十个完整物理行，核验 UTF-8、JSON 和 SHA256 后保存。已有文件一致时只核验，不一致时拒绝覆盖。

Range 按字节截取，尾部可能是半个字符或半条 JSON，不能把整个片段强行解码，也不能用 errors="ignore" 掩盖损坏。网络或哈希检查失败时先解决来源问题，不随意换数据后继续引用旧结果。下载脚本属于辅助准备，不新增核心实现任务。

Mac/Linux 可按同一 manifest 用字节工具恢复，或者复制已有样本并核验哈希；不要直接执行 PowerShell 语法。

### 10.2 审计与观察

~~~powershell
uv run --locked minimind-audit data/source_samples/pretrain_sample.jsonl --dataset-type pretrain --output-dir data/audit/pretrain
uv run --locked minimind-audit data/source_samples/sft_sample.jsonl --dataset-type sft --output-dir data/audit/sft
~~~

历史两份报告均为 total=20、accepted=20、rejected=0。打开本次 report.json 核对，再观察 accepted 中原始字段，不只看程序退出。

### 10.3 字段解释


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

## 11. 来源与归档

### 11.1 清单

创建 data/source_samples/manifest.json，完整内容参考
[本次来源清单](../implementation/data/source_samples/manifest.json)，至少记录：

- source_repo、source_revision：来源仓库与版本。
- byte_range、selection：下载片段和抽取方法。
- upstream_file、record_count：原文件与样本条数。
- sha256：本地文件的字节身份。

在 PowerShell 中核验：

~~~powershell
Get-FileHash data/source_samples/pretrain_sample.jsonl -Algorithm SHA256
Get-FileHash data/source_samples/sft_sample.jsonl -Algorithm SHA256
~~~

本轮 Pretrain 为 a2aa375582bfd2c0bd87be9b031e1b6e187fa48b3b034ac246b177c8319b9db8，
SFT 为 525420d6d82b484ddc8e8cb88032b89acefc060e51d31be004b8e64716007d18。
重新格式化或改变换行符也会改变文件哈希；文件名相同不代表内容相同。

### 11.2 Git 边界

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

~~~powershell
git check-ignore -v data/source_samples/pretrain_sample.jsonl
git check-ignore -v data/source_samples/manifest.json
git status --short
~~~

JSONL 应被忽略，manifest 不应被忽略。第二条没有输出、退出码为 1，表示没有命中忽略规则。
代码、测试、依赖锁文件和 manifest 纳入版本管理，样本与输出留在本地。
保留 manifest 不等于保留数据本身；复现仍依赖固定版本的上游可获取或本地副本。

## 12. 验收与衔接

### 12.1 调用链回顾

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

### 12.2 完成标准

~~~powershell
uv run --locked pytest tests/test_reader.py tests/test_validator.py tests/test_pipeline.py tests/test_audit_cli.py -q
~~~

检查正常路径、三类退出状态和产物内容，再检查回归。历史第一周全仓 41 项通过、两类真实样本各二十条通过结构审计；这些不代表全量质量。当前输出会覆盖同名文件，I/O 中途失败也不保证回滚，不同实验使用不同输出目录。

本轮核心由学习者在指导下完成；助手负责讲解、排错提示、部分来源准备、复核和教程整理。可选复练是新增一条规则及对应测试，不是追加必做任务。

下一周将 accepted.jsonl 连接到 tokenizer、Dataset 和 batch。进度以 [实践路线](../PRACTICE_ROADMAP.md) 为准。

## 13. 学习问答

### 13.1 为什么合法 JSON 仍被拒绝？

JSON 语法和训练记录格式不同。数字、列表、null 都是合法 JSON，validator 则要求当前合同中的字段和类型。

### 13.2 为什么调用生成器不报错，转 list 才报错？

含 yield 的函数先返回生成器，迭代取值时才执行读取。缺失文件测试必须消费生成器。测试转 list 便于检查，正式流水线仍逐行消费。

### 13.3 为什么错误数大于拒绝行数？

一条记录可能同时缺 content、没有 assistant，产生两个错误，但只拒绝一条。两种统计的单位不同。

### 13.4 return 与 SystemExit 有什么区别？

return 只是函数结果；SystemExit 将整数交给操作系统。PowerShell 使用 $LASTEXITCODE 读取数值退出码，Bash 才使用 echo $?。PowerShell 的 $? 是布尔成功状态，不能区分 1 和 2。

### 13.5 为什么 CLI 叫 audit，以后不能训练了吗？

这是一个职责明确的入口。同一 Python 包可以另加训练和评测入口，不需要把所有功能塞进审计命令。

### 13.6 tmp_path、capsys、monkeypatch 从哪里来？

它们由 pytest 按测试参数名注入，分别提供临时目录、捕获输出和临时替换依赖。不要手动调用 fixture；测试结束后由框架清理其作用范围。

### 13.7 结构通过等于有可用监督吗？

不等于。当前允许空 content，没有完整检查工具调用和轮次语义。之后构造 SFT 数据还要检查模板、assistant mask 与有效监督 token。

### 13.8 换机器找不到 accepted.jsonl 怎么办？

样本和审计输出被 Git 忽略。先按 manifest 恢复 source_samples，再运行审计。来源清单不是数据文件本身，不能直接交给 Dataset。
