# 第 2 周：Tokenizer 与训练数据实践教程

> 教程整理：2026-10-19。输入是第一周审计通过的文本，输出是模型将接收的 batch；QA 在文末。

## 1. 本周任务与准备

第一周回答“这条记录格式是否合格”，本周回答“如何把文本变成可计算、可监督的张量”。先实现一套教学 BPE 理解编号从何而来，再换成冻结的 MiniMind tokenizer，构建实际训练数据管线。教学 BPE 和正式 tokenizer 的 ID 不能混用。

~~~text
小语料 → BPE 训练 → vocab + merges → encode / decode → 保存重载
已审计 JSONL → 冻结 tokenizer → Dataset → Collator → DataLoader
                                                  ↓
                               input_ids / attention_mask / labels
~~~

### 1.1 环境与输入

命令在 implementation 目录执行，使用 PowerShell。先运行 uv sync --locked。观察代码保存成注明的 scripts 文件，再用 uv run --locked python 执行，不把 Bash heredoc 直接粘到 PowerShell。

第一周的 data/audit/pretrain/accepted.jsonl 不在 Git 中。缺失时按第一周样本恢复与审计步骤重建。本周算法小实验不依赖该文件，最后真实数据接线才需要它。使用已有仓库的 tests 和 assets；重写核心在练习副本中进行。


**分步测试约定**：下面的 pytest 命令用于已有完整仓库的复练核验。从空文件逐步实现时，先执行本步给出的手算或最小断言；现有测试文件可能在收集时就导入后续类，-k 不会跳过这些导入。相关模块完成后再运行对应测试文件，不为通过尚未实现的后期测试提前增加功能。

### 1.2 实践顺序

| 步骤 | 本步结果 | 验证重点 |
| --- | --- | --- |
| 2–4. 三个 BPE 操作 | 统计、合并、选择 | 重叠计数与非重叠消费不同 |
| 5. 训练循环 | 固定词表与合并规则 | 每轮处理更新后的序列 |
| 6–7. 编解码 | 文本与 ID 往返 | 新文本复用规则，字节整体解码 |
| 8. 保存重载 | 持久化文件 | 文件合同与往返行为都正确 |
| 9–10. 官方文件与观察 | 冻结 tokenizer | 词表、特殊 token、字节身份明确 |
| 11. 训练数据管线 | 三个同形状张量 | padding、mask、labels 和 shift 职责清楚 |
| 12. 验收 | 首个真实 batch | 上下游真正接通 |

完成标准不是“分词打印正常”，而是保存后 ID 语义不变，Dataset 能接收真实文件，collator 的逐元素结果正确，并能解释哪些 token 最终参与 loss。

## 2. 相邻对统计

### 2.1 目的

BPE 从小单元开始，不断把经常相邻的两个 token 合成新 token。
我们的初始单元是 UTF-8 字节，第一步先统计哪些相邻组合出现最多。

```python
list("abab".encode("utf-8"))
# [97, 98, 97, 98]
```

`str.encode` 得到 bytes；将 bytes 转成 list，得到每个字节的整数值。
本实现初始 token ID 就等于字节值，范围是 0–255。

### 2.2 实现

创建 `models/__init__.py` 和 `models/bpe.py`，实现：

```python
def count_pairs(
    sequences: list[list[int]],
) -> dict[tuple[int, int], int]:
    ...
```

输入是多条独立序列，输出是 pair 到累计次数的字典。
对每条序列遍历相邻位置，可以用索引，也可以用：

```python
for a, b in zip(sequence, sequence[1:]):
    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
```

例如 `[97,98,97,98]` 产生 `(97,98)`、`(98,97)`、`(97,98)`。
统计只在每条序列内部进行，不能把上一条结尾和下一条开头算成一个 pair。

### 2.3 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def count_pairs(
    sequences: list[list[int]],
)-> dict[tuple[int, int], int]:
    pair_counts: dict[tuple[int, int], int] = {}
    for sequence in sequences:
        for (a, b) in zip(sequence, sequence[1:]):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    return pair_counts
~~~

### 2.4 测试

创建 `tests/test_bpe.py`，导入 count_pairs 并验证：

```python
assert count_pairs([[97, 98, 97, 98], [97, 98], [], [99]]) == {
    (97, 98): 3,
    (98, 97): 1,
}
assert count_pairs([[1, 1, 1]]) == {(1, 1): 2}
```

第二条有两个重叠的相邻位置，统计时都算。空列表和单 token 没有相邻 pair。
用 `uv run pytest -q tests/test_bpe.py` 执行。

最初写在 bpe.py 底部的示例和 print 只是手动观察，不会自动成为 pytest 测试，
还会在导入模块时打印。把验证移入 test_* 函数，库模块保留定义即可。

## 3. 非重叠合并

### 3.1 目的

统计后需要一种操作：把指定 pair 替换成新 ID。
这个函数只执行替换，暂时不决定选哪个 pair，也不分配 ID。

```python
def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    ...
```

### 3.2 实现

从左到右移动指针 i：匹配时追加新 ID 并前进 2；不匹配时保留当前 token 并前进 1。
如果循环只检查到倒数第二个位置，最后还要处理未消费的尾 token。
返回新列表，不修改传入的 sequence。

```text
merge_pair([97,98,97,98], (97,98), 256) -> [256,256]
merge_pair([1,1,1],      (1,1),   256) -> [256,1]
```

统计允许相邻位置重叠，合并不允许同一个位置被消费两次。因此 `[1,1,1]` 虽然 pair 次数是 2，
从左到右这一轮只能替换一次。

### 3.3 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_token_id:int,
)-> list[int]:
    result = []

    i = 0
    while i < (len(sequence) -1):
        if (sequence[i], sequence[i+1]) == pair:
            result.append(new_token_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1

    if i == len(sequence) - 1:
        result.append(sequence[i])

    return result
~~~

### 3.4 测试

使用参数化测试覆盖下面的输入和预期结果：

| sequence | pair | new_token_id | expected |
| --- | --- | --- | --- |
| [97,98,97,98] | (97,98) | 256 | [256,256] |
| [1,1,1] | (1,1) | 256 | [256,1] |
| [1,2,3] | (4,5) | 256 | [1,2,3] |
| [] | (1,2) | 256 | [] |
| [1] | (1,2) | 256 | [1] |

另写一个测试，在调用后检查原 sequence 未改变。
测试必须实际调用 merge_pair，只导入模块或打印一次结果不能验证此函数。

## 4. 最优对选择

### 4.1 目的

从计数字典中选择本轮要学习的 pair，并明确次数相同怎么办。

```python
def select_best_pair(
    pair_counts: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    ...
```

### 4.2 合同

频次最高者优先；同频时选择 tuple 值更大的 pair；空字典返回 None。
同频规则是本教学实现的确定性约定，不要求与官方训练器的同频处理一致。

复合比较键可以写成：

```python
key=lambda pair: (pair_counts[pair], pair)
```

max 先比较第一个元素，即次数，再比较第二个元素，即 pair。
没有 pair 通常表示序列已不足两个 token，是正常停止条件。

### 4.3 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def select_best_pair(pair_counts: dict[tuple[int, int], int],) -> tuple[int, int] | None:
    if not pair_counts:
        return None

    best_pair = max(pair_counts, key= lambda pair: (pair_counts[pair], pair))

    return best_pair
~~~

### 4.4 测试

```python
assert select_best_pair({(97, 98): 3, (98, 97): 1}) == (97, 98)
assert select_best_pair({(97, 98): 2, (99, 100): 2}) == (99, 100)
assert select_best_pair({}) is None
```

本轮曾把空字典改成 ValueError，测试也跟着期待异常。虽然测试全绿，但偏离了约定。
修正时同时恢复实现和测试：测试必须有独立的合同依据，不能只是照着代码当前行为写断言。

## 5. 训练循环

### 5.1 目的

前三个函数只做单项操作。train_bpe 负责初始化、反复调用它们，并保留学习结果。
这里的“训练”是统计与合并，不涉及梯度、loss 或 optimizer。

```python
def train_bpe(
    texts: list[str],
    vocab_size: int,
) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    ...
```

返回值分别是：vocab 记录“ID 表示什么字节”，merges 记录“哪两个 ID 合成哪个新 ID”。

### 5.2 初始化

检查 vocab_size 不小于 256，否则抛 ValueError。然后准备：

```python
vocab = {token_id: bytes([token_id]) for token_id in range(256)}
merges = {}
sequences = [list(text.encode("utf-8")) for text in texts]
```

`bytes([97])` 构造一个字节，显示为 `b'a'`；这不是十进制转十六进制。
bytes 的显示可能包含 `\x..`，那只是不可打印字节的显示形式。

### 5.3 循环

```text
当前 sequences
  -> count_pairs：统计当前 token 的相邻组合
  -> select_best_pair：选本轮规则
  -> 没有 pair：停止
  -> 分配新 ID，拼接左右 token 的 bytes，写入 vocab 和 merges
  -> 对每条 sequence 调用 merge_pair
  -> 用更新后的 sequences 进入下一轮
```

新 ID 为 len(vocab)，从 256 递增。停止条件是达到目标词表大小，或已经没有 pair。
vocab_size 是目标上限，不保证小语料一定能达到它；本实现允许频次为 1 的 pair 继续合并。

### 5.4 完整例子

设 texts 为 `["abab", "ab"]`，vocab_size 为 258：

| 状态 | 当前序列 | 本轮结果 |
| --- | --- | --- |
| 初始化 | [97,98,97,98] 和 [97,98] | 256 个单字节 token |
| 第 1 轮 | 统计 (97,98) 三次，(98,97) 一次 | 256 表示 b'ab' |
| 合并后 | [256,256] 和 [256] | 重新统计更新后的序列 |
| 第 2 轮 | (256,256) 一次 | 257 表示 b'abab' |
| 结束 | [257] 和 [256] | 词表大小达到 258 |

第一轮 `vocab[256] = vocab[97] + vocab[98]`；
第二轮 `vocab[257] = vocab[256] + vocab[256]`。
后续统计的 pair 可能包含已经合并过的 token，并不总是原始单字节。

### 5.5 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def train_bpe(texts: list[str], vocab_size: int) -> tuple[
        dict[int, bytes],
        dict[tuple[int, int], int]
    ]:
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256")

    # 0. 定义合并规则
    merges: dict[tuple[int, int], int] = {}

    # 1. 初始词表包含 0～255 的所有单字节 token。
    vocab = {
        token_id : bytes([token_id])
        for token_id in range(256)
    }

    # 2. 先将文本编码成序列
    sequences = [
        list(text.encode("utf-8"))
        for text in texts
        ]

    while len(vocab) < vocab_size:
        # 3. 统计序列中出现的字节对
        pair_counts = count_pairs(sequences)

        # 4. 选择出现次数最多的字节对
        best_pair = select_best_pair(pair_counts)

        if best_pair is None:
            break

        # 5. 合并原序列中的token 为 new_token_id
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 6. 记录合并规则
        merges[(best_pair)] = new_token_id

        # 7. 合并old token id to new token id
        sequences = [
            merge_pair(sequence, best_pair, new_token_id)
            for sequence in sequences
        ]

    return (vocab, merges)
~~~

### 5.6 测试

```python
vocab, merges = train_bpe(["abab", "ab"], vocab_size=258)
assert merges == {(97, 98): 256, (256, 256): 257}
assert vocab[256] == b"ab"
assert vocab[257] == b"abab"

vocab, merges = train_bpe(["a", ""], vocab_size=260)
assert len(vocab) == 256
assert merges == {}
```

再用 pytest.raises 验证 vocab_size=255 抛出 ValueError。
若使用 expected_merges 参数化，断言要使用该参数，避免声明了预期却在函数里重新写死另一份。

## 6. 编码

### 6.1 目的

训练已经确定词表，编码负责将规则用于新文本。不能对每次输入重新训练，
否则同一个 ID 的含义会变化，模型的 embedding 就无法保持一致。

```python
def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    ...
```

### 6.2 实现

先将 text 转成字节列表，再按 new_token_id 升序遍历 merges，逐次调用 merge_pair。
本实现中新 ID 按学习顺序分配，因此 ID 同时表示规则先后。
编码直接复用规则，无需 count_pairs 或 select_best_pair。

```text
(97,98) -> 256       先产生 256
(256,256) -> 257     再使用 256

"abab" -> [97,98,97,98] -> [256,256] -> [257]
"aba"  -> [97,98,97]    -> [256,97]  -> [256,97]
```

### 6.3 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def encode(text:str, merges: dict[tuple[int, int], int]) -> list[int]:
    # 1. 将文本编码成序列
    sequence = list(text.encode("utf-8"))

    # 2. 按照new token id 的从小到大的顺序来依次合并序列中的token
    for pair, new_token_id in sorted(merges.items(), key= lambda item: item[1]):
        sequence = merge_pair(sequence, pair, new_token_id)

    return sequence
~~~

### 6.4 测试

沿用上节学出的 merges，验证 abab、ab、aba 和空字符串分别得到
`[257]`、`[256]`、`[256,97]`、`[]`。
没有匹配的字节保持原 ID，不需要为没见过的普通字符新增 token。

## 7. 解码

### 7.1 目的

编码依赖 merges，解码只需 vocab：查出 token 对应的 bytes，拼起来恢复原文。

```python
def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    ...
```

### 7.2 实现

遍历 ID，未知 ID 抛出 `ValueError("unknown token id: ...")`。
把每段 bytes 放入列表，最后 `b"".join(byte_parts).decode("utf-8")`。
避免反复用 bytes += 拼接导致重复复制。

中文字符的 UTF-8 字节可能分在多个 token 中，不能逐 token 单独解码。
必须先拼字节，再整体 decode。完整 encode 结果应可还原原文；任意截断的 ID 序列则未必是合法 UTF-8，
当前教学实现会直接抛 UnicodeDecodeError，不属于流式生成解码器。

### 7.3 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    byte_part = []
    for token_id in token_ids:
        if token_id not in vocab:
            raise ValueError(f"unknown token id: {token_id}")

        byte_part.append(vocab[token_id])

    raw_bytes = b"".join(byte_part)

    return raw_bytes.decode("utf-8")
~~~

### 7.4 测试

先验证 `[257]` 可以还原 abab；再用参数化测试检查空串、hello、你好、hello，世界：

```python
vocab, merges = train_bpe(["你好你好", "hello hello"], vocab_size=270)
assert decode(encode(text, merges), vocab) == text
```

测试文本可以包含训练时没见过的字，例如世界，因为基础词表覆盖全部 256 种字节。
再验证未知 ID 999 的错误。完整测试示例见 test_bpe.py。

## 8. 保存与重载

### 8.1 目的

训练结果需要在程序重启后继续使用。保存后重载，必须维持相同 token ID 与文本含义。

```python
def save_bpe(path: str | Path, merges: dict[tuple[int, int], int]) -> None:
    ...

def load_bpe(path: str | Path) -> tuple[
    dict[int, bytes], dict[tuple[int, int], int]
]:
    ...
```

### 8.2 文件合同

JSON 不接受 tuple 作为对象键，也不能直接保存 bytes。将每条规则写成三元列表，
按 new_token_id 升序保存：

```json
{
  "format_version": 1,
  "merges": [
    [97, 98, 256],
    [256, 256, 257]
  ]
}
```

每项表示 `[left_id, right_id, new_id]`。当前格式无需保存 vocab_size 或完整 vocab：
初始化 0–255 的 bytes 后，依次执行 `vocab[new_id] = vocab[left_id] + vocab[right_id]` 就能重建。

### 8.3 实现

保存时把 path 转为 Path，创建父目录，按 ID 排序，写 UTF-8 JSON 并补末尾换行。
版本使用整数 1。加载时读 JSON，检查版本，再逐条建立 merges 与 vocab，返回两者。

本轮修正过三个问题：写死 vocab_size=258、用字符串 "1" 表示版本、保存时依赖 dict 插入顺序。
这些错误可能被互相配合的保存与加载代码掩盖，所以必须直接断言文件格式。

### 8.4 参考实现

按上面的顺序在 models/bpe.py 实现；前面定义的辅助函数继续保留。本步完整函数如下：

~~~python
import json
from pathlib import Path

def save_bpe(path: str | Path, merges: dict[tuple[int, int], int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merge_records = [
        [pair[0], pair[1], new_token_id]
        for pair, new_token_id
        in sorted(
            merges.items(),
            key=lambda item: item[1]
        )
    ]

    save_bped = {
        "format_version": 1,
        "merges": merge_records
    }

    path.write_text(
        json.dumps(
            save_bped,
            ensure_ascii=False,
            indent=2)
        + "\n", encoding="utf-8")

def load_bpe(path: str | Path) -> tuple[
    dict[int, bytes],
    dict[tuple[int, int], int]
]:
    path = Path(path)
    load_bped = json.loads(path.read_text(encoding="utf-8"))

    vocab = {
        token_id : bytes([token_id])
        for token_id in range(256)
    }

    if load_bped["format_version"] != 1:
        raise ValueError("unsupported BPE format version")

    merges: dict[tuple[int, int], int] = {}
    for merge_record in load_bped["merges"]:
        pair = (merge_record[0], merge_record[1])
        new_token_id = merge_record[2]
        merges[pair] = new_token_id
        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

    return vocab, merges
~~~

### 8.5 测试

第一组在 tmp_path 中保存重载，比较数据和行为：

```python
vocab, merges = train_bpe(["abab", "ab"], vocab_size=258)
path = tmp_path / "tokenizer.json"
save_bpe(path, merges)
loaded_vocab, loaded_merges = load_bpe(path)

assert loaded_vocab == vocab
assert loaded_merges == merges
assert encode("abab", loaded_merges) == encode("abab", merges)
assert decode(encode("abab", loaded_merges), loaded_vocab) == "abab"
```

第二组故意颠倒传入字典的插入顺序，读回 JSON 后断言它仍等于上面的整数版本、有序文件合同。
第三组手工写 version=2，用 pytest.raises 验证错误消息 `unsupported BPE format version`。

这些自动测试在同一测试进程中读取文件，证明持久化前后内容与行为一致；尚未编写独立子进程重启测试。
加载器当前面向本程序保存的文件，并未完整校验损坏格式、重复 ID、缺失字段或非法引用。

## 9. 官方文件与来源冻结

### 9.1 文件准备

已有仓库直接使用 [assets/minimind_tokenizer](../implementation/assets/minimind_tokenizer/manifest.json) 中登记的两个文件，不再依赖旧机器上的外部源码路径。先检查：

~~~powershell
Test-Path assets/minimind_tokenizer/tokenizer.json
Test-Path assets/minimind_tokenizer/tokenizer_config.json
~~~

两项都应为 True。若缺失，从本项目 Git 版本恢复这两个文件，或按 manifest 中的仓库与冻结提交取得原文件再核验。不要拿教学 save_bpe 的 JSON 覆盖官方 tokenizer.json，两者格式不同。

### 9.2 来源

新建 manifest.json，记录源仓库、冻结提交与两个文件的 SHA256。
完整内容见 [官方 tokenizer 来源清单](../implementation/assets/minimind_tokenizer/manifest.json)。

```powershell
Get-FileHash assets/minimind_tokenizer/tokenizer.json -Algorithm SHA256
Get-FileHash assets/minimind_tokenizer/tokenizer_config.json -Algorithm SHA256
```

| 文件 | SHA256 |
| --- | --- |
| tokenizer.json | 71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668 |
| tokenizer_config.json | d7cd6a60c9f191c4f9ffec69b2eb289ad2f960366724f328b681ae1e46e4c110 |

源提交为 `f659b55761b754d306bd140573493a6543cafd7f`，仓库为 `jingyaogong/minimind`。
词表文件决定映射，配置文件还影响特殊标记和模板，所以两个都要记录。

### 9.3 自动测试

在 tests/test_official_tokenizer.py 用以下方式定位 assets，避免依赖终端当前目录：

```python
TOKENIZER_PATH = Path(__file__).parents[1] / "assets" / "minimind_tokenizer"
```

导入 pathlib.Path 和 transformers.AutoTokenizer，分别写两个测试：

1. 加载后 len(tokenizer)=6400，BOS=1、EOS=2、PAD=0。
2. 对观察文本，默认编码等于 add_special_tokens=False，并能解码还原。

文件哈希已经通过终端复核，但目前未加入自动测试；两个行为测试也不验证全词表身份。
完整实现见 [官方 tokenizer 测试](../implementation/tests/test_official_tokenizer.py)。

```powershell
uv run pytest -q tests/test_bpe.py tests/test_official_tokenizer.py
uv run pytest -q
```

这两个行为测试不验证整个词表的字节身份，哈希核验仍单独进行。

## 10. 官方 Tokenizer 观察

### 10.1 差异

教学 BPE 已完成算法闭环。接下来加载官方文件，建立后续训练实际使用的 token 身份。

| 项目 | 教学实现 | 冻结版本的官方实现 |
| --- | --- | --- |
| 训练 | Python 统计与合并 | tokenizers 的 BPE/BpeTrainer |
| 输入处理 | 每条文本直接转 UTF-8 bytes | ByteLevel 预切分与字节映射 |
| 词表 | 256 字节加自学 merges | 6400 项的现成词表 |
| 特殊标记 | 尚未设计 | BOS/EOS/PAD、思考与工具等标记 |
| 使用目的 | 理解合并、编码、解码 | 作为后续训练的正式 tokenizer |

官方参考为冻结仓库的 `trainer/train_tokenizer.py`、`model/tokenizer.json` 和
`model/tokenizer_config.json`。同样是 BPE，不代表词表、预切分、同频规则或 token ID 相同。

### 10.2 加载

本轮按官方 requirements 安装了 Transformers 4.57.6，其 tokenizers 依赖由 uv 解析并锁定。
复练新环境可执行 `uv add "transformers==4.57.6"`；当前工程直接用 `uv sync --locked`。

最初从参考源码加载，后续复制到 assets；现在运行下面这段即可复现观察：

把下面代码保存为 scripts/inspect_official_tokenizer.py（复练新增观察脚本，不是仓库当前必需文件）：

~~~python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "assets/minimind_tokenizer", local_files_only=True,
)
text = "你好，MiniMind!"
ids = tokenizer(text, add_special_tokens=False).input_ids
print("vocab size:", len(tokenizer))
print("bos:", tokenizer.bos_token, tokenizer.bos_token_id)
print("eos:", tokenizer.eos_token, tokenizer.eos_token_id)
print("pad:", tokenizer.pad_token, tokenizer.pad_token_id)
print("input_ids:", ids)
print("tokens:", tokenizer.convert_ids_to_tokens(ids))
print("decoded:", tokenizer.decode(ids, skip_special_tokens=False))
~~~

~~~powershell
uv run --locked python scripts/inspect_official_tokenizer.py
~~~


local_files_only=True 只从本地加载；此操作无需下载模型权重。
缺少文件时返回上一节恢复并校验 assets，不临时换 tokenizer 绕过。

### 10.3 实际结果

```text
vocab size: 6400
bos: <|im_start|> 1
eos: <|im_end|> 2
pad: <|endoftext|> 0
input_ids: [1968, 294, 80, 301, 108, 80, 916, 36]
tokens: ['ä½łå¥½', 'ï¼Į', 'M', 'in', 'i', 'M', 'ind', '!']
decoded: 你好，MiniMind!
```

这句被切为 `你好 | ， | M | in | i | M | ind | !`，共 8 个 token。
ByteLevel 把字节映射为可打印字符，convert_ids_to_tokens 展示的是该内部形式，
因此看到 ä½łå¥½ 不代表编码损坏；decode 会执行逆转换。

冻结文件中有 256 个基础字节项、6108 条 merge 和 36 个登记的 added tokens，词表共 6400。
36 个 added tokens 不全都被标记为 special；例如思考和工具标记的配置需要区分对待，
不能假定 skip_special_tokens 会去掉所有这些标记。

### 10.4 边界标记

已验证这句文本默认调用与 add_special_tokens=False 得到同样的 ID，没有自动加 BOS/EOS。
在官方 Pretrain 路径里，由 Dataset 手动构造：

```python
bounded_ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
```

解码可看到 `<|im_start|>你好，MiniMind!<|im_end|>`。
此结论针对普通文本编码；apply_chat_template 还会根据模板插入角色和边界，不能混为同一行为。

## 11. Pretrain 数据管线

### 11.1 目标

这一部分接上第一周的 `accepted.jsonl`，把每条 `text` 转换为模型可以接收的定长张量：

```text
accepted.jsonl
-> PretrainDataset[index]
-> encode_pretrain_text
-> 未 padding 的 input_ids
-> PretrainCollator
-> 定长 input_ids / attention_mask / labels
-> DataLoader batch
```

这里刻意把官方 `PretrainDataset` 内的职责拆成三层：

| 层 | 职责 | 不负责 |
| --- | --- | --- |
| `encode_pretrain_text` | 单条文本的 tokenize、截断、BOS/EOS | 文件读取、padding、张量化 |
| `PretrainDataset` | 建立索引访问，取出第 N 条文本并编码 | 组 batch、padding、labels |
| `PretrainCollator` | 把多个样本补齐并构造训练字段 | 读取 JSONL、选择样本顺序 |

拆分让每层可以独立验证。普通文本的 token 顺序与固定长度行为可对照参考实现；特殊 ID 的屏蔽规则应明确记录差异，而不是默认完全一致。

### 11.2 官方对照

冻结参考提交的 dataset/lm_dataset.py::PretrainDataset 行为为：

1. 读取记录的 text。
2. 不自动添加特殊 token，把正文截断至 max_length-2。
3. 手动添加 BOS/EOS。
4. 在右侧 padding 到 max_length，转换为 torch.long。
5. 复制 input_ids 作为 labels，把 input_ids 等于 PAD ID 的位置设为 -100。
6. 返回 input_ids 与 labels，DataLoader 将定长样本堆叠成 batch。

官方模型在计算 loss 时做 causal shift，因此数据层不提前 shift。官方预训练循环当前没有传
`attention_mask`，但模型 `forward` 支持该参数；“当前没传”不等于“模型不支持”。

本轮保留官方的固定 `max_length` 行为，但把 padding 和 labels 移到 collator，并额外交付
`attention_mask`。这是职责拆分；对普通文本保留边界与截断语义。但真实正文含 PAD ID 时，我们按补入位置屏蔽，与参考代码按 ID 屏蔽的边界行为不同，不能笼统宣称完全相同。

### 11.3 单条编码

`encode_pretrain_text` 的合同是：

```python
def encode_pretrain_text(text, tokenizer, max_length):
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 2,
    )
    return [tokenizer.bos_token_id] + encoding.input_ids + [tokenizer.eos_token_id]
```

必须先为 BOS 和 EOS 预留两个位置。如果先把正文截断到 `max_length`，再添加边界，最终长度会超过合同；
如果超长后直接切尾，又可能把 EOS 切掉。

测试覆盖了四类情况：完整文本、正文截断、空文本以及 `max_length < 2`。例如：

```text
原始正文 IDs：1968 294 80 301 108 80 916 36
max_length=5：1 1968 294 80 2
              ^             ^
             BOS           EOS
```

### 11.4 Dataset

MiniMind 官方使用 `datasets==3.6.0` 的 `load_dataset("json")`。本轮保持相同依赖，Dataset 只返回
未 padding 的 Python 列表：

```python
{
    "input_ids": [1, ..., 2]
}
```

`__len__` 给出记录数，`__getitem__(index)` 决定第 `index` 条样本是什么。样本如何组合、是否打乱以及
每批取多少条，不属于 Dataset 本身。

这一层暴露了两个实际问题：

1. pytest fixture 不能像普通函数一样直接调用。测试函数应把 fixture 名写成参数，由 pytest 注入。
2. 类型合同声明 `str | Path` 后，不能只测字符串。`datasets==3.6.0` 的 `data_files` 在这里需要
   `str(data_path)`；临时测试先转成字符串曾掩盖了真实 `Path` 输入的错误。

第二个问题是在直接读取第一周真实审计产物时发现的，说明单元测试通过不等于真实入口已经接通。

### 11.5 Collator

collator 接收若干条未 padding 的记录，并输出三个 `torch.long` 张量：

```text
input_ids       实际 token + 右侧 PAD
attention_mask  实际位置为 1，padding 位置为 0
labels          实际位置复制 input_ids，padding 位置为 -100
```

以 `max_length=6` 为例：

```text
features:
  [1, 10, 11, 2]
  [1, 20, 2]

input_ids:
  [1, 10, 11, 2, 0, 0]
  [1, 20,  2, 0, 0, 0]

attention_mask:
  [1, 1, 1, 1, 0, 0]
  [1, 1, 1, 0, 0, 0]

labels:
  [1, 10, 11, 2, -100, -100]
  [1, 20,  2, -100, -100, -100]
```

实现时出现过一次测试假阳性：测试选取的两条文本都恰好截断到 `max_length=6`，所以
`padding_length=0`，错误的 mask 构造也能通过。改为直接输入两条不同长度的 ID 列表，并断言完整张量后，
才真正覆盖 padding 分支。

mask 必须依据补入的位置构造，而不能把所有等于 `pad_token_id` 的真实 token 都自动视为 padding。
边界测试专门使用 `[1, 0, 2]`，证明中间真实位置的 ID 0 仍被监督：

```text
labels = [1, 0, 2, -100, -100]
```

另外还测试了空 features 和超长 input_ids，避免返回形状不确定或静默截断的数据。

### 11.6 数据模块完整实现

在 src/minimind_lab/data/pretrain.py 中依次实现单条编码、Dataset 和 Collator。上面的表用于先手算预期；下面是可与分步结果核对的完整文件：

~~~python
import torch

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path


def encode_pretrain_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[int]:
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2")

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length-2,
    )

    content_ids = encoding.input_ids
    full_ids = [tokenizer.bos_token_id] + content_ids + [tokenizer.eos_token_id]

    return full_ids


class PretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int) :
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.data = load_dataset(
            "json",
            data_files=str(data_path),
            split="train",
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) :
        text = self.data[index]["text"]

        input_ids = encode_pretrain_text(
            text,
            self.tokenizer,
            self.max_length,
        )

        return {
            "input_ids" : input_ids
        }


class PretrainCollator:
    def __init__(self,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
    ) -> None:
        if max_length < 2:
            raise ValueError("max length must be at least 2")

        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features: list[dict] ) -> dict[str, torch.Tensor]:
        if features is None or len(features) == 0:
            raise ValueError("features must not be empty")

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for feature in features:
            input_ids = feature["input_ids"]
            if len(input_ids) > self.max_length:
                raise ValueError(f"input_ids exceed max_length")

            padding_length = self.max_length - len(input_ids)
            real_length = len(input_ids)
            # 1. 右侧padding
            input_ids = input_ids + [self.pad_token_id] * padding_length

            # 2. 构造attention_mask
            attention_mask = [1] * real_length + [0] * padding_length

            # 3. 构造labels
            labels = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(input_ids, attention_mask)
            ]

            # 4. 合并batch
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids" : torch.tensor(
                input_ids_batch,
                dtype=torch.long),
            "attention_mask" : torch.tensor(
                attention_mask_batch,
                dtype=torch.long),
            "labels" : torch.tensor(
                labels_batch,
                dtype=torch.long)
        }
~~~

### 11.7 DataLoader

DataLoader 把前三层装配起来：

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator

tokenizer = AutoTokenizer.from_pretrained("assets/minimind_tokenizer", local_files_only=True)
dataset = PretrainDataset("data/audit/pretrain/accepted.jsonl", tokenizer, max_length=512)
collator = PretrainCollator(tokenizer, max_length=512)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
)
batch = next(iter(dataloader))
```

一次迭代的实际调用顺序为：

```text
确定索引 0、1
-> dataset[0]、dataset[1]
-> 两条未 padding 的 input_ids
-> collator([sample0, sample1])
-> 三个 [B, T] 张量
```

集成测试同时覆盖了一条被截断的样本和一条需要 padding 的样本，固定完整的 `input_ids`、
`attention_mask` 和 `labels`，而不只检查 shape。

### 11.8 真实样本验证

当前仓库的 inspect_pretrain_batch.py 已在第四周扩展到模型更新与生成；第二周从零复练不要直接依赖它。将下面的纯数据观察保存为 scripts/inspect_week02_batch.py（练习文件），只使用本周已实现的模块：

~~~python
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator

tokenizer = AutoTokenizer.from_pretrained("assets/minimind_tokenizer", local_files_only=True)
dataset = PretrainDataset("data/audit/pretrain/accepted.jsonl", tokenizer, max_length=512)
loader = DataLoader(
    dataset, batch_size=2, shuffle=False,
    collate_fn=PretrainCollator(tokenizer, max_length=512),
)
batch = next(iter(loader))
print("Dataset size:", len(dataset))
for name, tensor in batch.items():
    print(name, tensor.shape, tensor.dtype)
print("Valid tokens:", batch["attention_mask"].sum(dim=1))
print("loss tokens:", (batch["labels"][:, 1:] != -100).sum(dim=1))
assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
real = batch["attention_mask"].bool()
assert (batch["labels"][real] == batch["input_ids"][real]).all()
print("padding check passed")
print("label check passed")
~~~


~~~powershell
uv run --locked python scripts/inspect_week02_batch.py
~~~


脚本读取第一周产出的 `data/audit/pretrain/accepted.jsonl`，而不是测试夹具。固定二十条样本的历史结果为：

```text
Dataset size: 20
input_ids shape: torch.Size([2, 512]), dtype: torch.int64
attention_mask shape: torch.Size([2, 512]), dtype: torch.int64
labels shape: torch.Size([2, 512]), dtype: torch.int64
Valid tokens: tensor([355, 137])
loss tokens: tensor([354, 136])
padding check passed
label check passed
```

20 条样本的编码长度范围为 49–451，因此在 `max_length=512` 下没有样本被截断。首批两条样本分别
有 355 和 137 个非 padding token。

`labels != -100` 的数量与有效 token 数相同，但模型随后使用：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

所以真正进入交叉熵的目标 token 数分别是 354 和 136。第 0 位 BOS 不作为预测目标，EOS 仍被监督。

### 11.9 验收

历史阶段命令与结果（当前用例已增加，不要求匹配旧总数）：

```powershell
uv run pytest -q tests/test_pretrain_data.py
# 10 passed

uv run pytest -q
# 77 passed
```

至此，第一周与第二周已经接成一条可执行链路：

```text
原始 JSONL
-> reader / validator / audit
-> accepted.jsonl
-> Dataset / tokenizer
-> collator
-> DataLoader batch
```

模型尚未参与其中。第三周开始实现模型基础组件，第四周才接入 causal attention、语言模型输出和 loss。

## 12. 验收与衔接

### 12.1 最终检查

~~~powershell
uv run --locked pytest tests/test_bpe.py tests/test_official_tokenizer.py tests/test_pretrain_data.py -q
uv run --locked python scripts/inspect_week02_batch.py
~~~

算法先检查 abab 的两次合并和未见文本往返；数据管线检查一条截断、一条 padding，以及真实位置恰好包含 PAD ID 的反例。只看 shape 或只用刚好填满长度的样本，无法发现错误屏蔽。

教学 BPE 加载器只面向本程序输出，尚非防御性处理任意损坏文件的通用解析器。数据 collator 采用固定长度右填充；这里尚未做 packing、SFT assistant-only mask 或分布式数据读取。

### 12.2 产物与下一步

输入、算法词表文件、官方 tokenizer 和数据 batch 的职责应能区分。第三周在这些 token ID 上建立可训练特征表示；第四周再加入注意力和 next-token loss。数据层不提前 shift，避免模型层重复偏移。

本轮核心、测试及观察脚本由学习者在指导下编写；助手提供合同、示例、复核和整理。历史第二周验收是数据测试 10 项、全仓 77 项，不作为当前独立实现能力的量化分数。完整路线见 [PRACTICE_ROADMAP](../PRACTICE_ROADMAP.md)。

## 13. 学习问答

### 13.1 三个函数和训练循环是什么关系

count_pairs 观察、select_best_pair 决策、merge_pair 应用；train_bpe 重复并保存结果。

### 13.2 print 结果对了，pytest 却没增加

print 是手动观察，需要新增 test_* 才会自动收集。

### 13.3 空 pair 抛异常也能通过测试

实现和测试一起偏离合同；空 pair 应是正常终止信号。

### 13.4 bytes([id]) 是十进制转十六进制吗

它构造实际字节，显示形式与字节值是两回事。

### 13.5 统计重叠，合并为什么不重叠

统计候选位置，合并时每个位置只能消费一次。

### 13.6 encode 为什么不再选最高频 pair

新文本复用固定规则；重新选会改变编码语义。

### 13.7 中文为什么需要先拼字节再解码

一个字符的字节可能分散在多个 token 中。

### 13.8 save/load 都通过但文件合同仍有问题

往返测试只能证明互相兼容；需独立检查版本、字段和排序。

### 13.9 tokenizer 里的中文看似乱码

看到的是 ByteLevel 内部字节映射，decode 才恢复原文。

### 13.10 Dataset、Collator、DataLoader 各自负责什么？

Dataset 回答第 N 条样本是什么；Collator 把若干样本补齐、构造 mask 与 labels；DataLoader 按索引取样、分批并调用 Collator。不要在三个地方重复编码或 shift。

### 13.11 labels 的有效数为什么比 loss 里的有效数多一？

本周 labels 复制全部真实输入，含首位 BOS。因果预测用 logits 的前 T-1 个位置预测 labels 的后 T-1 个位置，首位不是目标，EOS 仍可作为目标。应在 shift 后统计真正的监督数。

### 13.12 fixture 为什么不能直接调用，Path 为什么要再转 str？

pytest 按参数注入 fixture；直接调用不是普通辅助函数用法。Dataset 的公开接口接受 Path，但这里锁定版本的 load_dataset 调用需要 str(data_path)。测试应传真实 Path，不能先在测试里转字符串掩盖边界错误。

### 13.13 真实 token 恰好为 PAD ID，是否都应忽略？

不应只凭编号判断是否是补齐。本实现根据实际长度构造 attention_mask 和 labels，因此 [1,0,2] 的中间 0 仍是监督目标。补入的 0 才被屏蔽。这是本工程明确选择的边界语义。
