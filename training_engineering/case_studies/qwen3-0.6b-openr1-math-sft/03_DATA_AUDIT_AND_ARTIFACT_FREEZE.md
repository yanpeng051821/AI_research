# 03 数据审计、人工复核与 Artifact 冻结

## 1. 目的

训练数据不是“能被 DataLoader 读取就合格”。这一阶段要把上游数据转换成一组满足
训练合同、可以复现、不会被后续静默修改的 artifact。

本案例的数据源为固定 revision 的 `open-r1/OpenR1-Math-220k`。一条原始记录可能包含
题目、多个候选 generation、验证结果和来源信息。SFT 最终只能选择一条符合规则的
回答进入训练，因此需要明确选择策略，而不是默认取第一个候选。

本案例状态：`COMPLETE`。以下流程是方法说明，验收框是本次实际执行结果，不是待办列表。

## 2. 输入与输出

输入：

```text
原始数据集名称与 revision
固定 tokenizer/chat template revision
候选回答与 verifier 字段
污染检查数据集 revision
人工排除列表
审计参数、split salt、review salt
```

输出：

```text
train.jsonl
validation.jsonl
rejected.jsonl
review_samples.jsonl
length_filtered_train.jsonl
length_filtered_validation.jsonl
data_manifest.json
人工 review decisions 与 summary
```

## 3. 自动审计流程

从实验仓库根目录执行审计脚本。正式运行前先用少量数据验证参数和输出结构，再跑全量。

```bash
uv run python scripts/audit_openr1_math.py \
  --output-dir artifacts/<new-artifact-id> \
  --validation-size 2000 \
  --max-length 32768 \
  --manual-exclusions configs/gate0b/manual_exclusions.jsonl
```

实际参数以脚本 `--help` 和冻结合同为准，不要复制示例后跳过 dry-run/小样本检查。

审计至少执行以下检查：

1. 原始行是否为合法对象，必需字段是否存在且类型正确。
2. 候选回答与 verifier 结果是否能一一对应。
3. 是否存在至少一个通过验证的候选回答。
4. 选中的回答是否完整、非空、推理闭合。
5. 题目 hash `sample_id` 是否重复。
6. 题目和回答联合 hash `content_sha256` 是否稳定。
7. 是否命中人工排除规则或评测污染规则。
8. 应用 Qwen chat template 后，input IDs、assistant labels 和结束 token 是否正确。
9. token 长度是否超过冻结上限。
10. train/validation split 是否由稳定 hash/salt 决定，而不是依赖输入顺序。

## 4. 候选回答如何选择

正确流程不是“每条候选都作为一个 SFT 样本”，而是：

```text
一条题目
-> 检查 candidates 与 verifier
-> 找到符合冻结选择规则的 verified generation
-> 选择一条作为 assistant completion
-> 记录候选索引和来源
-> 构造唯一训练样本
```

若候选为空、验证字段矛盾、找不到通过候选或结构不合法，应进入 `rejected.jsonl`，并保留
明确 rejection code。不要为了扩大数据量自动降级规则。

## 5. 人工复核

自动审计解决结构和可编程规则，人工复核解决语义质量：

- 题目是否完整且可回答；
- 回答是否真的对应题目；
- 推理是否自洽；
- 最终答案是否清楚；
- 特殊 token、乱码或模板内容是否异常；
- verifier 通过是否与人的判断一致；
- 是否存在明显评测污染或答案泄漏。

复核命令将决定写入新文件，不直接修改 `review_samples.jsonl`：

```bash
uv run python scripts/review_audit_samples.py \
  --review-samples artifacts/<artifact-id>/review_samples.jsonl \
  --data-manifest artifacts/<artifact-id>/data_manifest.json \
  --output evidence/<reviewer>-review-decisions.jsonl \
  --reviewer <reviewer-id>
```

验收时需要同时检查：抽样数是否达到约定、是否全部有决定、是否有 unresolved 项、review
文件 hash 是否进入 server bundle manifest。

## 6. 为什么不静默截断长样本

运行时 `truncation=True` 会让一条样本后半部分消失，可能截掉最终答案、推理结束或
assistant EOS。训练仍可运行，但监督语义已经改变，而且很难从日志发现。

本案例实际发生：

```text
最大样本 22,295 token
-> A100-80GB 最长样本训练探针失败
-> 没有静默截断
-> 生成 16,384-token 派生 artifact
-> 超长样本单独隔离
```

最终结果：

| 文件 | 条数 | 说明 |
| --- | ---: | --- |
| `train.jsonl` | 61,224 | 正式训练集 |
| `validation.jsonl` | 1,968 | 正式验证集 |
| `length_filtered_train.jsonl` | 984 | 超长训练样本 |
| `length_filtered_validation.jsonl` | 32 | 超长验证样本 |
| `rejected.jsonl` | 29,525 | 其他审计拒绝样本 |
| `review_samples.jsonl` | 50 | 人工复核样本 |

“过滤”和“截断”是两个不同实验。过滤改变样本集合；截断改变样本内容。两者都必须版本化。

## 7. Artifact 冻结

冻结不等于把目录改成只读，而是让以下身份不可被悄悄改变：

```text
source dataset + revision
tokenizer/chat template + revision/hash
审计代码 commit/runtime-tree hash
审计参数
人工排除文件 hash
train/validation/rejected/review 文件的 count、bytes、SHA-256
split 和 review 的 salt
污染检查数据 revision
```

冻结后重新计算服务器上的真实文件 SHA-256，并与 manifest 比较。只比较文件名或行数不够。

## 8. 验收门

- [x] 全量审计正常结束，manifest 终态为成功。
- [x] 同输入重复执行得到相同核心 artifact hash。
- [x] 所有 review samples 均有明确决定。
- [x] train/validation 没有重复 `sample_id`。
- [x] 每条保留样本至少有一个有效 assistant token。
- [x] EOS 和 chat-template 边界经过抽样核验。
- [x] 超长样本被显式隔离，没有运行时静默截断。
- [x] 服务器文件 hash 与本地 manifest 一致。
- [x] 正式训练配置只引用冻结 artifact。

这些门在本案例中已关闭。下一次数据或 tokenizer 变化时，任一项不满足仍不能进入 B0 或
正式训练。

## 9. 可复用边界

可复用的是审计分层、人工复核、hash、manifest 和派生 artifact 机制。候选选择、污染规则、
长度限制、chat template 和内容质量标准必须针对新数据重新设计。
