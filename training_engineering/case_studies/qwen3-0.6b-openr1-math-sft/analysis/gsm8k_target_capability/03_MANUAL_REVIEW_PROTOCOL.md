# GSM8K 第三层：人工复核协议

## 1. 复核目的

人工复核不是重新凭感觉给模型打分，而是解释 B0 到 S1 为什么发生变化。每条记录同时阅读
题目、gold、B0 输出和 S1 输出，先独立判断两个输出的错误位置，再选择一个主要变化机制。
qem 只作为已知评测结果，不替代对推理过程的检查。

## 2. 单侧输出标签

### 题意理解 `understanding`

- `correct`：正确识别已知量、目标量和约束；
- `incorrect`：误读目标、遗漏关键条件或建立了错误的数量关系；
- `unclear`：输出不足以判断。

### 推理方法 `reasoning`

- `valid`：方法和主要推理链成立；
- `invalid`：使用了不成立的公式、关系或逻辑步骤；
- `incomplete`：方向可能正确，但缺少完成结论所需的关键步骤；
- `not_applicable`：没有形成可识别的推理；
- `unclear`：无法可靠判断。

局部加减乘除算错但方法成立，标为 `reasoning=valid`、`arithmetic=error`，不要把所有数值
错误都归入推理错误。

### 算术 `arithmetic`

- `correct`：可见数值运算正确；
- `error`：至少一个可定位的算术或代数计算错误；
- `not_applicable`：没有实际计算；
- `unclear`：无法可靠检查。

### 最终答案 `final_answer`

- `correct`：最终提交值与 gold 一致；
- `incorrect`：存在明确最终值，但不正确；
- `missing`：没有可识别的最终答案；
- `ambiguous`：出现多个冲突值，无法确定提交值。

### 格式 `format`

- `parseable`：最终答案可由当前 GSM8K scorer 稳定提取；
- `unparseable`：答案内容可能存在，但没有形成 scorer 可提取的提交值；
- `ambiguous`：多个候选答案可能造成提取歧义。

### 终止行为 `termination`

- `normal`：推理正常结束；
- `abrupt`：输出在完成答案前异常停止；
- `repetitive`：明显重复、循环或无效延展；
- `length_limited`：有证据表明触及生成预算；
- `unclear`：无法可靠判断。

## 3. 配对变化标签

`primary_mechanism` 选择最能解释 qem 状态变化或持续失败的一个因素：

- `understanding_change`：关键条件或目标量的理解发生变化；
- `reasoning_change`：解题方法或关键逻辑关系发生变化；
- `arithmetic_change`：方法基本相同，局部计算变化决定结果；
- `final_answer_selection_change`：推理中存在相关数值，但最终选取或汇总值发生变化；
- `format_or_extraction_change`：主要差异是 scorer 能否提取答案；
- `termination_or_repetition_change`：异常停止、重复或延展改变结果；
- `multiple_changes`：至少两个因素共同作用，无法合理确定单一主因；
- `no_material_change`：两侧实质错误相同，仅措辞变化；
- `unclear`：证据不足。

`secondary_mechanisms` 只记录有直接文本证据的次要因素，不能用作不确定性的垃圾桶。拿不准
时使用 `unclear` 并降低置信度。

## 4. Scorer 审核

`scorer_error` 用来区分真实能力变化和评测解析问题：

- `none`：人工判断与 B0/S1 qem 一致；
- `baseline`：只怀疑 B0 qem 误判；
- `s1`：只怀疑 S1 qem 误判；
- `both`：两侧都可能误判；
- `unclear`：无法确认。

不能因为模型没有写 `####` 就自动认定 scorer 错误。如果模型没有明确提交正确答案，qem
判错是合同预期行为。

## 5. 复核顺序和批次

1. 标签 schema 和本协议先冻结；
2. 按 `review_index` 阅读，不替换“不典型”样本；
3. 第一批复核 9 条 `regressed`；
4. 每批结束后先做标签一致性检查，再进入下一批；
5. 完成 18 条回归后，再处理 improved，最后处理 both_wrong；
6. 原始抽样文件保持只读，判断写入独立 `decisions.jsonl`。

抽样有意覆盖罕见层，人工标签比例不能直接外推到总体。报告必须同时引用对应分层的总体
数量，并把观察性结论写成后续可验证假设。

