# GSM8K 人工复核批次 01：回归样本 19-27

## 1. 批次范围

本批按固定抽样文件中的 `review_index=19..27` 复核 9 条 regressed 样本。所有判断写入
`review_v1/decisions.jsonl`，并通过 schema、样本哈希、row index 和 outcome 一致性检查。

## 2. 逐题判断

| Review | Row | 主要变化机制 | 核心证据 |
| ---: | ---: | --- | --- |
| 19 | 1221 | understanding_change | S1 误读 8:00 到 11:00 及半小时休息，并继续产生单位和算术错误 |
| 20 | 433 | reasoning_change | B0 也错误地使用 180 个剩余日，但取整后碰巧得到 5；S1 改成 360 日后得到 9 |
| 21 | 645 | reasoning_change | S1 算出单件折后价 14，却遗漏 4 件衬衫的乘法 |
| 22 | 834 | understanding_change | S1 把“Farm Y 卖出两倍数量”的参照量从 10 误解为 Farm Y 原有的 45 |
| 23 | 307 | understanding_change | S1 计算尚需金额时遗漏已有的 80 美元储蓄 |
| 24 | 1040 | reasoning_change | S1 的流水账中遗漏生日获得的 23 张贴纸 |
| 25 | 451 | multiple_changes | S1 同时误读每位客人吃 3 个半蛋、破坏蛋与半蛋换算、重复延展并触及 256-token 上限 |
| 26 | 534 | understanding_change | S1 只为 24 位客人计算披萨，遗漏 12 位队员和 3 位教练 |
| 27 | 702 | arithmetic_change | 方法和费用都正确，但把 350000+17500+42000 算成 419500 |

## 3. 批次质量检查

- 9 条均为高置信度判断；
- 9 条人工最终答案判断均与 qem 一致，未发现 scorer error；
- 8 条 S1 正常终止，1 条达到 256-token 上限并在句中停止；
- 1 条 B0 虽然 qem 正确，但推理本身无效，属于“幸运命中最终答案”；
- 本批没有把所有数值错误笼统归为 arithmetic error：遗漏变量和错误参照量分别归入
  understanding/reasoning，只有 row 702 是清晰的局部加法错误。

## 4. 分析过程中的证据修正

第一层最初把 LightEval `truncated=0` 解释为生成没有截断。复核 row 451 时发现 S1 文本
在句中停止，且生成长度恰好为 256。回查 invocation manifest 后确认
`max_new_tokens=256`，进一步统计得到 105/1319 条 S1 生成触及上限，且 105 条全部 qem
错误。

因此已修正分析脚本、机器可读结果和第一层报告：LightEval 的 `truncated` 字段不能替代
“生成是否达到 max_new_tokens”的统计。这个纠错本身也是评测工程经验，后续所有生成评测
都要同时记录 invocation budget 和实际 completion token 长度。

## 5. 只能形成的初步假设

本批是 coverage-first 样本，不能用 `4/9` 等比例估计全部回归的原因分布。目前只形成三个
待后续批次检验的假设：

1. 回归可能经常表现为遗漏题目中的实体、数量或汇总步骤，而不是复杂数学方法退化；
2. B0 的部分正确答案可能包含不可靠推理，单看 qem 会高估其推理质量；
3. S1 的重复和终止异常可能构成独立回归来源，需要在完整触顶集合中单独分析。

这些假设至少要经过剩余 9 条回归样本复核，才能决定是否值得扩大到总体或进入新实验。

