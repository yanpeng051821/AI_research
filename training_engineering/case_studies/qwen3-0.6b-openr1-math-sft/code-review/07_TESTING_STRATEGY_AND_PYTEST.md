# 07 测试策略与 pytest

## 本次要解决的问题

前六个主题都使用测试证明训练合同，但“能运行现有测试”不等于“能独立设计测试”。本主题
系统学习 pytest 的发现、组织、隔离、断言和失败诊断，并把它们应用到本项目的数据、数值、
状态恢复、CLI 和框架集成测试中。

本主题放在业务代码回顾之后：先知道每条合同为什么重要，再学习如何选择最低成本且足够有力
的测试证明它。目标不是记忆 pytest API，而是能够从风险反推测试层级和断言。

## 学习范围

1. pytest 如何发现测试模块、函数和参数化 case。
2. Arrange-Act-Assert 与“一项测试只证明一个主要行为”。
3. `fixture`、`tmp_path`、`monkeypatch`、`capsys` 和资源清理。
4. `pytest.raises`、`pytest.mark.parametrize` 与失败分支设计。
5. `assert`、`pytest.approx`、`torch.testing.assert_close` 的适用边界。
6. fake、stub、tiny model 与真实集成测试分别能证明什么。
7. 纯函数单测、模块集成、CLI 子进程和端到端 smoke 的成本梯度。
8. `-q`、`-x`、`-k`、节点 ID、collection error 和 traceback 的阅读方法。
9. 如何先写失败测试，再完成修复和回归验证。
10. 如何识别脆弱测试、重复覆盖、错误 mock 和“测试通过但合同未被证明”。

## 本项目的测试地图

| 层级 | 代表文件 | 主要证明内容 |
| --- | --- | --- |
| 纯计算 | `test_sft_loss.py` | shift、mask、reduction、梯度 |
| 模块行为 | `test_sft_data.py`、`test_engine.py` | padding、累积、更新时序和状态 |
| artifact / 状态 | `test_artifact_data.py`、`test_checkpointing.py` | 文件读取、身份和恢复边界 |
| 框架对齐 | `test_reference_alignment.py`、`test_trl_training.py` | 独立实现与锁定框架行为 |
| CLI 集成 | `test_cli.py`、`test_trl_reference_cli.py` | 参数、退出码、文件产物和错误路径 |
| 真实依赖集成 | `test_qwen_tokenizer_integration.py` | 锁定 tokenizer/chat template 的真实行为 |

## 计划中的动手练习

1. 从一个现有失败日志判断它属于收集失败、测试准备失败、行为失败还是断言设计错误。
2. 为纯函数补一个参数化边界测试，并解释每个 case 为什么必要。
3. 使用 `tmp_path` 测试一次 artifact 或 checkpoint 写入，不污染真实运行目录。
4. 使用 fake/stub 验证调用时序，再说明它不能替代哪一项真实集成测试。
5. 为一个 CLI 门禁先写 RED 测试，检查退出码、stderr 和“不应产生的文件”。
6. 对同一合同设计单元测试与集成测试，比较证据强度和运行成本。

## 验收标准

- [x] 能解释 pytest 从命令到收集、fixture、执行、teardown 和报告的过程。
- [x] 能根据合同选择断言与测试层级，而不是照抄已有测试结构。
- [x] 能独立使用参数化、临时目录、异常断言和 CLI 子进程测试。
- [x] 能区分 fake 对调用合同的证明与真实依赖对行为合同的证明。
- [x] 能从 traceback 找到第一个属于本项目的失败位置。
- [x] 能完成一次 RED-GREEN-REGRESSION，并说明剩余未覆盖风险。

## 快速问答与学习记录

学习日期：2026-09-22。

本轮使用项目现有测试快速建立了测试地图：纯计算测试负责局部数值合同，模块测试负责文件、
状态和协作行为，CLI 子进程测试负责真实入口、退出码、标准流与副作用，真实依赖集成测试负责
验证 fake 无法证明的外部行为。pytest 的执行链路是配置读取、收集、fixture setup、测试执行、
teardown 和报告；collection error 与行为断言失败发生在不同阶段。

快速问答确认：

1. `tmp_path` 是 pytest 提供的独立临时目录，类型为 `pathlib.Path`，避免测试污染正式产物。
2. `monkeypatch` 只证明替换接口下的调用和分支合同，不能代替真实第三方依赖的行为验证。
3. CLI 门禁需同时验证退出码、`stderr` 和不应产生的文件，才能证明失败原因与副作用边界。
4. 参数化 case 可被独立收集、执行和报告；函数内部循环不能提供同等的 case 隔离与节点 ID。
5. 默认配置包含 `-m "not integration"`，因此真实 Qwen tokenizer 测试需要显式选择。

本轮沿用了主题 06 已完成的 RED-GREEN-REGRESSION：先用测试暴露“指标不变时丢失已变化
prediction”的风险，再修正输入构造与实现回归。代表性验证覆盖配置参数化、checkpoint 失败
清理、CLI 子进程门禁和评测回归，共 16 个测试通过。真实 tokenizer integration 本轮未执行；
它仍是验证锁定 Qwen chat template 行为的独立证据层。
