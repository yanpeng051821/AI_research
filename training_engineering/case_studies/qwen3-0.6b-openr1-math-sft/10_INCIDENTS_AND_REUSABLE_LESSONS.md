# 10 真实波折、事故复盘与可复用规则

## 1. 为什么要保留波折

这次工程不是按最初设想直线完成的。真实价值恰恰来自这些偏差：它们暴露了数据合同、
运行环境、资源边界、阶段定义和评测闭环中原先看不见的假设。

记录事故不是为了堆报错文本，而是回答七个问题：

```text
当时想做什么
实际发生了什么
故障发生在哪一层
用什么证据定位
做了什么修复
修复改变了什么合同
它是否影响已有实验结论
```

如果最后一个问题不回答，就容易把环境失败误判成算法失败，或者把一次不合格运行包装成
实验成功。

## 2. 事件时间线

| 阶段 | 原计划 | 实际波折 | 分类 | 修复 | 对结论的影响 |
| --- | --- | --- | --- | --- | --- |
| 历史 baseline | 复用记忆中的 MATH-500 结果 | 约 10% 或 26% 至 28% 的历史说法没有命令、配置和逐样本证据 | 实验合同 | 在冻结合同下重跑完整 B0，得到 pass@1:1 `43.2%`、pass@1:4 `44.9%` | 历史值被废弃，不能参与比较；新 B0 可用 |
| 全量数据冻结 | 使用全部审计后数据训练 | 原最长样本 22,295 token 在 A100 80GB 上 OOM | 数据与资源 | 明确生成 16K derived artifact，排除超长样本，不静默截断 | 改变了正式训练数据合同，必须重算样本数、step 和 hash |
| 合同计算 | 按旧记录准备训练 | 文档残留 62,208 条、486/487 updates，与真实 61,224 条不一致 | 配置与文档 | 从冻结 artifact 重算为 479 updates、15 warmup | 旧调度配置不可用；重新 dry-run 后合同可用 |
| 服务器环境 | checkout 后直接运行 | 旧 `.venv` 的 editable install 从旧数据盘导入 `post_training_core` | 环境身份 | 新 checkout 执行 `uv sync --frozen`，检查 `module.__file__`，把 source tree hash 写入 manifest | 旧失败不能归因于算法；修复后的运行身份才可信 |
| CPU dry-run | 用正式配置做数据预检 | runner 在判断 `--dry-run` 前构造 BF16/GPU `SFTConfig`，CPU 环境被 TRL 拒绝 | 代码职责 | 先用纯函数计算 schedule 和写 identity，真实训练时才构造 GPU config | 关闭了数据/合同预检，但不能据此声称 GPU 路径通过 |
| 数据索引 | dry-run 写冻结顺序 | 生成 sample order 时再次打开、解码全部 JSONL | 性能与职责 | indexed dataset 在建索引时保存 `sample_id`，增加 `sample_id_at` | 不改变训练语义，降低重复 I/O 与额外失败面 |
| DataLoader | 用两个 worker 提高吞吐 | 长 JSONL 读取出现解析失败，随后出现 `Too many open files` 和共享内存 FD 错误 | 系统资源 | 证明主进程可读同一数据；pilot 冻结 `dataloader_num_workers=0` | worker launch 无效，但不说明数据坏了或 loss 错了 |
| attention backend | FP32 加载后进入 BF16 训练 | FlashAttention 给出 dtype/backend 警告 | 性能后端 | 固定依赖并检查真实 backend；无数值异常时不擅自换算法 | 仅有警告时不推翻训练正确性，但必须确认正式路径 |
| pilot 边界 | 执行有限步 pilot | 初始配置没有清楚表达预期 `max_steps`，一次运行又因成本评估被人工中断 | 阶段与流程 | 配置和 CLI 同时明确 `--stop-after-steps`；manifest 标记 paused/failed/completed | 被中断运行只证明部分链路可执行，不能称为完整 pilot |
| 显存判断 | 通过 `nvidia-smi` 判断是否泄漏 | 一次观测接近 79,001 MiB，恢复运行外部峰值约 51,471 MiB，无法解释 allocated/reserved | 可观测性 | runner 加入 phase-aware allocator JSONL，要求覆盖第一个 AdamW step | 现有运行证明曾经 fit，但不足以做泄漏诊断和容量规划 |
| 20-step smoke | loss 下降后进入下一阶段 | NLL 从 `0.8050` 降到 `0.6754`，但 4 个 prompt 在 512/2048 token 内 EOS 为 0/4 | 模型行为 | 把训练链路门与生成健康门拆开记录，不用低 NLL 覆盖 EOS 失败 | 训练路径通过，生成门失败；不能启动正式一轮训练 |
| 100-step pilot | 从头完成 100 steps | 中间遇到 worker、source path 和人工中断，最终从 checkpoint-50 恢复到 100 | 恢复工程 | 使用 TRL 原生 checkpoint，验证 global step、optimizer 和 scheduler 连续 | 最终 pilot 可作恢复和趋势证据，不是正式 S1 checkpoint |
| B0 regression | 缓存齐全后离线评测 | `HF_HUB_OFFLINE=1` 阻止 LightEval 解析 `ai2_arc` 等数据集短名 | 评测环境 | payload 使用缓存，但允许 Hub 名称解析；固定超时和 `HF_HOME` | 失败尝试无分数效力；修复后结果可作为 B0 |
| vLLM 启动 | 直接运行 LightEval | CUDA 已在父进程初始化，fork worker 报不能重新初始化 CUDA | 进程模型 | 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 属运行时故障，不是模型能力变化 |
| Triton 链接 | vLLM 编译 kernel | 主机有 `libcuda.so.1`，但链接器找不到 `-lcuda` | 系统链接 | 设置 `LIBRARY_PATH=/usr/local/cuda/lib64/stubs` | 修复后重新评测；失败输出不能比较 |
| MATH-500 | 与其他 suite 类似地估时 | 少量样本不 EOS，跑到 32K token 上限，长尾消耗约一半 GPU 时间 | 评测成本 | 保存 completion 长度分布和 EOS 率，给评测单独预算 | 不能因成本临时降低 S1 输出上限，否则 B0/S1 失配 |
| pilot MATH | 启动后等待结果 | invocation 长期停留 `running`，后来又出现 vLLM cache 初始化失败，未形成可靠终态 | 证据生命周期 | 正式 S1 前要求 B0 与 checkpoint 双边小样本入口预飞行，失败也写终态 manifest | pilot 没有可用 MATH 分数，正式评测门仍未关闭 |
| SSH 观察 | 从另一会话检查后台训练 | 一次密码字符转录错误导致误以为服务器无法连接 | 访问与观察 | 复核主机、端口和凭据，再直接检查 PID、日志、manifest | 不影响已经运行的进程，不能记录成训练失败 |

## 3. 最关键的五次认知修正

### 3.1 “数据合法”不等于“数据可训练”

22,295-token 样本在 schema、角色顺序和内容上可以完全合法，但在当前单卡资源和训练栈下
无法完成一次真实 optimizer step。数据门禁必须同时包含：

```text
语义/结构质量
+ tokenizer 后长度分布
+ 目标硬件上的最坏样本容量
```

长度策略是数据合同的一部分。过滤、截断和 packing 会改变训练分布，任何一种都必须显式
版本化。本案例选择 16K 过滤，是为了走通可信流程，不代表 16K 对其他模型或任务最优。

### 3.2 “代码文件是新的”不等于“进程运行的是新代码”

Python editable install、`PYTHONPATH`、当前工作目录和多个 clone 会共同决定导入来源。只有
`git status` 和 `git rev-parse HEAD` 不足以证明进程实际使用该 checkout。

最低验证是：

```bash
uv run python -c "import post_training_core; print(post_training_core.__file__)"
```

更强的做法是把运行时代码树 hash 和 import path 写入 run manifest。环境身份错误时，首先
修环境，不要修改 loss 或数据来“试试看”。

### 3.3 “GPU 显存占满”不等于“计算图泄漏”

`nvidia-smi` 看到的是进程持有的设备内存，不能区分 PyTorch 当前 tensor allocation 和缓存
池 reservation。判断泄漏至少需要同时观察：

```text
memory_allocated
memory_reserved
max_memory_allocated
max_memory_reserved
mem_get_info free/total
phase + optimizer step + micro-step
```

尤其要观察第一个 `optimizer.step()`，AdamW 常在此时第一次创建 `m/v`。如果 allocated 在
每个 micro-batch 都单调增加，才应重点检查是否把带梯度的 loss/logits 放进长期容器。

### 3.4 “loss 下降”不等于“模型达到目标”

20-step smoke 与 100-step pilot 都说明模型更会预测目标 validation completion。它们没有单独
证明数学正确率提升，也不能覆盖 EOS、格式、解析率和通用能力回退。

证据层次必须分开：

| 证据 | 能支持什么 | 不能支持什么 |
| --- | --- | --- |
| tiny numerical tests | loss、mask、梯度和累积语义 | 真实任务能力 |
| smoke/pilot loss | 训练链路和拟合趋势 | 正式效果大小 |
| validation NLL | 目标数据分布拟合 | 数学答案一定正确 |
| MATH/GSM8K | 指定评测合同下的任务变化 | 所有通用能力 |
| regression panel | 被选通用任务是否回退 | 未测能力没有变化 |
| paired error analysis | 哪些样本和错误类型变化 | 合同之外的泛化 |

### 3.5 “任务跑完”不等于“实验完成”

没有终态 manifest、退出码、配置、逐样本输出和对照身份的分数，不属于正式实验结果。
同样，进程失败并不等于实验假设失败。先区分：

```text
代码/算法失败
数据失败
环境失败
资源失败
评测合同失败
证据落盘失败
```

不同类型的失败有不同含义，不能全部用“重跑一次”处理。

## 4. 分层诊断方法

遇到问题时按数据流方向定位，不要同时修改多个变量：

```text
输入 identity
-> JSONL/offset/sample_id
-> tokenizer/mask/collator
-> model forward/loss
-> backward/accumulation
-> optimizer/scheduler
-> checkpoint/resume
-> generation/evaluation
-> comparison/report
```

### 4.1 进程启动即失败

检查顺序：

1. traceback 中实际导入文件的绝对路径。
2. `git rev-parse HEAD`、runtime tree hash、config hash。
3. Python、PyTorch、CUDA、TRL、Transformers 和 FlashAttention 版本。
4. 数据与模型 snapshot 是否命中固定 hash/revision。

此时不要先降低长度、换 optimizer 或修改 batch，这些动作会引入新变量。

### 4.2 DataLoader worker 失败

先在主进程逐条读取同一 offset 和 JSON 行。若主进程稳定而 worker 报文件描述符、共享内存
或 EOF 类错误，应检查：

```text
ulimit -n
num_workers
persistent_workers
每个 worker 是否重复持有文件句柄
/dev/shm 与进程启动方式
```

只有主进程也在同一记录失败，才优先怀疑 artifact 损坏。

### 4.3 OOM 或显存高位

按 phase 看 allocator 记录：模型放置、首个 forward、backward、首个 optimizer step、
zero_grad、evaluation 和 checkpoint。分别判断参数、activation、gradient、optimizer state、
KV cache 或缓存池占用。

一次只改变一个容量变量，并重新版本化配置。不要同时改 `max_length`、batch、precision 和
attention backend，否则即使成功也不知道是哪一项解决了问题。

### 4.4 loss 正常但生成异常

检查：

- completion mask 是否覆盖 assistant 结束 token。
- tokenizer 的 EOS 与 chat template 结束 token 是否一致。
- 保存前后 `model.config.use_cache` 和 generation config。
- 输出是截断、空串、格式错误，还是答案错误。
- smoke 的生成预算是否足以观察停止行为。

训练链路门和生成健康门分别记录，不用其中一个代替另一个。

### 4.5 评测卡住或无终态

检查：

1. invocation manifest 是否停留在 `running`。
2. vLLM worker 的进程启动方式与 CUDA 初始化顺序。
3. KV cache 是否有可用显存，prompt 是否占满 context。
4. Hub 离线模式是否阻止数据集名称解析。
5. completion 长度、EOS 率和少量长尾样本。
6. failure manifest 是否包含命令、return code 和 traceback。

评测失败时不能只保留半截 stdout，也不能把缺失指标当作零。

## 5. 哪些波折改变了合同

不是所有修复都可以在同一个实验中无声完成。

| 修复 | 是否改变实验合同 | 处理方式 |
| --- | --- | --- |
| 修正旧 editable import，使其指向冻结 commit | 否，恢复了预期身份 | 重新运行失败阶段并保存环境证据 |
| `num_workers: 2 -> 0` | 通常不改变样本与优化语义，但改变吞吐实现 | 固定配置并验证顺序、batch 和结果合同 |
| 增加 allocator 日志 | 否，只增加可观测性 | 通过测试确认不改变训练目标 |
| 22,295 token 改为 16,384 token 过滤 | 是，改变数据集 | 生成新 derived artifact、hash、样本数和 schedule |
| 改 B0/S1 generation 参数 | 是，改变评测合同 | 建立新版本并重跑 B0，不能只改 S1 |
| pilot 人工中断后恢复同 checkpoint | 不改变合同，前提是状态完整且 identity 相同 | 新 attempt 记录，验证 optimizer/scheduler/sample position |
| 修改 loss、mask 或 token normalization | 是，改变训练目标 | 回到数值测试和 first-batch shadow，建立新实验版本 |

## 6. 可复用的阶段门禁

未来的 SFT、mid-training、DPO 或 RL 实验可以复用以下骨架：

```text
问题与成功标准
-> 数据来源、审计、泄漏检查、人工复核
-> 冻结 artifact 与身份
-> 核心计算数值测试
-> 框架映射与 first-batch shadow
-> B0 同合同基线
-> 环境和最坏样本预飞行
-> smoke
-> bounded pilot + resume
-> 正式训练 GO/NO-GO
-> 正式训练
-> 同合同配对评测
-> 错误分析和下一轮决策
```

可复用的是流程和证据结构，不是本案例的 16K、`4e-5`、128 累积或 A100 80GB。换模型、
数据、训练目标和硬件后，必须重新证明容量与优化合同。

## 7. 每次服务器运行的最低证据包

```text
run_manifest.json
attempt manifest
resolved config
source commit and runtime tree hash
data/model/tokenizer identity
sample order identity
environment and package versions
stdout.log / stderr.log / return code
CUDA allocator telemetry
checkpoint metadata
validation and generation health
evaluation manifest and per-sample records
incident note when a gate fails
```

服务器销毁前，应将小体积证据提交 Git，将大体积权重和原始输出保存到持久化盘，并在 Git
里记录路径与 SHA-256。仅写“跑通了”或“显存 79GB”无法支持复现。

## 8. 当前案例的真实结论边界

案例收尾时可成立的结论：

- 核心 completion-only SFT loss、mask、gradient accumulation 和 TRL first-batch 对齐已经
  通过本地与真实模型测试。
- 数据经过自动审计、人工复核和确定性 16K 派生，训练与验证 artifact 已冻结。
- 完整 B0 已建立，历史不可核对的 MATH 数字已被替代。
- 20-step smoke 与 100-step TRL resume pilot 证明训练、保存、恢复和 validation NLL 下降。
- 100-step pilot 的全量 validation NLL 为 `0.5952246359`，相对同合同 B0
  `0.7804831795` 下降约 23.74%。这是拟合趋势，不是正式能力结论。
- 正式 S1 已在 A100 80GB 上完成 61,224 条、479 updates 和 15 warmup updates，训练、
  checkpoint、最终模型保存及 CUDA telemetry 均有终态证据。
- 正式 S1 的 token-weighted validation NLL 从 `0.780483` 降至 `0.546614`，GSM8K
  从 `0.476118` 升至 `0.514784`。
- 59 项通用回归面板宏平均下降约 1.49 个百分点，说明目标改善伴随可观测副作用。
- 完整 MATH-500 因长生成导致预计成本超出预算而被主动中止；该行为作为评测合同事故，
  已推动预算探针、显式 stop token 和受保护入口进入 v2 合同。

当前不能成立的结论：

- 数学能力已经全面提高。当前只有 validation 拟合和 GSM8K 的正向证据。
- 通用能力没有回退。现有回归面板反而给出整体负向证据。
- 80GB 显存在所有模型、长度和后端上都有充足余量。本结论仅适用于冻结配置。
- 正式 MATH-500 已形成可比较结果。smoke 受截断影响，full 没有完成。
- EOS 配置差异已经被彻底解释。它仍是后续分析和推理适配需要验证的因素。

## 9. 从案例提炼 playbook 的条件

这份案例已具备从真实证据提炼 `training_engineering/playbooks/` 的条件。提炼时应满足：

1. 规则在本案例至少被真实执行过，不是只写过方案。
2. 写清适用前提和反例，不把 Qwen3/OpenR1 参数当普遍标准。
3. 对应脚本、测试和 artifact schema 有稳定入口。
4. 能被下一个不同模型或训练阶段复用并再次验证。

候选 playbook 包括数据 artifact 冻结、服务器预飞行、CUDA allocator 观测、checkpoint
恢复验证、评测成本探针和同合同配对评测。它们仍需在下一种模型或训练阶段复用后，才能
从案例经验升级为跨项目标准。
