# 技术参考

[返回总览](README.md) · [执行指南](EXECUTION_GUIDE.md) · [冻结与验收规范](CONTRACTS.md)

本页供按需查阅，不是另一份并行路线。实习生按执行指南行动，再跳到相应小节。
涉及工具能力的要求由准备方满足，**不安排实习生手动实现脚本、算法或训练框架**。

## 导航

1. [存储与版本](#r1)
2. [设备与依赖](#r2)
3. [数据处理](#r3)
4. [基线与评测](#r4)
5. [Smoke 启动](#r5)
6. [上游行为](#r6)
7. [训练观察](#r7)
8. [保存与恢复](#r8)
9. [规模与对照](#r9)
10. [SFT 细则](#r10)
11. [缺陷分类](#r11)
12. [后续阶段](#r12)

<a id="r1"></a>
## 1. 存储与版本

### 1.1 目录建立

先请管理员指出服务器上**重启或换实例后仍保留的数据盘**。执行 `df -h`、`findmnt` 查看挂载信息；
看不懂时把输出交给管理员确认，不能自行把 `/tmp` 或容器可写层当长期存储。

建议结构如下。路径是约定，不代表服务器已经存在这些目录。

```text
本案例/
├── README.md                         课题总览
├── EXECUTION_GUIDE.md                逐步执行与验收
├── CONTRACTS.md                      共用冻结与团队交付规范
├── TECHNICAL_REFERENCE.md            本技术参考
└── implementation/                   后续实习生建立，代码纳入 Git
    ├── docs/                         任务、环境、问题记录
    ├── scripts/                      数据、评测、检查入口
    ├── src/                          可测试的数据和训练实现
    ├── tests/fixtures/                少量合成测试数据
    ├── configs/                      已冻结的配置
    ├── reports/                      小型报告和汇总，不含原始敏感数据
    └── requirements/                 环境锁定和安装说明

<持久盘>/intern-testing-model/
├── reference/                        固定版本的上游源码
├── raw/                              原始下载，保留原样
├── staging/                          审计与转换的中间结果
├── frozen/                           冻结后的数据版本，只读使用
├── cache/                            模型及数据缓存
├── workspaces/                       上游 smoke 的独立运行副本
├── runs/                             每次实验独立目录
└── backups/                          待转存备份，不是唯一备份
```

在管理员确认路径后，建立目录。以下以 Bash 为执行环境；变量必须换成真实绝对路径：

```bash
# 直接执行：先修改这两个路径，不能原样使用占位符。
export PROJECT_ROOT="/实际代码目录/implementation"
export LAB_ROOT="/实际持久盘/intern-testing-model"
mkdir -p "$PROJECT_ROOT"/{docs,scripts,src,tests/fixtures,configs,reports,requirements}
mkdir -p "$LAB_ROOT"/{reference,raw,staging,frozen,cache,workspaces,runs,backups}
export HF_HOME="$LAB_ROOT/cache/huggingface"
```

将变量写入自己的本地环境配置，并在 `docs/storage.md` 记录实际映射、容量、保留规则。
密码、token、完整环境变量转储不能提交 Git。权重、大数据、缓存、虚拟环境和完整运行日志也不提交 Git；
提交代码、配置、数据清单、汇总和校验值即可。内部数据未来出现时另行审批存储与访问权限。

### 1.2 源码固定

本手册的 MiniMind 命令以本地已核对的提交为准：

```text
仓库：https://github.com/jingyaogong/minimind
提交：f659b55761b754d306bd140573493a6543cafd7f
```

在服务器的新目录执行，不要覆盖已有工作区：

```bash
# 直接执行：网络可用，目标目录尚不存在。
git clone https://github.com/jingyaogong/minimind.git "$LAB_ROOT/reference/minimind"
git -C "$LAB_ROOT/reference/minimind" checkout --detach f659b55761b754d306bd140573493a6543cafd7f
git -C "$LAB_ROOT/reference/minimind" rev-parse HEAD
```

将提交号记录到 `docs/references.md`。如果该提交无法获取，先解决版本获取问题，
或明确更新本手册及接口检查；不能默默切到最新 `main`。数据仓库版本另行固定，不能用代码提交代替数据版本。

<a id="r2"></a>
## 2. 设备与依赖

### 2.1 设备盘点

先进入管理员提供的训练容器或环境。向管理员索取镜像名称及 digest、驱动和 DTK 配套说明。
不要先运行 `pip install -U torch`，也不要把个人 Mac 的锁文件直接用于海光。
海光维护了自己的 PyTorch 适配包，具体版本须符合安装环境的配套关系，参见[官方说明](https://github.com/HYGON-AI/pytorch-das)。

检查系统和管理工具：

```bash
# 直接执行：只读检查。某个工具不存在不代表机器没有卡。
uname -a
df -h
command -v hy-smi
command -v rocm-smi
python --version
python -m pip show torch
```

如果 `hy-smi` 或 `rocm-smi` 存在，先读它的 `--help`，按该版本帮助查看设备清单、总显存和进程占用。
不同平台工具可能不同，不能把 `nvidia-smi` 当作海光机器必备命令。两个工具都找不到时，向管理员确认路径。

在已经安装兼容 PyTorch 的环境中运行：

```bash
# 直接执行：检查 PyTorch 看到的设备；尚未安装 torch 时先找管理员配置。
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("cuda_api_available:", torch.cuda.is_available())
print("visible_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, "total_GiB=", round(p.total_memory / 1024**3, 2))
PY
```

**怎么看结果**：`visible_device_count` 是当前进程可见卡数；每行 `total_GiB` 是相应设备可见的总显存，
不是当前空闲显存。容器只分配一张卡时，这个数字不等于整台物理服务器的卡数。
将 PyTorch 输出与管理工具对照，确认分配给你的卡以及已有进程占用。

部分兼容栈通过 `torch.cuda` 暴露设备接口，名字含 CUDA 不等于装着 NVIDIA 卡。
若结果是不可用或返回错误，先检查容器设备映射、权限和配套软件，不能把 CPU 回退误报为适配成功。

### 2.2 依赖安装

1. 保存安装前的包列表：`python -m pip freeze`，记录 Python 可执行文件路径。
2. 检查固定提交的 MiniMind `requirements.txt`，先只安装本轮需要的数据、tokenizer、模型依赖。
3. 让管理员提供海光 torch 的固定版本或 wheel，并用约束文件保护；检查其他包是否会替换它。
4. 安装后运行 `python -m pip check`，再次保存包列表，比较 torch 是否改变。
5. 将实际安装命令、镜像 digest、wheel 来源及校验值、版本差异写入 `docs/environment.md`。

不要为满足某个可选库而直接重装驱动。第一轮禁用 `torch.compile`、第三方 FlashAttention、
量化和多卡；后面只有确实需要时才单独验证。上游使用的 PyTorch SDPA 也要在本机测过，不能只看接口名称。

### 2.3 现有探针的检查范围

**准备方交付要求（T1），不是实习生开发任务。** 优先使用已有探针，接收设备、dtype、seed 和输出路径；结果包含 JSON 或可映射的文本记录，失败返回非零退出码。
按顺序完成以下检查，每通过一层再加下一层：

1. CPU 上创建固定小张量，保存输入和权重；相同值复制到目标设备。
2. 验证矩阵乘法、Embedding、RMSNorm、SiLU、交叉熵的前向结果和梯度。
3. 验证 causal attention：修改未来位置不能改变过去位置的输出；检查 SDPA 与普通实现的偏差。
4. 用一个小 dense 模型运行 forward、backward、AdamW 第一次 `step()`、清梯度。
5. 在同一小 batch 上连续更新，检查 loss 有限、参数确实变化、loss 趋势下降。
6. 保存权重并重新加载，比较固定输入输出。

先测 FP32，再按支持情况测 BF16/FP16。容差要按 dtype 写入测试并解释，不能为了通过随意放宽。
不要要求不同硬件逐 bit 相同，也不要接受 NaN/Inf。FP16 若需要 GradScaler，必须验证其行为。

显存记录至少包含：模型加载后、forward 后、backward 后、第一次 optimizer step 后、清梯度后。
兼容栈支持时记录 `memory_allocated`、`memory_reserved`、`max_memory_allocated` 和设备空闲量；
不支持的字段写 `null` 并注明原因，不填零。后台管理工具同时观测，区分缓存、活跃张量和其他进程。

<a id="r3"></a>
## 3. 数据处理

### 3.1 下载

使用 [MiniMind 官方数据仓库](https://huggingface.co/datasets/jingyaogong/minimind_dataset)。
第一轮候选文件为 `pretrain_t2t_mini.jsonl`；通用 SFT 后续使用 `sft_t2t_mini.jsonl`。
先确认文件列表和许可，不混用其他教程版本的文件名及超参数。

1. 在数据仓库取得实际 revision，记录在 `docs/references.md`。
2. 记录来源 URL、revision、文件名、许可说明和获取日期。
3. 下载到 `raw/minimind/<revision>/`，不要直接下载到训练输出目录。
4. 对下载文件计算 SHA-256、字节数和实际行数；保存 `source_manifest.json`。
5. 原始文件保留不变；修复、过滤和转换都输出到 `staging/`。

有 Hugging Face CLI 时可用：

```bash
# 直接执行：先将 MM_DATA_REV 替换为查询到的真实提交；hf 工具须已安装。
export MM_DATA_REV="实际数据提交号"
hf download jingyaogong/minimind_dataset pretrain_t2t_mini.jsonl \
  --repo-type dataset --revision "$MM_DATA_REV" \
  --local-dir "$LAB_ROOT/raw/minimind/$MM_DATA_REV"
sha256sum "$LAB_ROOT/raw/minimind/$MM_DATA_REV/pretrain_t2t_mini.jsonl"
```

CLI 下载工具可以放在独立工具环境，不要为了下载而污染训练环境。下载失败先排查网络、权限及存储，
不要换到来历不明的数据镜像而不记录。不要在报告中展示访问 token。

### 3.2 结构与重复

**操作入口由 T2 工具包交付。** 结构审计优先复用已有 `minimind-audit`，去重由交付的数据处理流程另行执行。大文件采用流式处理。
预训练记录最小格式：

```json
{"text": "一段待学习的自然语言文本。"}
```

按以下顺序检查，并保存各阶段去向；结构审计与去重分别对账：

1. UTF-8 解码、空行、JSON 是否有效，最外层是否是对象。
2. `text` 是否存在、是否为字符串、是否只有空白；可疑控制字符单独统计。
3. 生成 `source_id`，定位到原文件校验值及原始行号。
4. 对原始内容计算 `content_sha256`；另定义用于完全重复判定的规范化规则。
5. 第一轮只做保守规范化，例如统一换行；不要删除代码缩进、标点或改变大小写来“清洗”。
6. 完全重复只保留一个代表，保留所有来源映射；无法可靠判断的近重复先标记和抽检。
7. 结构审计输出 accepted/rejected，去重阶段输出 retained/duplicates；拒绝记录包含原因和来源。

结构审计写入 `staging/mm-pt-v1/audit/`：`accepted.jsonl`、`rejected.jsonl`、`report.json`（现有 CLI 命名）；
去重另写 `staging/mm-pt-v1/dedup/`。团队工具生成汇总 `audit_report.json` 或明确其字段映射。
两阶段的输入/输出/原因分布分别对账，不能把 duplicate 再从结构 accepted 中重复扣除。

准备方交付前的自检至少覆盖：坏 JSON、空文本、非字符串、重复内容、中文、空文件、最终一行无换行。
再固定随机种子抽查接受及拒绝样本，记录规则是否误伤，不把抽样复核文件再次拼入训练集。

### 3.3 分组划分

**T2 中的分组划分流程，由准备方给出真实命令。** 先划分，再切片或扩展样本，不能把同一原始文档的不同段落分到训练和测试。

1. 有文档/来源组 ID 时按组划分；没有时至少按完全重复内容的组划分，并注明近重复隔离不足。
2. 起步可按组做约 98%/1%/1% 的 train/validation/test 划分；数据很小时先保证验证和测试规模足够。
3. 固定 seed 和算法，将组到 split 的映射写入 `split_manifest.jsonl`。
4. 检查三个集合的组 ID 和完全重复内容交集均为空。
5. smoke、pilot、后续扩大训练只从 train 中抽；validation 用于调试和选择方案，test 留到方案冻结后的最终评估。
6. 同一数据版本中，小训练子集应能在大子集中找到。不要用文件前 N 行代表随机样本。

输出 `staging/mm-pt-v1/split/{train,validation,test}.jsonl`。本阶段的数据检查可以看测试标签和结构统计，
但不能根据测试集模型表现调规则、提示词或超参数。

### 3.4 Token 统计

**调用 T2 中已交付的统计入口。** 使用固定提交的 `model/` tokenizer，记录所有 tokenizer 文件的校验值。
逐条计算原始 token 长度，再模拟实际 Dataset 的 BOS/EOS、截断、padding 和 shift。

报告至少包含：样本数、去重后 token 总量、长度 p50/p90/p95/p99、超过各候选长度的样本比例、
被丢弃的 token 比例、padding 比例、shift 后有效监督 token 数。两种“长样本比例”不能混为一谈。

检查固定版本的 `dataset/lm_dataset.py`：预训练添加 BOS/EOS，右侧 padding，padding 标签为 `-100`；
模型内部做 causal shift。手工打印 3 条短样本的 token、输入、标签，确认没有重复 shift 或监督 padding。

第一轮 smoke 取 `max_seq_len=128` 仅为便宜排错，不是推荐的正式上下文长度。
正式长度要结合分布、截断损失和实测资源决定，不能只看模型支持的最大长度。
预训练截断是改变数据消费方式；SFT 截掉答案则可能直接破坏监督，两者分别设计规则。

### 3.5 冻结导出

**调用 T2 中已交付的冻结/导出流程。** 建立新目录 `frozen/mm-pt-v1/`，存在同名版本时拒绝覆盖。

```text
frozen/mm-pt-v1/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── smoke_train.jsonl                 从 train 固定抽取 128 条，起步建议
├── pilot_train.jsonl                 从 train 固定抽取，数量按预算填写
├── records.jsonl                     ID、来源、split、内容 hash、长度等旁路元数据
├── split_manifest.jsonl
├── data_manifest.json
└── audit_summary.json
```

交给上游 Dataset 的文件只需满足原生输入格式；来源和审计字段可以放在旁路文件，不丢失对应关系。
manifest 记录各文件 hash、记录数、有效 token 统计、来源版本、处理代码提交、规则和 tokenizer 版本。
新规则、新 tokenizer、新切分都生成新版本，不在已冻结目录原地修改。

### 3.6 已有审计命令

在现有学习工程及其已配置环境中，已存在以下 CLI；`LEARNING_ROOT` 指向其 `implementation/`。
它是可复用入口，不是海光安装命令；由准备方先确认运行环境，不直接移植该工程的 torch 锁版本。

```bash
# 准备方确认环境及路径后执行。
cd "$LEARNING_ROOT"
python -c 'from minimind_lab.audit_cli import entrypoint; entrypoint()' \
  "$LAB_ROOT/raw/minimind/$MM_DATA_REV/pretrain_t2t_mini.jsonl" \
  --dataset-type pretrain --output-dir "$LAB_ROOT/staging/mm-pt-v1/audit"
```

退出码 0 表示没有结构拒绝，1 表示发现拒绝记录，2 表示输入/执行错误。
码 1 需要查看报告而不是误判成程序崩溃；也不能认为码 0 就证明去重和质量审计已经完成。

<a id="r4"></a>
## 4. 基线与评测

### 4.1 操作顺序

**复用 T4 的批量 NLL/生成入口，具体命令由准备方交付。** 上游交互式 `eval_llm.py` 用于体验，
不是自动冻结的评测流程；本课题需单独保存逐样本结果。

1. 固定验证子集、tokenizer、截断规则、精度和 batch 策略。
2. 使用 D2-S6 留存的同一随机初始化权重跑验证；不是事后随便新建一个随机模型充当 B0。
3. 训练后加载导出权重，运行 `eval()` 和无梯度推理。
4. NLL 使用全部有效 token 的 `loss_sum / valid_tokens`，不要平均各 batch 的平均 loss。
5. 固定少量文本前缀，记录生成；起步使用 greedy、`max_new_tokens=128`，保存是否达到长度上限。
6. 写入 `eval/validation.jsonl`、`eval/summary.json` 和 `eval/generations.jsonl`。

评测输入绝不能含生成目标答案。预训练模型尚未进行指令对齐，不能只用“不会聊天”判定训练失败。
小型 smoke 的验证 loss 未下降也不自动说明代码错误，应结合小 batch 过拟合和数值测试定位。

### 4.2 初值一致性

B0 在 smoke 之前运行。准备方交付初值保存/核对工具，固定 seed、结构、tokenizer 及初始化顺序。
若原生入口重新随机初始化，必须证明与 B0 权重一致；若采用显式加载初值的补丁，记录补丁身份。
实习生运行和检查结果，不编写权重导出或评测逻辑。

### 4.3 重新加载

先明确训练参数 dtype 与导出 dtype。FP16 导出用对应精度的参考输出、相同输入和 eval 模式比较；
不要把原精度与导出精度存在正常偏差误判为坏文件，也不要无依据放宽容差。

<a id="r5"></a>
## 5. Smoke 启动

### 5.1 使用前提

这是**供准备方验证并交付的 Bash 模板**，不是实习生需要编写的作业，也不是已经完成的训练工具包。
它只包装固定官方入口，不替代 T2/T4 的数据和评测工作。

1. D1 已确认 BF16、`cuda:0` 和所需依赖；否则更换已验证配方。
2. T2 已生成恰好 128 条的 `frozen/mm-pt-v1/smoke_train.jsonl`。
3. 准备方已建立本轮 RUN_DIR，放入身份/配置/合同文件，完成初值一致性检查。
4. `preflight.json` 的 `overall_status` 为 `pass`；B0 的 `summary.json` 状态为 `completed`，
   并包含与本轮合同文件实际 SHA-256 相同的 `eval_contract_sha256`。原生格式不同由 T4 适配。
5. 全部路径已人工或工具核验。该模板不能防止其他进程在检查后修改输入，冻结文件应只读管理。

### 5.2 脚本模板

由准备方将以下内容作为已审查的 `run_smoke.sh` 交付。**通过 `bash run_smoke.sh` 启动，不要 source。**
其中的 `exit` 只结束子脚本，不关闭交互式 SSH shell。

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${LAB_ROOT:?设置持久盘 LAB_ROOT}"
: "${RUN_ID:?设置本次唯一 RUN_ID}"
export RUN_DIR="$LAB_ROOT/runs/$RUN_ID"
MM_RUN="$LAB_ROOT/workspaces/$RUN_ID"
PYTHON="$(command -v python)"

# RUN_DIR 由训练前准备建立；工作副本和启动标记必须是新的。
test -d "$RUN_DIR"
test ! -e "$MM_RUN"
test ! -e "$RUN_DIR/started_at.txt"
test -f "$LAB_ROOT/frozen/mm-pt-v1/smoke_train.jsonl"

"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
for name in ("config.resolved.yaml", "identity.json", "environment.json",
             "eval_contract.json", "expectation.json", "preflight.json"):
    if not (run / name).is_file():
        raise SystemExit(f"missing: {name}")
preflight = json.loads((run / "preflight.json").read_text())
if preflight.get("overall_status") != "pass":
    raise SystemExit("preflight has not passed")
baseline = json.loads((run / "eval/b0/summary.json").read_text())
contract_hash = hashlib.sha256((run / "eval_contract.json").read_bytes()).hexdigest()
if baseline.get("status") != "completed":
    raise SystemExit("B0 incomplete")
if baseline.get("eval_contract_sha256") != contract_hash:
    raise SystemExit("B0 contract mismatch")
PY

finish() {
    status=$?
    trap - EXIT
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/finished_at.txt"
    printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
    exit "$status"
}
trap finish EXIT
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$RUN_DIR/started_at.txt"
cp "$0" "$RUN_DIR/launcher.sh"
exec > >(tee -a "$RUN_DIR/launcher.log") 2>&1

git clone --no-hardlinks "$LAB_ROOT/reference/minimind" "$MM_RUN"
git -C "$MM_RUN" checkout --detach f659b55761b754d306bd140573493a6543cafd7f
cd "$MM_RUN/trainer"

cmd=("$PYTHON" -u train_pretrain.py
  --data_path "$LAB_ROOT/frozen/mm-pt-v1/smoke_train.jsonl"
  --save_dir ../out --save_weight pretrain
  --hidden_size 64 --num_hidden_layers 2 --use_moe 0
  --max_seq_len 128 --batch_size 2 --accumulation_steps 1
  --epochs 1 --learning_rate 0.0005 --grad_clip 1.0
  --device cuda:0 --dtype bfloat16 --num_workers 0
  --log_interval 1 --save_interval 32 --seed 42
  --from_weight none --from_resume 0 --use_compile 0)
{
    printf '#!/usr/bin/env bash\n'
    printf 'cd %q\n' "$PWD"
    printf '%q ' "${cmd[@]}"
    printf '\n'
} > "$RUN_DIR/command.sh"

set +e
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
if (( pipeline_status[0] != 0 )); then
    exit "${pipeline_status[0]}"
fi
exit "${pipeline_status[1]}"
```

### 5.3 检查结果

1. 检查 exit_code、起止时间和训练日志；上面固定配置应消费 128 条并完成 64 次更新。
2. 上游产物在 `MM_RUN/out/` 和 `MM_RUN/checkpoints/`，登记路径、大小与 hash 到 artifacts。
3. 运行交付的加载检查和训练后评测，形成团队报告。
4. command.sh 是取证用实际命令，不应直接重跑覆盖原实验；重跑建立新 run。
5. 该模板处理普通退出和命令失败，硬杀/机器掉电时 trap 可能不执行，按缺失退出记录处理。

### 5.4 参数边界

此版本没有 `--max_steps`；通过小数据与 epoch 限制本轮规模。
也没有有效的 GPU `--dtype float32` 分支，不能自行传它假装启用 FP32。
首次 smoke 使用 accumulation 1，避免把累积差异混入接线检查；后续按已说明的官方语义配置。

<a id="r6"></a>
## 6. 上游行为

本手册核对的上游版本有以下行为，必须读代码确认后再改：

- 日志中的 `step` 是 micro-batch 次数，不一定是 optimizer 更新次数。
- 学习率函数按 micro-batch 进度调用，原生方案没有 warmup；不能在报告中写成另一套 scheduler。
- 累积时每个 micro-batch 的平均 loss 除以固定累积数；尾部不足一个窗口也使用固定除数。
- 日志 loss 是当前 micro-batch 的值，不是全窗口有效 token 加权平均。
- `--save_dir` 不控制所有路径：恢复文件使用 `../checkpoints`，初始化权重默认从 `../out` 读取。
- 权重导出及恢复快照含有 FP16 模型副本；恢复不是自动保留全部原精度状态。
- 恢复文件没有显式保存完整 RNG 和冻结样本顺序合同；`from_resume=1` 找不到文件时可能从头开始。

这些是对固定提交已核对的行为，不是待证实猜测。实习生按以下文件定位理解即可：
`trainer/train_pretrain.py`、`trainer/train_full_sft.py`、`trainer/trainer_utils.py`。
不要求修改这些行为；采用不同实现或准备方补丁时记录差异，所有对照组保持一致。
升级提交后重新核对，不能把本记录直接套用到另一版本。

<a id="r7"></a>
## 7. 训练观察

### 7.1 现成入口

已有学习工程提供下列 CPU 小实验，准备方核对环境后可直接使用：

- `scripts/inspect_gradient_accumulation.py`：梯度累积。
- `scripts/inspect_token_accumulation.py`：token 数与累积。
- `scripts/inspect_gradient_clipping.py`：裁剪前后梯度。
- `scripts/inspect_scheduler.py`：LR 推进。
- `scripts/inspect_training_resume.py`：教学恢复示例。

```bash
# 在已有配置好的学习环境执行，不是在海光环境安装其全部依赖。
cd "$LEARNING_ROOT"
python scripts/inspect_token_accumulation.py
python scripts/inspect_gradient_clipping.py
python scripts/inspect_scheduler.py
```

这些示例作用是辅助解释，不代表官方 MiniMind 实机行为完全相同。
它们的可运行性和输出说明由准备方先验收，实习生不需要编写新测试。

### 7.2 理解顺序

1. 从 Dataset 得到 input_ids/labels，确定哪些 token 被监督。
2. 查看模型输出 loss 的归一化口径，再看累积是否又缩放一次。
3. 对照日志区分 micro-step 和 optimizer 更新数，确认尾部窗口。
4. 查看 clipping、optimizer、LR 的调用顺序及日志时点。
5. 把该版本实际语义记入配方，而不是凭概念推测实现。

全窗口 token 加权平均是一种明确的训练目标；上游 micro-batch 平均不是同一个加权方式。
本实习先识别差异，不强制重写。要比较方法时先统一计算口径与数据曝光量。

<a id="r8"></a>
## 8. 保存与恢复

### 8.1 功能恢复操作

1. 使用单卡、worker 0、accumulation 1 的小任务，保持原定总 epoch/数据/seed 不变。
2. 准备方指定一个完整 checkpoint 的确认方法和安全停止方法，避免在文件尚未写完时复制或终止。
3. 保存模型、optimizer、scaler、epoch/step 等实际字段清单和文件 hash。
4. 在原运行工作副本的 `trainer/` 使用同一完整配置与 `--from_resume 1`；
   先核对 `../checkpoints/<save_weight>_<hidden_size>_resume.pth` 存在、可加载且身份正确。
5. 检查续训日志中的起始 epoch/step、LR、数据位置及更新后的参数。
6. 另一个小任务连续训练作比较，记录偏差；不要把计划从 10 步缩为 5 步来模拟中断，
   否则 LR 计划本身已经改变。

原生 from_resume 是在对应工作副本内的自动查找，不是任意恢复路径参数。
缺文件时可能从头开始，因此交付启动检查必须在入口之前拦截。

### 8.2 能力边界

上游恢复保存 FP16 模型副本且没有完整 RNG 身份合同，因此功能恢复成功不能自动证明严格连续等价。
保留训练产物不仅要导出权重，也要真正的 resume 文件。正式长期运行前由导师认可可接受的恢复边界。
需要严格等价时优先选择已验证框架能力，或由准备方补齐并验证，不要求实习生开发。

### 8.3 严格等价的额外检查

若后续研究确实依赖严格恢复，应检查模型原精度、optimizer、LR/scaler、全部相关 RNG、
已提交数据位置、样本顺序、配置及数据身份，并进行连续/恢复对照。
这部分是有条件的加严验收，不是所有 demo 的编码作业。

### 8.4 状态判断

checkpoint 必须是完整保存点。声明恢复失败时不能静默新训；
进程被硬杀时 manifest 可能仍为 running，应结合进程、日志时间和退出文件判断。
工具的错误路径保护由准备方交付，实习生执行副本上的故障演练。

<a id="r9"></a>
## 9. 规模与对照

### 9.1 规模决策

1. 固定 dense 架构和 tokenizer；打印实际总参数量及可训练量，避免只按 hidden size 猜模型规模。
2. 如采用上游 768 hidden、8 层配置，先重新测显存和吞吐；Demo 2 的 64 hidden 小模型不代表其资源消耗。
3. 用正式候选长度做常见长度、长尾长度和最长允许长度探针，覆盖第一次 AdamW step。
4. 确定 max length、micro-batch、累积数、精度、训练 token 预算；若改变，记录新合同。
5. 从 pilot 中测真实有效 token/s 和更新耗时，剔除启动热身，另计保存和验证开销。
6. 估算总时间：计划实际消费 token / 实测同口径吞吐 + 验证 + checkpoint + 数据处理时间。
7. 给预算留出故障排查、评测和备份余量；导师确认后才启动长任务。

token 预算是累计消费量，不是单条样本长度；同一语料重复多轮会增加消费量，但不增加独立数据量。
同时记录原始 token、实际非 padding 输入 token、有效监督 token，使用哪一种估算吞吐必须写清楚。

### 9.2 数据扩展

1. 第一档先扩大 `pretrain_t2t_mini.jsonl` 的冻结 train 子集，不立即换数据来源。
2. 如果目标是完整使用这个文件，通过原始审计后只消费 train 部分；validation/test 仍不得加入。
3. 如果决定使用更大 `pretrain_t2t.jsonl`，重新检查与原文件的重叠、来源、长度和许可，生成新版本。
4. 跨新旧训练集和既有评测集查重；重叠无法消除或追溯时，明确降低结论强度。
5. 多来源混合先统计每个来源的 token 占比，而非只统计样本条数；第一轮保留清楚可解释的配比。
6. 保存最终被消费的 ID 顺序和 token 计数，核对实际数据量与预算。

“全量”必须说明是哪个 revision、哪个文件、经过哪些规则后的 train 全量，不是含糊的“完整 MiniMind 数据”。

### 9.3 单变量问题

第一轮只选一个问题，最多少量配置，不同时改数据、结构、长度和优化器。

推荐问题：**相同训练 token 预算下，使用更多独立数据，是否优于反复训练较小子集？**

1. 从同一冻结 train 构造相互嵌套的大小子集，尽量保持来源分布一致。
2. 两组使用相同初始化、架构、tokenizer、优化器、评测集和总 token 预算。
3. 小子集循环消费，大子集消费更多独立内容；两组均按消费 token 记录训练进度及 LR。
4. 报告实际计算耗时，样本长度不同可能导致相同 token 预算也不等耗时。
5. 比较训练 NLL、验证 NLL、重复率、泛化差距和成本；预算允许时换 seed 复核趋势。

另一种问题是“更多训练 token 是否有帮助”，此时计算量本身就变化，不能称为等算力对照。
不要从两三个教学点外推 scaling law，也不要据此宣布某个数据量适用于所有模型。

### 9.4 团队配方

记录实测 token/s、显存、数据分布和故障条件，注明它们只适用于当前设备/模型/长度。
下一次专用模型训练可以复用测量方法，不能直接继承本次吞吐估计和最佳超参数。

<a id="r10"></a>
## 10. SFT 细则

### 10.1 消息数据

按 Demo 2 的获取流程下载并冻结 `sft_t2t_mini.jsonl` 的 revision。上游常见输入形态为：

```json
{"conversations":[{"role":"user","content":"解释什么是单元测试。"},{"role":"assistant","content":"单元测试用于检查一个较小代码单元的行为是否符合预期。"}]}
```

审计时依次检查：

1. `conversations` 是非空列表，role/content 类型合法，至少存在可监督 assistant 内容。
2. 消息顺序符合本轮范围；第一轮只支持明确的普通对话，工具消息等不支持形态单独隔离并统计。
3. 多轮对话按完整会话分组；相同问题及相近模板尽量分组，不按单条回答随机散入测试集。
4. 重复、空回答、模板标记异常、疑似低质量回答分别记录，抽检清洗规则。
5. 按冻结模板 tokenize，统计完整长度、assistant 有效 token 数和零监督样本。
6. 第一轮优先过滤超过长度上限的完整会话，保留统计；若截断，必须验证答案边界与结束标记，不静默切掉回答尾部。

审计输出沿用 Demo 2 的文件结构，版本命名 `mm-sft-v1`，但使用独立的规则和 manifest。
不能把 PT 的 text 检查直接当作 SFT 校验。

### 10.2 监督范围

逐条打印单轮、多轮、含 system、刚好达到长度边界的样本，人工确认：

```text
messages → chat template → input_ids → assistant 区间 → labels → causal shift → loss
```

提示和 padding 不参与回答 loss，assistant 内容及约定结束标记参与监督。
运行交付检查确认每个监督 token 的来源，而不是只看 `labels` 和 `input_ids` 形状相同。

该上游 Dataset 含随机插入 system、处理空 think 标签等训练增强。先阅读实现并记录是否启用。
冻结验证时关闭随机变换或预先固化其结果；同一条验证样本不能每次产生不同提示。
不要不知不觉把“遵循上游增强”写成“使用完全固定模板”。

### 10.3 阶段交接

1. 输入 checkpoint 指向 Demo 4 明确选定的预训练产物；记录 hash，不使用模糊的 `latest`。
2. tokenizer、词表、hidden size、层数和模型结构与权重一致。
3. 检查加载时的 missing/unexpected keys；原生 `strict=False` 不应成为忽略结构不匹配的理由。
4. 新 SFT 阶段建立新 optimizer 和 LR 计划；这是新实验，不是预训练断点恢复。
5. 使用官方 `train_full_sft.py` 或准备方已验证的成熟框架，先小 batch 过拟合、smoke、pilot，再正式训练。
6. 参照上游 `trainer/train_full_sft.py` 解释差异，采用哪套计算语义写入配置。

若只是检查原生入口，其 `--from_weight pretrain` 默认从当前工作副本的 `../out` 读取，
不是由 `--save_dir` 指定加载位置；预训练权重必须放对位置且结构匹配。交付配置必须写明实际初始化文件位置。

### 10.4 评测分析

1. 训练前的预训练 checkpoint 与 SFT checkpoint 使用相同模板、测试提示和生成预算。
2. 对两者计算同一 SFT 验证集的 assistant-token NLL。
3. 用同一份保留的普通文本集检查预训练分布 NLL 是否回退；不能直接对比 PT 训练 loss 与 SFT 训练 loss。
4. 固定一组开发提示检查指令遵循、复读、结束和截断，最终测试提示不参与调参。
5. 保存每条输出、生成 token 数、停止原因、耗时和配置 hash。
6. 按改善、回退、不变分组看样本，说明 NLL 改善与回答质量改善是否一致。

不要临时给较差的一方更长生成预算来“补救”正式比较。最大生成 token 数与训练 batch 的 padding 长度是两回事。

<a id="r11"></a>
## 11. 缺陷分类细则

### 11.1 任务定义

推荐候选为 [CodeXGLUE Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection)，
来源为 Devign。官方提供函数内容、二分类标签和固定划分，适合低成本建立客观评测。
输入只包含函数代码；输出限定为 `0` 或 `1`，映射以数据集定义为准。

第一轮不要求模型编造根因解释。数据没有提供可靠逐条解释标签时，不生成“标准解释”再当真值训练。
数据标签也不等于完备的安全证明：输出 `0` 仅表示本任务的负类，不保证代码不存在任何缺陷。

将任务定义、标签解释、主指标和成本上限写入 `configs/defect_task.yaml`，由导师确认。
业务目标未来若转向日志根因分析，应另建数据合同，不把这个实验直接改名为 AnaDroid 复现。

### 11.2 来源处理

1. 固定 CodeXGLUE 的代码提交，记录官方 README 的数据下载入口和许可信息。
2. 按官方入口下载 `function.json`，保存到 `raw/devign/<version>/`，核对内容、大小和 SHA-256。
3. 保存同一代码提交下 `dataset/train.txt`、`valid.txt`、`test.txt` 的原始划分索引和 hash。
4. 在 `staging/devign-v1/official/` 放置原始文件的工作副本、三份索引及固定版 `preprocess.py`。
5. 在这个 staging 目录执行 `python preprocess.py`，保留 stdout/stderr，得到三个 JSONL。
6. 验证输出 ID 与原始索引对应，检查 `func` 非空、`target` 为 0/1、`idx` 唯一，确认无重复/遗漏索引。
7. 统计各 split 的样本数、正负比例、可获取的项目分布、函数长度及重复/近重复情况。
8. 保存审计结果到 `staging/devign-v1/audit/`，原始数据不覆盖。

官方切分混合了不同项目，不能据此声称测试了跨项目泛化。若发现跨 split 重复，
保留官方版本作为来源证据，另生成明确命名的清洗版本，并记录移除规则和数量；不能暗改后继续报官方榜单结果。
按固定规则优先从 train 删除与 validation/test 重合的记录；开发集与测试集重合时也隔离处理。
未来做跨项目实验时重新按项目/提交/函数族分组，作为另一份合同，不与随机划分结果混报。

### 11.3 消息转换

生成会话训练记录，原始 ID 和标签放在旁路元数据中，避免把真值拼入用户提示：

```json
{"conversations":[{"role":"system","content":"判断给定函数的缺陷标签。只输出 0 或 1。"},{"role":"user","content":"<待判断的原始函数代码>"},{"role":"assistant","content":"1"}]}
```

上例只是格式展示，不是带真实标签的训练样本。实际脚本从 `func` 和 `target` 生成，并记录转换代码版本。
若所选框架使用 `messages` 而非 `conversations`，在适配层转换并测试，不能随意改字段后假设能读取。

1. 固定 system 提示和输出标签，先在开发集检验模板，不依据 test 调整。
2. 用候选基座 tokenizer 统计模板后的长度；代码不删缩进，不按字符数冒充 token 数。
3. 第一轮超长函数单独隔离，不把一个函数任意切段后给每段继承相同标签。
4. 训练、验证、测试的长度处理事先固定，报告过滤覆盖率和保留后的类别比例。
5. 若只评估可容纳的子集，结论明确限定这个子集；不把剩余难样本悄悄忽略。
6. 抽查各类别、不同长度、不同来源的转换结果，确认 assistant 标签未出现在评测输入。
7. 冻结到 `frozen/devign-sft-v1/`，包括原始 split、筛选规则、旁路 ID、模板和 tokenizer 校验值。

### 11.4 框架与基线

先选一个在海光上能稳定 forward/backward、具备代码基础的小型公开模型。
参数规模、context、许可、chat template、dtype 和内存占用均实测后确定，不提前承诺 7B 或 32B。

1. 固定模型仓库 revision、权重和 tokenizer 校验值，保存候选选择理由。
2. 先跑多数类常量预测，得到不训练也能达到的下限。
3. 跑 B0：原基座 + 冻结提示，先 20 条 smoke，再固定开发子集，最终方案冻结后完成规定测试评测。
4. 在相同合同下验证拟采用的 LoRA 或全参训练能更新参数；LoRA 更省资源，但不是默认一定更好。
5. 第一轮不使用 QLoRA，除非量化库已在此海光环境通过验证。
6. 只在选定框架能可靠工作时使用它；保留版本、训练参数和数据适配测试，不能把框架封装当免检证明。

基线先统一使用 greedy、相同 `max_new_tokens`，例如 8 个新 token 作为起步候选。
解析只接受去掉首尾空白后的完整 `0` 或 `1`；其他输出标为 invalid，不从解释段落里随意抓一个数字。
如果改用标签条件概率打分，B0/S1 都必须使用同样的标签 token 序列评分方式，这是另一份评测合同。

数据转换、分类评测和框架配置由准备方交付；实习生执行并检查。
具体框架需在 BW1000 验证后冻结，不把“可用 Transformers/PEFT 或 LLaMA-Factory”写成已确认兼容。

### 11.5 结果解释

每条评测结果保存：`sample_id`、真值、原始输出、解析标签、是否 invalid、
输入/输出 token 数、停止原因、耗时、模型身份和评测合同 hash。

1. 主指标使用预先约定的 accuracy 与 macro-F1，另报正类 precision/recall、混淆矩阵和 invalid rate。
2. invalid 不能删掉后只评剩余样本。accuracy 按错误计；macro-F1 明确只对真实 0/1 类取平均，invalid 作为对应真实类的漏判。
3. 按 ID 配对 B0/S1，列出正确→正确、错误→正确、正确→错误、错误→错误。
4. 报告绝对百分点变化，不只报告相对百分比；少量变化需结合样本量和不确定性解读。
5. 分类别、长度和可用项目字段分析回退，但不要用事后挑选的最好子组代表整体。
6. 分层固定抽样查看代码和输出，判断是否格式错误、长度不足、标签噪声或实际判断错误。
7. 最终 test 在方案冻结后进行；如果根据 test 发现再改方案，后续结果标为探索，另需未使用评测证明。

首个消融建议比较**全参与 LoRA**或**两个训练子集规模**，只选一个。
前者记录可训练参数、数据曝光量、LR 调参预算和资源成本；使用同一 LR 只是控制变量实验，
不能据此宣布某一种方法在充分调参后更优。

<a id="r12"></a>
## 12. 后续阶段

以下分支均优先使用现成入口与团队工具；是否进入由结果决定，不作为所有人必须跑遍的算法清单。
实习生继续负责执行与记录，准备方提供数据适配、现成框架配置和验收工具。

### 12.1 领域继续预训练

适用线索：模型明显缺少领域语言或代码知识，且有大量可用的无标注领域文本。

1. 先确认无标注数据来源和许可，排除验证/测试代码及相关修复内容泄漏。
2. 审计、去重、按来源划分，统计 token 分布及与原基座数据的差异。
3. 固定同一基座，比较直接 SFT 与 CPT 后 SFT；两组下游 SFT 数据和评测一致。
4. 预先限制 CPT token 预算，记录增加的计算成本，不能把额外训练当作免费算法收益。
5. 检查目标能力和通用能力回退，再决定扩大。

这不是重新从随机初始化训练。没有数据或没有证据表明知识不足时，不做。

### 12.2 偏好优化

适用线索：已有能够回答的 SFT 模型，错误体现为回答质量偏好，而且有可信 chosen/rejected 对。

1. 固定同一 prompt，为两个候选确定可靠偏好和理由。
2. 不将重复、完全相同、标签冲突或只是长短差异的回答直接当有效偏好。
3. 按 prompt/来源分组划分，检查长度截断后偏好差异仍存在。
4. 固定 policy 初值、reference、beta、数据和预算，先做数值测试及小型 smoke。
5. 比较 SFT 与 SFT+DPO 的相同任务结果，同时记录回答长度和有效率，防止只学会长度偏好。

只有 0/1 分类标签时，通常先做监督学习即可，不为了走流程硬造偏好数据。

### 12.3 可验证奖励

适用线索：后续任务是代码生成、修复或测试生成，有可执行、独立且难投机的验证器。

1. 先在隔离容器验证数据和测试，禁止网络、限制时间/CPU/内存，不直接在工作主机运行生成代码。
2. 区分公开反馈测试与隐藏验收测试，防止模型只迎合可见用例。
3. 测试验证器的误判、超时、崩溃和奖励漏洞，记录奖励覆盖了什么、没有覆盖什么。
4. 先跑 SFT 模型的采样基线，再决定是否承担在线采样和 RL 的预算。
5. 冻结 reward、采样数、长度和训练预算；保留所有轨迹、奖励分项及最终独立评测。

Agent RL 还需要多步环境、工具权限和轨迹评测。当前分类 demo 不需要这些，不把它们设为结题必选项。

### 12.4 论文边界

[AnaDroid](https://pkun.life/papers/issta26/) 可作为未来失败测试分析的参考：从测试资料中提取证据，
再做领域适配和偏好优化。它针对的是移动测试失败，不能用代码二分类分数替代其三类任务结论。
[作者仓库](https://github.com/pppppkun/AnaDroid)说明公开了部分脱敏案例，训练权重未公开；
因此不把完整工业训练集视为当前可用资源，也不要求实习生复现同等数据规模的指标。

今后拿到内部数据，先做来源、标签、可用时点和按故障事件分组的审计。
事后修复结论可以是训练目标，却不能悄悄放进模型推断时拿不到的输入。

## 13. 来源与既有材料

- [MiniMind 固定源码](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f)：本手册的原生 CLI 和路径行为以此为准。
- [预训练入口](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/train_pretrain.py)、[训练工具](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/trainer_utils.py)、[数据处理](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/dataset/lm_dataset.py)。
- [MiniMind 数据](https://huggingface.co/datasets/jingyaogong/minimind_dataset)：使用前另行固定数据 revision。
- [海光 PyTorch](https://github.com/HYGON-AI/pytorch-das)：软件配套和安装入口，实际环境以管理员确认版本为准。
- [CodeXGLUE 任务说明](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection)、[官方预处理](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/Defect-detection/dataset/preprocess.py)：用于 Demo 6 的来源与字段核验；实施时固定提交。
- [AnaDroid 论文](https://pkun.life/files/issta26.pdf)、[项目](https://github.com/pppppkun/AnaDroid)：未来测试失败分析的方法参考，不是本课题已复现结果。
- [既有 MiniMind 实践入口](../minimind/README.md)：可参考已有分模块讲解，但个人学习环境不直接作为海光环境锁文件。
- [数据审计教程](../minimind/practice_notes/week_01_data_audit.md)、[tokenizer 与 Dataset](../minimind/practice_notes/week_02_tokenizer_dataset.md)、[训练系统](../minimind/practice_notes/week_05_training_system.md)、[验证与 checkpoint](../minimind/practice_notes/week_06_validation_and_checkpoint.md)。
