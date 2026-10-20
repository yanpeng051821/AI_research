# 第 6 周：验证与训练恢复实践教程

> 教程重写与回归：2026-10-19，全项目 249 项测试通过。本文带你在第五周训练循环上增加验证、保存、恢复和命令行入口；所有学习 QA 集中在最后一节。

## 1. 本周任务与准备

### 1.1 从参数更新到可恢复训练

第五周的 train_update 已能完成一次参数更新，train_epoch 能处理整轮数据和不足一个累积窗口的尾部。现在还缺少两个能力：训练之外怎样检查模型；进程退出后怎样继续原来的训练。

本周先把验证损失算对，再逐步保存模型、优化器、调度器、随机数和样本位置。最后用两条训练路径对照，证明恢复后不是悄悄开始了一次新实验。

~~~text
已有：batch -> forward -> loss -> backward -> optimizer / scheduler
新增：                        |
                              +-> 验证 NLL（不更新）
                              +-> 更新成功后提交样本位置
                              +-> 保存完整更新边界的状态
                                       |
                                 重建对象并加载
                                       |
                                 继续剩余样本
~~~

完成标准有三项：验证不改变训练参数；中断恢复与连续训练在本实验条件下得到一致结果；命令行能保存成功结果，普通异常能留下错误并失败退出。本周不要求训练出有语言能力的模型。

### 1.2 环境与已有接口

以下命令均在案例的 implementation 目录运行，命令块使用 PowerShell。先确认环境：

~~~powershell
uv sync --locked
uv run --locked python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
uv run --locked pytest tests/test_training.py -q
~~~

现有仓库已包含本周答案与测试；首次实现时在自己的练习副本中操作，不要删除现有模块。复习时可以直接运行现有脚本，再选择某一步独立重写。测试数随后续工程增加，重点是相关断言通过，不必复现历史总数。

本周复用以下接口，无需重写模型：

| 已有接口 | 输入与返回 |
| --- | --- |
| MinimindForCausalLM(config) | 输入 token 张量和 attention mask，返回 logits |
| causal_lm_loss_sum_and_count(logits, labels) | 内部执行 causal shift，返回损失和、有效目标 token 数 |
| train_update(model, optimizer, micro_batches, scheduler) | 完成一个累积窗口，返回 loss、token 数、梯度范数和学习率 |
| train_epoch(model, optimizer, dataloader, scheduler) | 完成一轮剩余输入，返回逐更新记录列表 |

参考入口：[模型](../implementation/src/minimind_lab/models/model.py)、[loss](../implementation/src/minimind_lab/models/loss.py)、[单次更新](../implementation/src/minimind_lab/training/step.py)、[整轮循环](../implementation/src/minimind_lab/training/loop.py)。

### 1.3 实践顺序

| 步骤 | 本步交付物 | 进入下一步的条件 |
| --- | --- | --- |
| 2. 验证 NLL | evaluate.py | 拆批与整批验证结果一致 |
| 3. 训练状态观察 | inspect_checkpoint_state.py | 能找到参数、m/v、学习率和更新进度 |
| 4. 基础保存加载 | checkpoint.py | 重建对象后状态能加载，步数正确 |
| 5. 随机状态 | RNG 观察与 checkpoint 扩展 | 恢复后的下一组随机数一致 |
| 6. 样本位置 | sampler.py | 恢复到已提交位置 |
| 7. 提交接线 | loop.py 与 checkpoint 扩展 | 完整窗口和尾部都提交正确样本数 |
| 8. 联合恢复 | inspect_training_resume.py | 模型、优化器、调度器和数据进度一致 |
| 9. 运行入口 | train_pretrain.py | 成功结果、错误记录和输出保护可检查 |
| 10. 本周验收 | 回归与产物检查 | 已知范围内闭环，限制明确 |

先按顺序完成正文。遇到 API 疑问可查阅末尾 QA，再回到当前步骤，不需要提前读完全部问答。

## 2. 计算验证集 NLL

### 2.1 计算目标

训练损失来自正在更新的模型，不同更新之间参数不同。验证时固定模型，用完整验证集衡量它对正确文本的预测情况。这里仍是给定正确前缀预测下一个 token，不是自由生成。

创建 [training/evaluate.py](../implementation/src/minimind_lab/training/evaluate.py)，实现 evaluate_nll(model, dataloader)。每个 batch 沿用 input_ids、attention_mask、labels 三个字段，返回 nll 和 valid_tokens。

~~~text
验证 NLL = 所有 batch 的 loss_sum 之和 / 所有 batch 的有效 token 数之和
~~~

例如两批损失和分别为 4、18，token 数为 2、6，答案是 22/8=2.75。不能先算两批均值，再等权平均得到 2.5；那会让短批次的每个 token 权重更高。

### 2.2 实现顺序

先保存调用前的模式，再在 eval 和 no_grad 下前向。把恢复模式放进 finally，保证读取数据或计算中途出错时也执行。

~~~python
import torch
from minimind_lab.models.loss import causal_lm_loss_sum_and_count


def evaluate_nll(model, dataloader):
    was_training = model.training
    total_loss_sum = 0.0
    total_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss_sum, token_count = causal_lm_loss_sum_and_count(
                    logits, batch["labels"]
                )
                total_loss_sum += loss_sum.item()
                total_tokens += token_count.item()

        if total_tokens == 0:
            raise ValueError("total_tokens must be greater than 0")
        return {
            "nll": total_loss_sum / total_tokens,
            "valid_tokens": total_tokens,
        }
    finally:
        model.train(was_training)
~~~

不要加 backward、optimizer.step 或 zero_grad。验证既不更新参数，也不清理训练中已有梯度。本例要求输入与模型在同一设备；复用的 loss 函数要求每批至少有一个 shift 后有效目标。

### 2.3 运行与检查

在 [test_training.py](../implementation/tests/test_training.py) 的 test_evaluate_nll_matches_full_batch 中，准备两条不同有效长度的样本。先拼成一个大 batch 计算参考 loss，再拆成两个 batch 调用 evaluate_nll。

~~~powershell
uv run --locked pytest tests/test_training.py -q -k evaluate
~~~

检查 NLL 接近、有效 token 数一致，且调用前为 train 时调用后仍为 train。两个样本有效 token 数刻意不同，才能发现错误的 batch 等权平均；只有 shape 一致不能验证这个问题。

结果不符时，先检查是否重复 shift、padding 是否参与 loss、是否把 batch 均值直接相加。模式不符时，检查 finally 是否覆盖真正的前向计算。模式与梯度的区别见 QA 11.1。

完成后，你已有一条不更新参数的评估路径。下一步处理“哪些状态必须一起保存”。

## 3. 观察训练对象的状态

### 3.1 准备可观察的小模型

先不用完整 Transformer，在 [scripts/inspect_checkpoint_state.py](../implementation/scripts/inspect_checkpoint_state.py) 中创建一个线性层：

~~~python
import torch

model = torch.nn.Linear(2, 1, bias=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=0.5
)
print("model:", model.state_dict())
print("optimizer before:", optimizer.state_dict())

loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
loss.backward()
optimizer.step()
scheduler.step()

print("optimizer after:", optimizer.state_dict())
print("scheduler:", scheduler.state_dict())
~~~

~~~powershell
uv run --locked python scripts/inspect_checkpoint_state.py
~~~

不要求随机初始化权重与别人相同，重点观察结构：更新前 AdamW 的 state 为空；第一次更新后出现 step、exp_avg 和 exp_avg_sq；学习率变为 0.0005，scheduler.last_epoch 为 1。

### 3.2 确定保存范围

| 对象 | 保存的作用 |
| --- | --- |
| model | 当前参数与持久 buffer，是继续计算的起点 |
| optimizer | AdamW 的历史 m/v 和每个参数的 step |
| scheduler | 决定后续学习率的配置与进度 |
| global_step | 外层记录已完成多少次更新 |
| RNG | 随机计算从中断位置继续 |
| sampler | 下一次读剩余数据，而不是从第一条重来 |

本步先看清前三个对象，RNG 和 sampler 在后面分别接入。model.state_dict 不保存 Python 类定义，也不保存计算图和参数梯度，恢复时必须先创建相同结构。

若 scheduler 内部的 _step_count 为 2，不要认为已经更新模型两次，其中可能包含初始化调用。训练进度由自己的 global_step 明确记录。param_groups 与 state 的关系见 QA 11.2。

## 4. 保存并加载基础 checkpoint

### 4.1 确定保存时机

在 [training/checkpoint.py](../implementation/src/minimind_lab/training/checkpoint.py) 实现两个函数。先处理模型、优化器、调度器和步数，后续逐步增加 RNG 和 sampler。

~~~python
def save_checkpoint(path, model, optimizer, scheduler, global_step):
    ...

def load_checkpoint(path, model, optimizer, scheduler):
    ...
~~~

只保存完整更新边界：optimizer 和 scheduler 已更新，之后还会要求 sampler 已提交。不要在梯度累积到一半时调用保存函数；当前没有保存未完成梯度的逻辑。

### 4.2 临时文件与替换

先在文件顶部导入 Path 和 torch，然后实现：

~~~python
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scheduler, global_step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path
~~~

若直接覆写原文件，写到一半失败会使原 checkpoint 也损坏。同目录临时文件写完整后再替换，失败时允许留下 .tmp，但原正式文件仍保留。这是单写者约定，不是断电持久性保证。

### 4.3 加载到新对象

load_checkpoint 的基础版本：

~~~python
def load_checkpoint(path, model, optimizer, scheduler):
    state = torch.load(
        Path(path), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["global_step"]
~~~

调用前先创建模型，再用新模型参数创建 optimizer，再创建 scheduler。加载不会替你创建完整架构；参数组的数量和顺序也必须兼容。

weights_only=True 是受限反序列化模式，不是“只取 model 字段”。模型、优化器、调度器和普通整数都能一起读取。

### 4.4 最小往返检查

在第 3 节脚本的更新之后，临时加入下面的检查；通过后可保留为独立练习，不必修改正式观察脚本。

~~~python
from pathlib import Path
from tempfile import TemporaryDirectory
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint

with TemporaryDirectory() as directory:
    path = Path(directory) / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, global_step=1)
    restored = torch.nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    step = load_checkpoint(
        path, restored, restored_optimizer, restored_scheduler
    )
    assert step == 1
    torch.testing.assert_close(restored.weight, model.weight)
    assert restored_optimizer.param_groups[0]["lr"] == 0.0005
    assert restored_scheduler.state_dict() == scheduler.state_dict()
~~~

检查的是新对象，不是原对象。TemporaryDirectory 离开 with 后清理文件，所以加载必须在 with 内完成。

权重相同而学习率不对时，检查是否忘记恢复 optimizer，或是否在加载后又创建 scheduler。基础往返通过后继续保存随机状态，此时不能宣布完整训练可复现。

## 5. 恢复随机数生成进度

### 5.1 Seed 与状态观察

创建 [scripts/inspect_rng_state.py](../implementation/scripts/inspect_rng_state.py)，分清“从哪里开始”和“现在走到哪里”：

~~~python
import torch

torch.manual_seed(42)
first = torch.rand(3)
state = torch.get_rng_state()
expected_next = torch.rand(3)
torch.rand(10)
torch.set_rng_state(state)
restored_next = torch.rand(3)

print("first:", first)
print("expected next:", expected_next)
print("restored next:", restored_next)
torch.testing.assert_close(restored_next, expected_next, rtol=0, atol=0)
~~~

~~~powershell
uv run --locked python scripts/inspect_rng_state.py
~~~

后两行必须相同。恢复的是生成 first 之后的位置；若用 manual_seed(42) 替代 set_rng_state，下一组会回到 first，不能继续原随机序列。

### 5.2 扩展 checkpoint

在 save_checkpoint 字典里增加：

~~~python
"torch_rng_state": torch.get_rng_state(),
"cuda_rng_state_all": (
    torch.cuda.get_rng_state_all()
    if torch.cuda.is_available()
    else None
),
~~~

在 load_checkpoint 读出字典后、修改训练对象之前检查 CUDA：

~~~python
cuda_states = state["cuda_rng_state_all"]
if cuda_states is not None:
    if not torch.cuda.is_available():
        raise ValueError("checkpoint contains CUDA state but CUDA is unavailable")
    if len(cuda_states) > torch.cuda.device_count():
        raise ValueError("CUDA device count does not match checkpoint")
~~~

在模型、optimizer、scheduler 加载之后、return 之前恢复 RNG：

~~~python
torch.set_rng_state(state["torch_rng_state"])
if cuda_states is not None:
    torch.cuda.set_rng_state_all(cuda_states)
~~~

创建新模型通常消耗随机数，先完成这些操作，再恢复保存的随机状态。GPU 按可见设备编号对应；允许额外设备不等于允许随意改变分布式拓扑。

### 5.3 核验随机状态

有 CUDA 时，运行已有的 [inspect_checkpoint_roundtrip.py](../implementation/scripts/inspect_checkpoint_roundtrip.py)：

~~~powershell
uv run --locked python scripts/inspect_checkpoint_roundtrip.py
~~~

它真实更新 CUDA 线性层后保存，记录预期下一组 CPU/CUDA 数字，额外消耗随机数并重建对象，再加载和比较。无 CUDA 的机器保留 CPU 小实验即可，不必运行这个 CUDA 脚本。

RTX 4060 Ti 上历史验证通过，CPU 下一组约为 [0.3829,0.9593,0.3904,0.6009,0.2566]，CUDA 约为 [0.6130,0.0101,0.3984,0.0403,0.1563]。依据是断言，不是打印舍入后的数字。

随机数一致仍不能代替数据恢复。从第一个样本重新读不是续训，下一步开始管理 sampler。


## 6. 可恢复的样本顺序

### 6.1 数据位置

前面的恢复解决了“用什么参数继续计算”，这里解决“从哪条数据继续”。Dataset 按索引返回内容，Sampler 提供索引，DataLoader 组织批次。它们各有职责，保存模型不会自动保存数据位置。

本周使用固定的一轮随机顺序，并区分两种位置：

| 字段 | 含义 | 何时推进 |
| --- | --- | --- |
| position | 已经发出的索引数 | 每次迭代取出索引 |
| committed_position | 已经完成参数更新的样本数 | train_update 成功返回后 |

例如，已经读取四条样本，但只完成前两条对应的更新。恢复时应从第三条重新读，不能跳过尚未完成更新的两条。两个位置的单位都是**样本数**，不是 batch 数或 optimizer step 数。

### 6.2 顺序生成与迭代

新建 [data/sampler.py](../implementation/src/minimind_lab/data/sampler.py)，先实现初始化和迭代。下面所有方法都属于同一个 StatefulSampler 类。

~~~python
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class StatefulSampler(Sampler[int]):
    def __init__(self, size: int, seed: int = 42):
        super().__init__()
        if size < 0:
            raise ValueError("size must be non-negative")
        self.generator = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(size, generator=self.generator).tolist()
        self.position = 0
        self.committed_position = 0

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self.indices):
            index = self.indices[self.position]
            self.position += 1
            yield index
~~~

独立 Generator 让样本打乱不消耗模型初始化所用的默认随机数流。randperm 生成完整排列，tolist 把张量转换为 Python 整数列表。迭代时只读取已经生成的列表，不重新打乱。

先在 Python 中检查：

~~~python
from minimind_lab.data.sampler import StatefulSampler

sampler = StatefulSampler(5, seed=42)
print(sampler.indices)
print(list(sampler))
print(sampler.position, sampler.committed_position)
~~~

本项目环境中顺序为 [2, 4, 3, 0, 1]，读完后两个位置分别为 5、0。第二次遍历会得到空列表，这是有状态迭代的预期行为。需要重做实验时重新创建 sampler，不要误用已经耗尽的对象。

### 6.3 消费确认

在类中增加 commit：

~~~python
    def commit(self, sample_count: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        next_position = self.committed_position + sample_count
        if next_position > self.position:
            raise ValueError("cannot commit unread samples")
        self.committed_position = next_position
~~~

它只确认已经读出的样本，不发出新索引。当前接口约定调用方传整数，不能把这里的边界检查理解为对任意 Python 输入都做了完整类型校验。

### 6.4 状态保存与恢复

继续在同一个类中增加：

~~~python
    def state_dict(self) -> dict:
        return {
            "indices": self.indices.copy(),
            "committed_position": self.committed_position,
        }

    def load_state_dict(self, state: dict) -> None:
        indices = state["indices"]
        committed = state["committed_position"]
        size = len(self.indices)
        if len(indices) != size:
            raise ValueError("sampler size mismatch")
        if not all(type(index) is int for index in indices):
            raise ValueError("indices must be integers")
        if sorted(indices) != list(range(size)):
            raise ValueError("indices must be a complete permutation")
        if type(committed) is not int or not 0 <= committed <= size:
            raise ValueError("invalid committed position")

        self.indices = indices.copy()
        self.position = committed
        self.committed_position = committed
~~~

先验证，再修改对象，避免坏状态只加载了一半。恢复后 position 回到 committed_position，原来已经读取但未提交的部分重新读取。这是恢复规则，不是遗失了读取进度。

用下面的实验验证这个区别：

~~~python
sampler = StatefulSampler(5, seed=42)
iterator = iter(sampler)
print([next(iterator) for _ in range(4)])
sampler.commit(2)
state = sampler.state_dict()

restored = StatefulSampler(5, seed=99)
restored.load_state_dict(state)
assert restored.position == restored.committed_position == 2
assert list(restored) == [3, 0, 1]
~~~

新对象用了不同 seed，但恢复的是保存的完整排列，所以仍继续原来的顺序。随后运行：

~~~powershell
uv run --locked pytest tests/test_sampler.py -q
~~~

这些测试还检查越界提交、重复或缺失索引、非整数索引，以及返回列表被修改时不能污染 sampler。它们分别保护不同的状态约束，不是重复证明随机顺序能生成。

### 6.5 DataLoader 接线

在 [scripts/inspect_sampler_loader.py](../implementation/scripts/inspect_sampler_loader.py) 中运行下面的完整小实验：

~~~python
import torch
from torch.utils.data import DataLoader
from minimind_lab.data.sampler import StatefulSampler

dataset = torch.arange(5)
sampler = StatefulSampler(len(dataset), seed=42)
loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler,
    num_workers=0,
    drop_last=False,
    generator=torch.Generator().manual_seed(123),
)
for batch in loader:
    print(batch.tolist(), sampler.position, sampler.committed_position)
~~~

~~~powershell
uv run --locked python scripts/inspect_sampler_loader.py
~~~

预期三行：

~~~text
[2, 4] 2 0
[3, 0] 4 0
[1] 5 0
~~~

这里故意只读不训练，所以 committed_position 始终为零。DataLoader 不知道参数是否更新成功，不能替我们确认消费。下一步把确认放回训练循环。

## 7. 训练循环与消费确认

### 7.1 更新边界

修改 [training/loop.py](../implementation/src/minimind_lab/training/loop.py)：给 train_epoch 增加可选参数 sampler=None。它仍然负责把 micro-batch 放入 pending_batches，凑满一个累积窗口后交给 train_update。

在 train_update 成功返回后、记录 history 和清空 pending_batches 之前，插入：

~~~python
if sampler is not None:
    sample_count = sum(
        batch["input_ids"].shape[0] for batch in pending_batches
    )
    sampler.commit(sample_count)
~~~

**完整窗口和循环结束后的尾部窗口都要插入。** 两处都必须使用各自尚未清空的 pending_batches。如果本地变量名不同，按现有名字替换；不要另外创建一个空列表来计数。

五条样本、batch_size=2、累积两批，实际过程是：

~~~text
读取 2 条 + 读取 2 条 → 一次更新成功 → commit(4)
读取最后 1 条        → 尾部更新成功 → commit(1)
~~~

所以本轮有三个 micro-batch、两次参数更新，最终提交五条样本。尾部不能因为不足一个完整窗口而直接丢弃。

### 7.2 Checkpoint 接线

给 save_checkpoint 和 load_checkpoint 都增加可选参数 sampler=None。在保存字典中加入：

~~~python
"sampler_state": sampler.state_dict() if sampler is not None else None,
~~~

在加载函数的 CUDA 条件检查之后、加载模型状态之前，加入：

~~~python
if sampler is not None:
    sampler_state = state.get("sampler_state")
    if sampler_state is None:
        raise ValueError("cannot restore sampler: checkpoint has no sampler state")
    sampler.load_state_dict(sampler_state)
~~~

这里的 state 就是第 4 节 torch.load 返回的字典；仓库参考实现把同一变量命名为 checkpoint，照教程实现时保持自己的命名一致。保留前面实现的加载顺序：检查恢复条件，加载 sampler、模型、优化器和调度器，最后恢复随机数状态并返回 global_step。到这里，完整实现可以与 [checkpoint.py](../implementation/src/minimind_lab/training/checkpoint.py) 对照。

### 7.3 失败边界

~~~powershell
uv run --locked pytest tests/test_checkpoint.py -q
~~~

重点读其中尾部更新失败的测试：前四条成功后，模拟最后一条对应的 train_update 抛错。预期 position=5、committed_position=4，表示这条数据不能被当作已训练。

这项保证有边界：commit 放在成功返回后，**不代表一次更新内部失败会自动回滚模型**。如果 optimizer 已部分修改参数后异常，应重新加载最近的完整 checkpoint，不能拿出错进程中的当前内存状态直接继续。

现在各个状态都有了保存位置。下一步不再分别检查文件里有哪些键，而是检查恢复后实际训练是否等价。

## 8. 连续训练与中断恢复对照

### 8.1 实验设计

新建 [scripts/inspect_training_resume.py](../implementation/scripts/inspect_training_resume.py)。只用五条短序列，把模型、数据和运行时间都压小，方便定位恢复错误。

| 分支 | 执行内容 | 作用 |
| --- | --- | --- |
| A | 从头完成两次更新 | 连续训练参考 |
| B | 同一起点，只完成第一次更新并保存 | 制造中断点 |
| C | 重建全部对象，加载 B，再完成尾部更新 | 验证恢复结果 |

B 的第一次 loss 应与 A 的第一次一致，不能拿它与 A 两次更新后的最终状态比较。C 才是最终与 A 比较的对象。

### 8.2 数据与模型

在脚本顶部写入以下导入和辅助函数。五条序列都长三，causal shift 后每条有两个有效预测 token，因此两次更新的有效 token 数分别是八和二。

~~~python
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint
from minimind_lab.training.loop import train_epoch
from minimind_lab.training.step import train_update


def make_loader():
    rows = torch.tensor([[1, token, 2] for token in range(3, 8)])
    dataset = [
        {
            "input_ids": row.clone(),
            "attention_mask": torch.ones_like(row),
            "labels": row.clone(),
        }
        for row in rows
    ]
    sampler = StatefulSampler(len(dataset), seed=42)
    loader = DataLoader(
        dataset, batch_size=2, sampler=sampler,
        num_workers=0, drop_last=False,
        generator=torch.Generator().manual_seed(123),
    )
    return loader, sampler


def make_training_objects():
    model = MinimindForCausalLM(
        ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler
~~~

这里使用独立的 DataLoader Generator，避免创建迭代器消耗默认 CPU 随机数而干扰对照。当前模型没有 Dropout，但随机数恢复仍应单独验证，不能仅凭这个模型的最终权重一致就宣称所有随机状态都正确。

### 8.3 连续训练与中断点

下面代码接在两个辅助函数后面，作为脚本顶层代码执行：

~~~python
torch.manual_seed(42)
model_a, optimizer_a, scheduler_a = make_training_objects()
loader_a, sampler_a = make_loader()
history_a = train_epoch(
    model_a, optimizer_a, loader_a, scheduler_a,
    gradient_accumulation_steps=2, sampler=sampler_a,
)

torch.manual_seed(42)
model_b, optimizer_b, scheduler_b = make_training_objects()
loader_b, sampler_b = make_loader()
iterator_b = iter(loader_b)
pending = [next(iterator_b), next(iterator_b)]
first_result_b = train_update(model_b, optimizer_b, pending, scheduler_b)
sampler_b.commit(sum(batch["input_ids"].shape[0] for batch in pending))

print("A:", history_a)
print("B first update:", first_result_b)
assert sampler_b.committed_position == 4
~~~

A、B 都在创建模型前设置 seed=42，保证相同初始参数。先创建模型后再设 seed，不能追溯改变已经生成的权重。

### 8.4 重建与续训

继续追加：

~~~python
with TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "checkpoint.pt"
    save_checkpoint(
        path, model_b, optimizer_b, scheduler_b,
        global_step=1, sampler=sampler_b,
    )

    torch.manual_seed(999)
    model_c, optimizer_c, scheduler_c = make_training_objects()
    loader_c, sampler_c = make_loader()
    restored_step = load_checkpoint(
        path, model_c, optimizer_c, scheduler_c, sampler=sampler_c,
    )
    assert restored_step == 1
    assert sampler_c.position == sampler_c.committed_position == 4
    assert optimizer_c.param_groups[0]["lr"] == 0.0005

    history_c = train_epoch(
        model_c, optimizer_c, loader_c, scheduler_c,
        gradient_accumulation_steps=2, sampler=sampler_c,
    )
    final_step = restored_step + len(history_c)
    print("C:", history_c)
    print("final global step:", final_step)
~~~

C 刻意使用不同初始化种子，恢复后仍应得到正确结果，避免把“恰好初始化一样”误认为加载成功。必须在首次遍历 loader_c 之前加载 sampler 状态。

### 8.5 最终状态比较

以下代码接在 with 语句后，回到顶层缩进。临时文件已删除，但已经加载到内存的对象仍然存在。

~~~python
parameters_a = dict(model_a.named_parameters())
parameters_c = dict(model_c.named_parameters())
assert parameters_a.keys() == parameters_c.keys()

for name, parameter_a in parameters_a.items():
    parameter_c = parameters_c[name]
    torch.testing.assert_close(parameter_a, parameter_c, rtol=1e-5, atol=1e-7)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            optimizer_a.state[parameter_a][key],
            optimizer_c.state[parameter_c][key],
            rtol=1e-5, atol=1e-7,
        )

assert optimizer_a.state_dict()["param_groups"] == optimizer_c.state_dict()["param_groups"]
assert scheduler_a.state_dict() == scheduler_c.state_dict()
assert sampler_a.state_dict() == sampler_c.state_dict()
assert final_step == len(history_a) == 2
print("continuous and resumed training match")
~~~

~~~powershell
uv run --locked python scripts/inspect_training_resume.py
~~~

预期：A 两条更新记录，B 一条，C 一条；最终 step=2、committed_position=5。历史实测两次 loss 约为 2.35122 和 2.92442，使用的学习率分别为 0.001 和 0.0005。具体浮点打印值不是验收合同，断言才是。

如果参数不一致，先查初始 seed、数据顺序和恢复位置，再查优化器状态、调度器状态和尾部窗口的归一化。若参数一致但 m/v 不一致，也不算完整恢复成功：下一次更新仍可能分叉。

本实验验证的是当前小模型、CPU、单进程、单轮、更新边界保存条件下的一致性，不推广为任意硬件或分布式训练都能逐位复现。


## 9. 可执行训练入口

### 9.1 入口职责

前面的实验直接构造 token，用来隔离恢复问题。现在把第二周的数据管线接回来，用 JSONL 运行完整流程：

~~~text
命令行参数
→ 检查输出目录
→ tokenizer + Dataset + Collator + DataLoader
→ 模型 + optimizer + scheduler
→ 可选 checkpoint 恢复
→ train_epoch
→ 保存 checkpoint
→ validation NLL
→ result.json；异常时 error.txt
~~~

新建 [scripts/train_pretrain.py](../implementation/scripts/train_pretrain.py)。本节代码按顺序拼成一个完整文件，不需要从其他脚本隐式继承变量。当前仍使用 CPU 小模型，不启动正式大规模训练。

### 9.2 参数与输入

先写导入和参数解析：

~~~python
import argparse
import torch
import json
import traceback

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.loop import train_epoch
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint
~~~

~~~python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small single-epoch pretraining experiment."
    )

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)

    return parser.parse_args()
~~~

四个参数分别指定训练数据、验证数据、新输出目录和可选恢复文件。输入路径不在脚本中写死，便于以后替换数据；模型规模等实验配置暂时固定，下一阶段再讨论实验配置的冻结。

接入本地 tokenizer 和既有数据模块：

~~~python
def build_data(train_path, validation_path):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(
        root / "assets/minimind_tokenizer",
        local_files_only=True,
    )
    max_length = 64
    collator = PretrainCollator(tokenizer, max_length=max_length)

    train_dataset = PretrainDataset(train_path, tokenizer, max_length)
    validation_dataset = PretrainDataset(
        validation_path, tokenizer, max_length
    )
    sampler = StatefulSampler(size=len(train_dataset), seed=42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(123),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(456),
    )
    return tokenizer, train_loader, validation_loader, sampler
~~~

这一步不重新实现分词或 padding。PretrainDataset 负责从记录构造序列，PretrainCollator 负责组 batch；训练 sampler 控制顺序，验证集不随机打乱。两个 loader 都使用独立 Generator。

max_length=64 是本次小实验的序列限制，batch_size=2 是每批样本数，两者不是同一概念。这里设置 num_workers=0，先排除多进程预取带来的额外状态。

### 9.3 对象构建与运行

词表大小从 tokenizer 读取，不能继续使用上节手造数据的 vocab_size=8：

~~~python
def build_training_objects(vocab_size):
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
    )
    model = MinimindForCausalLM(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.02,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )
    return model, optimizer, scheduler
~~~

接下来把调用顺序放进 run_training：

~~~python
def run_training(args):
    torch.manual_seed(42)

    tokenizer, train_loader, validation_loader, sampler = build_data(
        args.train_data,
        args.validation_data,
    )
    model, optimizer, scheduler = build_training_objects(len(tokenizer))

    print("train samples:", len(train_loader.dataset))
    print("validation samples:", len(validation_loader.dataset))
    print("vocab size:", len(tokenizer))
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    print("sampler position:", sampler.position)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            "output checkpoint already exists; use a new output directory"
        )

    global_step = 0
    if args.resume is not None:
        global_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            sampler=sampler,
        )

    history = train_epoch(
        model,
        optimizer,
        train_loader,
        scheduler,
        gradient_accumulation_steps=2,
        sampler=sampler,
    )
    global_step += len(history)

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        global_step=global_step,
        sampler=sampler,
    )

    validation = evaluate_nll(model, validation_loader)

    print("global step:", global_step)
    print("committed samples:", sampler.committed_position)
    print("validation:", validation)
    print("checkpoint:", checkpoint_path)

    return {
        "global_step": global_step,
        "committed_samples": sampler.committed_position,
        "updates_this_run": len(history),
        "history": history,
        "validation": validation,
        "checkpoint": str(checkpoint_path),
    }
~~~

阅读这段代码时关注三个边界：

1. 恢复发生在训练循环之前。global_step 来自 checkpoint，不从零重新计数。
2. global_step 加的是本次完成的更新次数，不是 micro-batch 数或样本数。
3. 先保存 checkpoint，再验证。验证失败时，训练状态可能已经成功保存，但整次运行还不能算全部完成。

### 9.4 结果与异常

最后添加 main 和入口保护：

~~~python
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if any(args.output_dir.iterdir()):
        raise FileExistsError("use a new or empty output directory")

    try:
        result = run_training(args)
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        error_path = args.output_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

if __name__ == "__main__":
    main()
~~~

输出目录必须为空，防止一次新实验覆盖旧结果。异常重新抛出，调用方才能通过非零退出码知道失败；只有写入 error.txt 然后正常返回，会把失败伪装成成功。

注意 try 的范围：参数解析、建目录和“目录非空”检查在 try 外，因此这些启动失败不保证产生 error.txt。文件系统不可写或进程被强杀，也不能指望 Python 一定来得及写错误记录。

### 9.5 首次运行

仓库已提供两个小输入文件：

- [训练样例](../implementation/tests/fixtures/pretrain_smoke/train.jsonl)：五条记录。
- [验证样例](../implementation/tests/fixtures/pretrain_smoke/validation.jsonl)：两条记录。

在 implementation 目录运行下面的 PowerShell 命令。输出目录若已存在且非空，换一个新名字，不要删除旧实验来强行复用。

~~~powershell
uv run --locked python scripts/train_pretrain.py --train-data tests/fixtures/pretrain_smoke/train.jsonl --validation-data tests/fixtures/pretrain_smoke/validation.jsonl --output-dir runs/week06-tutorial-first
Get-Content runs/week06-tutorial-first/result.json
~~~

预期产物是 checkpoint.pt 和 result.json。报告中的关键结果：

| 字段 | 预期 | 判断理由 |
| --- | --- | --- |
| global_step | 2 | 两个更新窗口 |
| updates_this_run | 2 | 本次从头完成两次更新 |
| committed_samples | 5 | 全部训练样本已用于成功更新 |
| validation.valid_tokens | 23 | 两条验证记录的有效预测 token 总数 |
| validation.nll | 历史约 8.64430 | 当前随机小模型的验证结果，不是能力达标线 |

如果本地 tokenizer 缺失，应先恢复第二周使用的 assets/minimind_tokenizer，再执行。不要临时换一个 tokenizer 来绕过报错，否则 token 和词表都变了，历史结果不再可比。

### 9.6 恢复与失败检查

使用新输出目录加载刚保存的 checkpoint：

~~~powershell
uv run --locked python scripts/train_pretrain.py --train-data tests/fixtures/pretrain_smoke/train.jsonl --validation-data tests/fixtures/pretrain_smoke/validation.jsonl --output-dir runs/week06-tutorial-restored --resume runs/week06-tutorial-first/checkpoint.pt
Get-Content runs/week06-tutorial-restored/result.json
~~~

这里预期 updates_this_run=0、global_step=2。因为保存的是本轮结束状态，数据已经消费完，恢复不意味着自动开启第二轮。真正“剩余数据继续训练”的证据来自第 8 节 A/B/C 对照。

接着运行入口测试：

~~~powershell
uv run --locked pytest tests/test_pretrain_entrypoint.py -q
~~~

测试在临时目录检查正常运行与恢复、缺失输入的非零退出和错误记录、已有输出不能被覆盖。没有必要为了练习再往长期保存的 runs 目录制造同样的失败产物。

## 10. 验收与范围

### 10.1 整体检查

完成教程后，先运行恢复实验，再运行相关测试：

~~~powershell
uv run --locked python scripts/inspect_training_resume.py
uv run --locked pytest tests/test_sampler.py tests/test_checkpoint.py tests/test_pretrain_entrypoint.py -q
uv run --locked pytest -q
~~~

本周验收看以下行为，不以测试条数作为学习目标：

- 验证 NLL 按有效 token 加权，评估不更新梯度，结束后恢复原来的模型模式。
- checkpoint 能恢复模型、AdamW 历史、调度器、步数和已支持的随机数状态。
- sampler 恢复到已完成更新的位置，未提交部分重新读取。
- 完整窗口和尾部窗口都能提交正确的样本数。
- A/B/C 的最终参数、优化器、调度器和数据状态一致。
- JSONL 入口有结果、有恢复路径、有失败退出，并且不覆盖已有产物。

### 10.2 测试分工

| 文件 | 主要保护的行为 |
| --- | --- |
| [test_training.py](../implementation/tests/test_training.py) | 更新、累积与验证等训练基础行为 |
| [test_sampler.py](../implementation/tests/test_sampler.py) | 顺序、提交边界、非法状态和复制隔离 |
| [test_checkpoint.py](../implementation/tests/test_checkpoint.py) | 状态恢复、消费提交和失败边界 |
| [test_pretrain_entrypoint.py](../implementation/tests/test_pretrain_entrypoint.py) | 命令行入口、产物与退出行为 |

测试的作用是让之后修改代码时及时发现行为变化。首次学习先理解“这个测试防止什么错”，不必为增加数量重复手写同类断言。

### 10.3 实践记录

2026-10-18 的阶段验收记录为：本周三个专项测试文件合计 19 项通过，全仓 249 项通过；RTX 4060 Ti 上的 checkpoint CPU/CUDA 随机数恢复检查通过。这里是历史证据，不保证未来代码变更后仍然自动成立。

核心模块沿对话由学习者在提示下实现，部分测试、收尾验证和文档整理由助手协助。这份教程不把协作完成的工作写成学习者独立完成，也不把文档重写视为新一轮正式训练。

2026-10-19 重写核验：从本文第 8 节直接提取代码，组合运行 A/B/C 对照通过；第 9 节各代码块组合后的 Python 语法树与仓库训练入口一致；文内相对链接全部有效；全仓回归 249 项通过，耗时 28.77 秒。本次没有重新进行 CUDA 专项验收。

### 10.4 当前边界

当前得到的是可以运行、可以验证恢复行为的小型单轮训练工程，不是覆盖所有运行场景的生产框架：

- 保存与恢复没有完整冻结数据、tokenizer、配置和代码指纹。换了同长度但不同内容的数据，不一定会被拦截。
- 入口在本轮训练后保存一次，不支持任意 micro-batch 中间无损恢复。
- 本周验证限定单进程数据加载；多 worker 预取、多 epoch、分布式训练还没有纳入合同。
- 尚未保存 Python、NumPy 的随机状态及混合精度 scaler 等本项目未接入的状态。
- 临时文件替换减少半文件风险，但不能等同于任何文件系统故障下的持久性保证。
- load_state_dict 并不是事务；后续组件加载失败，前面组件可能已被修改，应重新构造对象再恢复。

下一阶段再进入第七周的数据与实验准备，把“短小正确的运行”推进到有明确数据、预算和评价标准的训练。**本周到这里收尾，下面只补充学习中问过的问题，不新增实践任务。**

## 11. 学习问答

### 11.1 eval、no_grad 和 was_training 有什么区别？

model.eval() 改变 Dropout、BatchNorm 等模块的行为，不等于关闭自动求导，也不是冻结所有参数。torch.no_grad() 才是不为这段前向构建常规反向图。was_training 保存评估前的模式，finally 中 model.train(was_training) 将它恢复；原来若是 eval，也应仍然是 eval。

### 11.2 optimizer 为什么同时有 param_groups 和 state？

param_groups 是参数分组列表，每组字典保存 params 引用和 lr、weight_decay 等共同更新配置。state 是以具体 Parameter 对象为键的映射，AdamW 为各参数保存 step、exp_avg、exp_avg_sq。多个参数能共享学习率，但梯度历史不同，不能共用一份 m/v。

运行中的 optimizer.state[parameter] 用对象索引；序列化后的 state_dict 使用参数编号组织状态。加载依赖参数组与参数顺序匹配，不能把编号当成模型参数名称。

### 11.3 state_dict 是文件吗，拿到之后会自动冻结吗？

它是 Python 字典形式的状态描述，不是文件。torch.save 才把内容写到磁盘。某些 state_dict 中的张量仍与活对象共享存储，所以只把字典放到变量里，不一定得到不会变化的历史快照。要保留内存对照值，通常需要对应张量的 detach().clone() 或合适的深复制；立即序列化则保存序列化时的内容。

### 11.4 weights_only 和 map_location 分别做什么？

map_location="cpu" 让加载出的存储先落在 CPU，避免不必要的显存分配。weights_only=True 使用受限的反序列化路径；它并不表示字典中只能有模型权重，本教程的张量、数字、列表与字典组合也可以保存。仍然只加载可信来源的文件，不能把这个选项当成不受信任文件的绝对安全保证。

### 11.5 不同文件 import torch 会共享随机数状态吗？

同一进程里通常使用同一个导入的 torch 模块，调用默认 CPU 随机函数会推进同一默认 CPU Generator。CUDA 各设备还有自己的随机数状态。显式创建的 torch.Generator 可以独立管理；Python random 和 NumPy 又属于另外的随机数系统。不同文件不等于不同随机数空间。

### 11.6 Sampler[int]、Iterator[int] 和 yield 是什么？

Sampler[int] 表示这个 sampler 迭代时产出整数索引，是泛型类型标注，不是额外创建一种打乱算法。Iterator[int] 表示一个逐个返回整数的迭代器。含 yield 的函数每次返回一个值并暂停，下次取值再继续。类型提示帮助读者和检查工具理解接口，不会自动阻止所有不符合类型的运行时输入。

### 11.7 为什么列表要 copy，整数不用？

赋值保存的是对象引用。列表可变，两个名字指向同一列表时，原地修改能被双方看到；整数不可变，给一个名字加一是让该名字绑定到新的整数，不会修改旧整数。

浅复制新建外层容器，里面的对象仍共用；深复制递归复制内部可复制的对象。本例列表元素都是不可变整数，浅复制足够。区别不是“复制值还是复制地址”，而是复制到哪一层。

### 11.8 为什么索引校验使用 type(index) is int？

Python 中 True == 1，1.0 == 1，因此仅检查集合或排序结果可能接受布尔值和浮点数。本教程要求真实整数索引，显式排除这些看起来相等但类型不符的值。bool 还是 int 的子类，单用 isinstance(index, int) 也会接受布尔值。

### 11.9 Dataset 必须继承 Dataset 类吗？

对这里的 map-style 用法，普通列表或提供 __len__ 与 __getitem__ 的对象也可以使用；需要满足 DataLoader 按索引读取的协议。IterableDataset 走的是另一条迭代式路径，不能直接套用这里的随机索引 sampler。

### 11.10 只保存 position 为什么不够？

position 表示索引已发出，不代表对应梯度已经用于参数更新。数据可能还在 pending_batches，或者更新已经失败。checkpoint 应保存能与模型更新状态对应的 committed_position。DataLoader 预取会进一步扩大“已读取”和“已训练”的差距。

### 11.11 已经保存 RNG，为什么还保存完整索引顺序？

RNG 状态只能说明随机数流走到了哪里，不能自动告诉我们哪个模块在何时生成过排列。本周直接保存排列与已提交位置，使恢复不依赖重新生成顺序的调用过程。两者职责不同，不能互相替代。

### 11.12 B 只训练两批，为什么 loss 和梯度范数与 A 一样？

两批组成第一次更新窗口，B 的这次更新与 A 的第一次更新拥有相同初始权重、相同数据和相同计算过程，因此记录一致。A 之后还更新了一次；不能据此说 B 的最终模型已经等于完整训练后的 A。

### 11.13 global_step 为什么需要手动保存？

它是我们记录已完成更新次数的整数，不是模型参数。模型的 state_dict 不知道训练入口把什么算作一步。保存时写入，加载时返回，由入口加上本次成功完成的更新次数，避免恢复后日志从零开始或学习计划被误解。

### 11.14 比较优化器状态为什么先按参数名称对齐？

A、C 是重新创建的模型，对应 Parameter 对象不是同一个 Python 对象。先按参数名称找出相同位置，再分别用各自对象访问各自 optimizer.state，才能比较对应的 m/v。不能用 A 的 Parameter 去索引 C 的优化器。

### 11.15 assert_close 的 rtol、atol 和 msg 有什么作用？

rtol 控制随数值量级变化的相对误差，atol 提供接近零时的绝对误差容许量。torch.testing.assert_close 对有限数值使用类似“绝对差不超过 atol + rtol × 参考值绝对值”的判断。pytest.approx 的容差组合规则不同，不要把两者公式混用。

源码中的 msg=lambda message, name=name: ... 是失败信息处理函数：message 由 assert_close 提供，name=name 捕获当前参数名称，让错误指出是哪一层不一致。

### 11.16 为什么保存错误后还要 raise？

记录异常方便人阅读，重新抛出异常让进程以失败状态结束，脚本调用方才不会误认为成功。普通 except Exception 不兜住所有进程终止情形；强杀、断电等尤其不能保证有错误文件。

### 11.17 有 checkpoint 就算本次运行成功了吗？

不一定。当前入口先保存，再做验证。验证报错时可以有 checkpoint.pt 和 error.txt，却没有 result.json。训练状态能恢复，与整次任务所有步骤完成，是两个不同判断。

### 11.18 加载已完成本轮的 checkpoint，为什么没有新增训练？

恢复的是原来的训练状态和数据位置，不是隐式开始新的 epoch。已提交五条、总共五条，剩余数据为空，所以新增更新为零。要继续新一轮，需要显式设计 epoch 和新的顺序管理，而不是把 committed_position 随意归零。

### 11.19 这与 MiniMind 的训练代码是什么关系？

我们沿着 MiniMind 的模型与训练主题学习，但当前实现是逐步搭建的教学工程，不是逐行复制官方入口。模型前向、loss、反向、优化器、验证和 checkpoint 的基本职责相通；具体数据、配置、分布式和保存策略必须按对应版本核对。“尚非生产框架”描述覆盖范围，不意味着已经运行和验证的训练是假的。


