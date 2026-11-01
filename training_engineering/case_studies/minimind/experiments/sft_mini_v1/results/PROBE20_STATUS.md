# 真实GPU短跑与分配排查记录

## 1. 当前结论

2026-10-31，前三次启动均发生CUDA分配失败；用户进一步关闭应用后，第四次按相同配方完成20次更新、320条记录并保存checkpoint，状态paused，退出0，见[成功短跑报告](probe20_v4/README.md)。前三次失败记录保留，不覆盖；未改动batch、学习率、模型代码、精度或冻结数据，未启动正式训练。

| 启动 | 阶段 | 已完成更新 | 结果 |
| --- | --- | ---: | --- |
| [probe20_v1](probe20_v1/run_manifest.json) | 初始验证forward，softmax分配72MiB失败 | 0 | failed，训练尚未开始 |
| [probe20_v2](probe20_v2/run_manifest.json) | 初始验证通过，首个训练forward分配72MiB失败 | 0 | failed，训练尝试已开始但未更新 |
| [probe20_v3](probe20_v3/run_manifest.json) | 用户要求复查后按原配方重试；初始验证通过，首个训练forward的mask分配72MiB失败 | 0 | failed，训练尝试已开始但未更新 |
| [probe20_v4](probe20_v4/run_manifest.json) | 用户进一步关闭应用，系统余量明显改善，同配方通过验证与20次更新 | 20 | paused，checkpoint已保存，未正式长训 |

原始traceback保存在各目录error.txt，候选身份保存在candidate_contract.json。第二次验证事件保存在[metrics.jsonl](probe20_v2/metrics.jsonl)：BF16工程前缀64条、27,337个有效助手目标，NLL=2.8312329279284945，验证约2.62秒。这不是完整验证集FP32基线，不能与B0全量NLL直接比较。

## 2. 观察与待验证假设

1. 本机RTX4060Ti 8GB使用WDDM。失败时日志仍报告空闲显存约6.23GiB或5.72GiB，PyTorch活跃显存约580.61MiB或1.03GiB。仅凭OutOfMemoryError不能断言batch已经耗尽8GB物理显存。
2. 报错中的多个进程占用17179869184GiB显然不是可信实测值，不把它当成真实显存消耗，也不据此诊断泄漏。
3. 第一次失败后，系统报告物理内存约15.82GiB、可用约2.88GiB；虚拟/提交额度相关WMI指标总量约31.82GiB、可用约3.37GiB。D盘分页文件配置16,384MiB、当时使用7,830MiB。系统内存压力是待验证因素，不是已经确认的唯一根因；读数来自失败后的快照，不是失败瞬间连续采样。
4. 同配方只重试一次：第二次初始验证成功，但第一轮训练forward失败。没有继续盲目循环重试，也没有未经确认调低batch或修改驱动/分页设置。

微软文档说明WDDM存在动态的进程驻留预算，内存压力可影响预算；这支持继续检查Windows内存管理因素，但不足以证明本次特定失败的原因。[官方说明](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/process-residency-budgets)

## 3. 下一步

第三次运行前WMI读数：可用物理内存约1.72GiB、可用虚拟/提交相关空间约4.42GiB；GPU桌面占用1,452MiB。失败时PyTorch活跃约2.26GiB、未用缓存约73.94MiB，CUDA仍报告4.51GiB空闲。两次成功的初始验证NLL相同，但训练均未完成更新。不能把系统压力缓解等同于训练已经可运行。

随后做独立有界分配诊断：[cuda_allocation_v1.json](cuda_allocation_v1.json)。不加载模型，每次申请并填充64MiB GPU张量，上限1GiB，系统提交余量不足1.5GiB则提前停止。实际成功申请1GiB：系统提交量由28,908,990,464字节升至30,276,710,400字节；除首次分配的额外开销外，后续每增加64MiB显存，提交量也近似增加64MiB。诊断结束释放全部张量和CUDA缓存。这支持本机GPU分配会消耗系统提交额度的判断，但没有采到模型失败瞬间提交量，仍不将其写成唯一已确诊根因。

诊断入口为[inspect_cuda_allocation.py](../../../implementation/scripts/inspect_cuda_allocation.py)，没有修改模型或训练入口。微软说明系统提交额度受RAM和分页文件共同约束：[分页文件与提交上限](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)。独立分配成功不代表真实训练所需的激活、梯度和Adam状态均可分配。

已请求用户保存工作、关闭暂时不用的应用，未主动终止任何用户进程或修改系统配置。内存压力缓解后，优先按原候选配置、新输出目录重试，以观察是否改变结果；若仍失败，再做独立的最小CUDA分配/执行诊断，不能悄悄换配方并称原配置通过。

即使20步短跑成功，仍需真实恢复对照、耗时/预算及保存验证策略确认、最终配方冻结，之后才启动正式训练。

## 4. 环境改善后的成功对照

用户进一步关闭应用后，运行前可用物理内存约7.24GiB，系统虚拟/提交相关可用约15.22GiB，GPU桌面占用1,041MiB。同配置运行成功，峰值allocated约3.77GiB，末次reserved约3.88GiB，更新后allocated约0.79GiB，20步内稳定。结合独立分配实测，这强烈支持系统内存/提交压力是先前失败的影响因素；并没有逐一隔离应用、驱动预算等因素，不将其称为单一根因已被严格证明。

保存证据与checkpoint内部状态核对通过；真实恢复对照尚未执行。按短跑窗口中位数外推纯训练约10.2小时，不含验证、保存、启动与长期波动。后续以成功短跑报告为当前状态，不再沿用前三次“0步、无checkpoint”的结论。
