# MiniMind-3：CPU FP32 数值对照

> 2026-10-24。RMSNorm 运算写法对齐后，小尺寸与正式 Dense 尺寸的 CPU FP32 前向、loss 与梯度对照均通过。未执行参数更新或训练。

## 1. 条件与方法

参考提交 f659b55761b754d306bd140573493a6543cafd7f，模型源码 SHA256 为 548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f。使用 CPU FP32、官方手写 Attention、dropout=0，不启用 KV cache。先映射并严格加载相同权重，核对参数集合与权重共享，再使用相同输入比较 logits、causal loss 和全部独立参数梯度。Q/K Norm 缩放为非单位值。预设 atol=1e-6、rtol=1e-4。

## 2. 小尺寸结果

模型为 2 层、hidden=16、FFN=32、vocab=32、Q=4、KV=2，共 5,216 个独立参数、24 份参数梯度。输入为两条 4-token 序列，完整条件、源码哈希及依赖版本见 [tiny_report.json](tiny_report.json)。

| 输入 | logits 最大绝对差 | loss 绝对差 | 梯度最大绝对差 |
| --- | ---: | ---: | ---: |
| 无 padding | 2.9802322387695312e-8 | 0 | 1.1920928955078125e-7 |
| 右侧 padding | 2.9802322387695312e-8 | 0 | 1.1920928955078125e-7 |

两组均通过。结论限于已执行条件，不覆盖正式长度、CUDA autocast、SDPA/Flash、KV cache 或参数更新。

## 3. 正式尺寸诊断

脚本切换到 8 层、hidden=768、FFN=2432、vocab=6400、Q=8、KV=4；仍使用两条 4-token 输入。官方 RoPE 表只预计算到 32，足以覆盖本次输入，不代表正式上下文长度测试。

首次无 padding 的 logits 比较失败：51,200 个元素中 6 个不满足容差，失败元素的最大绝对差为 1.5422701835632324e-6。未提高容差。

本地 RMSNorm 使用 x / sqrt(mean(x^2)+eps)，官方使用 x * rsqrt(mean(x^2)+eps)。助手只在临时诊断进程中把官方 RMSNorm 改为本地除法写法，其余条件与容差保持相同。两组输入的 logits 最大差均变为 0，loss 差为 0，梯度最大差分别为 1.3560056686401367e-6 和 1.3709068298339844e-6，逐元素容差比较通过。最大绝对差本身不是全部判据，rtol 也参与比较；梯度并非逐位相同。

该诊断支持将前向差异定位到 RMSNorm 浮点运算。修改过的参考只用于定位，不能作为未修改官方模型的正式通过证据。随后由学习者将本地写法对齐官方，再运行原始参考验收，结果如下。

### 3.1 修正后的正式验收

参考源码保持原始内容，并通过既有 SHA256 校验；没有提高容差。正式尺寸比较 90 份独立参数的梯度，结果见 [dense_report.json](dense_report.json)。

| 输入 | logits 最大绝对差 | loss 绝对差 | 梯度最大绝对差 |
| --- | ---: | ---: | ---: |
| 无 padding | 0 | 0 | 1.1920928955078125e-6 |
| 右侧 padding | 0 | 0 | 1.4454126358032227e-6 |

两组均满足预设逐元素容差；梯度并非逐位相同。修正后小尺寸也重新通过，见 [tiny_rsqrt_report.json](tiny_rsqrt_report.json)。第 2 节的 tiny_report.json 保留为修正前的历史证据，文件内记录的本地模型哈希可区分两次状态。

该验收限于短输入、CPU FP32、手写 Attention、无缓存路径，尚未验证 CUDA autocast、正式训练长度、实际参数更新或训练效果。通用精度原理见[训练数值精度与混合精度](../../../../../model_architecture_and_algorithms/03_optimization/02_NUMERICAL_PRECISION_AND_MIXED_PRECISION.md)。

## 4. 复练

在 implementation 目录执行，输出目录使用新目录：

~~~powershell
uv run --locked python scripts/compare_minimind3_reference.py --profile tiny --output-dir runs/alignment_tiny_recheck
uv run --locked python scripts/compare_minimind3_reference.py --profile dense --output-dir runs/alignment_dense_recheck
~~~

脚本见 [compare_minimind3_reference.py](../../implementation/scripts/compare_minimind3_reference.py)。首次下载一份冻结源码并校验哈希，之后可用本地缓存，不下载模型权重。正式尺寸同时创建两份模型及梯度。脚本仅在完整比较通过后写 report.json；出现断言失败需检查原因，不能以目录是否存在判断完成。

核心模型修改由学习者完成；对照脚本、权重映射、回归测试与本次归档由助手完成。
