# 第 8 周：LoRA 低秩适配

第 8 周开始进入参数高效微调。

前面我们已经完成：

```text
pretrain -> SFT -> eval -> experiment
```

现在的问题变成：

```text
如果我已经有一个 SFT 模型，
想让它适配医疗、考试、教育助教、用户画像、代码解释等垂类任务，
是否每次都要重新训练全部参数？
```

LoRA 的回答是：

```text
不一定。
可以冻结原模型，只训练一小部分新增参数。
```

## 1. LoRA 解决什么问题

全量 SFT 会更新模型里的大量参数。

例如一个线性层：

```text
y = xW
```

如果 `W` 很大，全量微调就要更新完整的 `W`。

这会带来几个问题：

- 显存开销大。
- 训练成本高。
- 每个垂类任务都要保存一份完整模型。
- 多个任务之间切换不方便。

LoRA 的想法是：

```text
原来的 W 不动。
额外加一个很小的增量 ΔW。
训练时只训练 ΔW。
```

于是变成：

```text
y = xW + xΔW
```

LoRA 进一步假设：

```text
ΔW 可以用两个低秩矩阵表示。
```

也就是：

```text
ΔW = A B
```

在 MiniMind 代码里，由于输入先过 A 再过 B，所以写成：

```text
LoRA(x) = B(A(x))
```

最终输出：

```text
Linear(x) + LoRA(x)
```

## 2. 为什么叫低秩

假设原始线性层是：

```text
in_features = 768
out_features = 768
```

全量矩阵参数量是：

```text
768 * 768 = 589824
```

如果 LoRA rank = 16：

```text
A: 768 -> 16
B: 16 -> 768
```

LoRA 参数量是：

```text
768 * 16 + 16 * 768 = 24576
```

参数占比大约是：

```text
24576 / 589824 = 4.17%
```

这就是 LoRA 参数少的原因：

```text
它没有直接训练一个完整的 768 x 768 矩阵，
而是训练两个瘦长矩阵。
```

## 3. MiniMind 的 LoRA 类

源码位置：

```text
model/model_lora.py
```

核心代码：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

数据流：

```text
x
-> A: [B, T, C] -> [B, T, rank]
-> B: [B, T, rank] -> [B, T, C]
```

如果 `C=512`，`rank=16`：

```text
x:       [B, T, 512]
A(x):    [B, T, 16]
B(A(x)): [B, T, 512]
```

所以 LoRA 的输出 shape 和原始 Linear 输出 shape 一样，可以直接相加。

## 4. 为什么 B 初始化为 0

MiniMind 里：

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)
self.B.weight.data.zero_()
```

这里的 `normal_(mean=0.0, std=0.02)` 叫高斯初始化，也就是从正态分布里随机采样权重：

```text
均值是 0.0。
标准差是 0.02。
大多数值会落在 0 附近的小范围内。
```

所以 `A.weight` 不是全 0，而是一组很小的随机数。

`zero_()` 则是原地把 `B.weight` 全部置为 0。

这意味着训练刚开始时：

```text
LoRA(x) = B(A(x)) = 0
```

所以模型一开始的输出还是：

```text
Linear(x) + 0
```

好处是：

```text
刚加上 LoRA 时，不会立刻破坏原模型已经学到的能力。
```

然后训练过程中，B 和 A 会逐渐学出一个增量方向。

这很像在原模型旁边加了一条“可学习的旁路”：

```text
原模型负责通用能力。
LoRA 负责垂类增量。
```

为什么不是 A 和 B 都置 0？

```text
A 随机：提供不同的低维投影方向。
B 为 0：保证刚接入 LoRA 时，增量输出为 0，不破坏 base model。
```

训练一开始，B 会先收到有效梯度并被打开；当 B 不再是 0 后，A 也会逐渐参与有效学习。

## 5. apply_lora 做了什么

MiniMind 的 `apply_lora`：

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

它做了 4 件事：

```text
1. 遍历模型里所有模块。
2. 找到满足条件的 nn.Linear。
3. 给这个 Linear 挂一个 module.lora。
4. 替换这个 Linear 的 forward，让它变成 Linear(x) + LoRA(x)。
```

注意：

```python
setattr(module, "lora", lora)
```

等价于：

```python
module.lora = lora
```

这里的 `lora` 本身就是一个 `LoRA` 模块，不需要通过 `module.lora.linear` 访问。

挂载后的结构可以理解为：

```text
某个原始 Linear module
├── weight
├── bias
└── lora
    ├── A
    │   └── weight
    └── B
        └── weight
```

所以常见访问路径是：

```text
module.lora
module.lora.A
module.lora.B
module.lora.A.weight
module.lora.B.weight
```

注意 MiniMind 只给下面这种 Linear 加 LoRA：

```text
module.in_features == module.out_features
```

也就是说，它优先给方阵线性层加 LoRA。

在 Transformer 里，这通常会覆盖很多投影层，例如：

```text
q_proj
k_proj
v_proj
o_proj
部分 MLP 投影
```

具体覆盖哪些，要看模型中每个 Linear 的输入输出维度。

更具象地看，它不是给每个模块后面都加一层，而是遍历整棵模型树，只给符合条件的 Linear 内部并联一条 LoRA 支路：

```text
输入 x
├── 原始 Linear: Linear(x)
└── LoRA 支路: B(A(x))
最终输出 = Linear(x) + B(A(x))
```

## 6. monkey patch 是什么意思

这一句很关键：

```python
module.forward = forward_with_lora
```

它没有改 `nn.Linear` 类的源码。

它是在运行时，把当前这个 Linear 实例的 forward 函数替换掉。

原来：

```text
module.forward(x) = Linear(x)
```

替换后：

```text
module.forward(x) = Linear(x) + LoRA(x)
```

这就叫 monkey patch。

优点：

```text
代码简单，不需要重写整个模型结构。
```

缺点：

```text
和 torch.compile 等图编译机制可能不兼容。
```

所以 `train_lora.py` 里有：

```python
if args.use_compile == 1:
    args.use_compile = 0
    Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
```

## 7. LoRA 训练时冻结什么

`train_lora.py` 中：

```python
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

含义：

```text
名字里包含 lora 的参数：参与训练。
其他参数：冻结。
```

所以反向传播时：

```text
loss 会正常计算。
梯度会正常从输出往前传。
但只有 LoRA 参数会被 optimizer 更新。
```

这和 SFT 的差异非常关键：

```text
SFT：
更新全模型参数。

LoRA：
加载一个已有 SFT 模型，冻结原模型，只训练新增 LoRA 参数。
```

## 8. optimizer 只拿 LoRA 参数

MiniMind 里：

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

这意味着 optimizer 根本看不到原模型参数。

即使原模型参与了前向计算，它也不会被更新。

训练链路变成：

```text
input_ids
-> 原模型 + LoRA 分支
-> logits
-> labels 计算 loss
-> backward
-> 只更新 LoRA A/B 矩阵
```

## 9. save_lora 保存什么

`save_lora` 只保存 LoRA 参数：

```python
for name, module in raw_model.named_modules():
    if hasattr(module, 'lora'):
        lora_state = {
            f'{clean_name}.lora.{k}': v.cpu().half()
            for k, v in module.lora.state_dict().items()
        }
        state_dict.update(lora_state)
torch.save(state_dict, path)
```

也就是说，保存出来的：

```text
lora_xxx_512.pth
```

不是完整模型。

它只包含：

```text
各个 Linear 上挂载的 lora.A.weight
各个 Linear 上挂载的 lora.B.weight
```

这里容易混淆两个遍历：

```python
for name, param in model.named_parameters():
```

它是从整个模型视角看参数，返回的是完整路径和 `Parameter`，例如：

```text
model.layers.0.attention.wq.weight
model.layers.0.attention.wq.lora.A.weight
model.layers.0.attention.wq.lora.B.weight
```

而：

```python
for k, v in module.lora.state_dict().items():
```

它是走进某一个 LoRA 小模块内部看参数，返回的是局部名字和 `Tensor`：

```text
A.weight
B.weight
```

如果当前 Linear 的名字是：

```text
model.layers.0.attention.wq
```

那么保存时会把局部名字补成完整路径：

```text
model.layers.0.attention.wq.lora.A.weight
model.layers.0.attention.wq.lora.B.weight
```

以 `in_features=512`、`out_features=512`、`rank=16` 为例：

```text
A.weight: [16, 512]
B.weight: [512, 16]
```

因为 PyTorch 的 `nn.Linear(in_features, out_features)` 中，权重 shape 是：

```text
[out_features, in_features]
```

一句话记忆：

```text
model.named_parameters() 是从整栋楼看每个房间的完整地址。
module.lora.state_dict() 是走进一个 LoRA 小房间，只看里面的家具清单。
```

所以推理时如果要使用 LoRA，必须：

```text
先加载 base model。
再 apply_lora。
再 load_lora。
```

否则单独一个 LoRA 权重不能直接推理。

## 10. merge_lora 做什么

如果不想推理时还带着 LoRA 分支，可以把 LoRA 合并回原权重：

```python
state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data)
```

这对应公式：

```text
W' = W + ΔW
ΔW = B @ A
```

合并后得到的新模型：

```text
不再需要 LoRA 分支。
```

它变成一个普通模型权重。

## 11. LoRA 和教育 Agent 的关系

你未来想做教育 Agent。

LoRA 很适合做这些低成本适配：

- 小学数学讲解风格。
- 高中物理题解格式。
- 计算机基础教学助手。
- 用户错题总结风格。
- 学习计划生成风格。
- 特定课程知识库问答风格。

它不适合独自解决所有问题。

比如：

```text
实时查询天气
调用学习工具
长期记忆
复杂任务规划
```

这些更像 Agent harness、工具调用、记忆系统和后续 RL 的职责。

所以 LoRA 在你的路线里更像：

```text
让 base model 更懂某个学科/题型/教学风格。
```

不是：

```text
让模型凭空拥有完整 Agent 能力。
```

## 12. 本周第一层掌握标准

完成这一节后，你应该能回答：

- LoRA 为什么比全量 SFT 参数少？
- 为什么 LoRA 输出能和 Linear 输出相加？
- 为什么 B 初始化为 0？
- MiniMind 的 `apply_lora` 改了模型哪里？
- 为什么训练时只更新 LoRA 参数？
- LoRA 权重为什么不能单独推理？
- merge_lora 和 load_lora 的区别是什么？

## 13. 你的当前进度

你已经掌握：

```text
input_ids -> embedding -> block -> logits -> loss
pretrain labels
SFT assistant-only labels
训练循环
推理生成
采样参数实验
```

所以现在学习 LoRA 时，不需要重新理解训练闭环。

只需要把下面这个问题嵌入旧框架：

```text
同样的 SFT 训练循环里，哪些参数会被更新？
```

答案就是：

```text
只有 LoRA A/B 矩阵。
```

## 14. 关键 QA：LoRA 到底接在哪里

### 14.1 pretrain、SFT、LoRA 的区别是什么

可以放在同一条训练链路里理解：

```text
pretrain：
从相对随机初始化开始，使用大规模普通文本训练模型预测下一个 token。
目标是让模型学语言、常识、知识分布和基础推理模式。
更新：全模型参数。

SFT：
基于 pretrain 权重继续训练，使用指令/对话数据。
目标是让模型从“会续写文本”变成“会按用户指令回答”。
更新：通常也是全模型参数。

LoRA：
基于已有模型，通常是 pretrain 或 SFT 后的模型。
冻结原模型大部分参数，只在部分 Linear 层旁边加小的 A/B 矩阵。
目标是低成本适配某个垂类任务、风格或数据集。
更新：只更新 LoRA 新增参数。
```

更短的记法：

```text
pretrain：学通用语言能力。
SFT：学助手回答方式。
LoRA：在不大改原模型的情况下，学一个垂类增量。
```

需要注意：

```text
LoRA 不是更新原模块已有参数。
LoRA 是在原模块旁边新增参数，只训练这些新增参数。
```

### 14.2 低秩矩阵为什么值得研究

低秩矩阵的核心直觉是：

```text
很多复杂系统表面维度很高，但真正有效变化方向很少。
```

例如一个 `512 x 512` 的矩阵有：

```text
512 * 512 = 262144
```

个参数。

但一个具体任务可能不需要调整全部方向。

LoRA 假设有效更新可以表示成：

```text
Delta W = B @ A
```

如果 `rank=16`：

```text
A: 512 -> 16
B: 16 -> 512
LoRA 参数量 = 512*16 + 16*512 = 16384
```

这比完整更新少很多。

所以低秩的魅力在于：

- 省参数。
- 省显存。
- 易迁移。
- 不容易破坏原模型。
- 能抓住高维数据里的低维有效结构。

这个思想不只在 LoRA 里有，也出现在：

```text
PCA
SVD
推荐系统矩阵分解
图像压缩
表示学习
```

可以记成一句话：

```text
高维不等于高自由度。
```

### 14.3 LoRA 是给整个 model 加一个增量矩阵吗

不是。

LoRA 不是：

```text
整个模型：Model(x) + 一个巨大的 LoRA(x)
```

而是：

```text
某个 Linear 层：Linear_lora(x) = Linear(x) + LoRA(x)
```

如果模型里有很多个符合条件的 Linear，就会有很多套 LoRA。

例如：

```text
layer_1.linear: W1 + Delta W1
layer_2.linear: W2 + Delta W2
layer_3.linear: W3 + Delta W3
```

每个 Linear 都有自己的 A/B 矩阵。

不是所有层共享一个 `Delta W`。

在 Transformer 里，常见加 LoRA 的位置包括：

```text
Attention: q_proj / k_proj / v_proj / o_proj
MLP: gate_proj / up_proj / down_proj
有时也包括 lm_head
```

MiniMind 的实现更简化：

```python
if isinstance(module, nn.Linear) and module.in_features == module.out_features:
```

也就是只给输入输出维度相同的 Linear 加 LoRA。

### 14.4 LoRA 是怎么插入原模型架构的

LoRA 要新增线性层，但不是把模型结构大改。

它是在已有 `nn.Linear` 旁边外挂两层小 Linear。

原来的结构：

```text
x -> Linear(W) -> y
```

接入 LoRA 后：

```text
                 -> Linear(W) --------\
x ------------------------------------ + -> y'
                 -> Linear(A) -> Linear(B) -/
```

公式：

```text
y' = Linear(W)(x) + B(A(x))
```

MiniMind 通过 `apply_lora` 接入：

```python
lora = LoRA(module.in_features, module.out_features, rank=rank)
setattr(module, "lora", lora)
original_forward = module.forward

def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)

module.forward = forward_with_lora
```

这几行做了三件事：

```text
1. 新增 LoRA 小模块：module.lora = LoRA(...)
2. 保存原来的 Linear forward。
3. 替换当前 Linear 的 forward。
```

之后再调用：

```python
module(x)
```

实际执行的是：

```text
原 Linear 输出 + LoRA 分支输出
```

这就是 monkey patch。

它不是在 block 之间额外插一层，而是在每个被选中的 Linear 内部加一条并联支路。

### 14.5 LoRA 为什么不一定要求 Linear 前后维度一致

理论上，LoRA 不要求原始 Linear 必须是：

```text
in_features == out_features
```

它只要求：

```text
LoRA(x) 的输出 shape 能和原 Linear(x) 相加。
```

如果原 Linear 是：

```text
in_features -> out_features
```

那么 LoRA 可以是：

```text
in_features -> rank -> out_features
```

例如原 Linear：

```text
512 -> 2048
```

合法 LoRA：

```text
512 -> 16 -> 2048
```

因为最终输出也是：

```text
[B, T, 2048]
```

可以和原 Linear 输出相加。

非法 LoRA：

```text
512 -> 16 -> 512
```

因为它输出 `[B, T, 512]`，不能和 `[B, T, 2048]` 相加。

所以准确说：

```text
LoRA 不要求原 Linear 前后维度一致。
LoRA 只要求自己的输入维度等于原 Linear 输入维度，输出维度等于原 Linear 输出维度。
MiniMind 额外限制 in_features == out_features，是为了简化和控制训练范围。
```

MiniMind 这样做可能有几个原因：

- 简化教学代码。
- 控制 LoRA 参数量。
- 避免给所有投影层都加 LoRA。
- 降低初学者理解成本。

### 14.6 一个最终记忆图

把 LoRA 放回 Transformer 数据流中：

```text
input_ids
-> embedding
-> block 1
   -> attention 的若干 Linear + LoRA
   -> mlp 的若干 Linear + LoRA
-> block 2
   -> attention 的若干 Linear + LoRA
   -> mlp 的若干 Linear + LoRA
-> ...
-> logits
```

每当数据经过一个被 LoRA 改造过的 Linear：

```text
原始输出 = Linear(x)
增量输出 = LoRA(x)
最终输出 = 原始输出 + 增量输出
```

训练时：

```text
原始 Linear 的 W 冻结。
LoRA 的 A/B 更新。
```

推理时：

```text
base model + LoRA 权重一起使用。
```

## 15. 身份 LoRA 小实验结论

本次实操中，用 `lora_identity.jsonl` 约 91 条身份数据做 LoRA，尝试修改模型对“创建人/身份”的回答。

实验现象：

```text
训练流程可以正常跑通。
LoRA adapter 可以正常保存。
但推理时模型仍然倾向回答 base model 里原来的创建人名字。
```

这个现象说明：

```text
几十条身份 LoRA 数据不足以稳定覆盖 base model 中已有的强身份先验。
```

原因可以从三个角度理解：

- LoRA 是增量修正，不是重写底座记忆。base model 里已经形成的强回答模板，会继续影响输出。
- 91 条数据太少。如果 batch size 为 8、accumulation steps 为 4，一个 epoch 只有约 12 个 step，真正 optimizer 更新次数约 3 次。
- 身份问题需要高覆盖问法。只覆盖少量“你是谁/谁创建你”不够，需要大量同义问法和边界问法。

所以这次更准确的定位是：

```text
小规模身份 LoRA 适合验证训练流程。
但不适合期待它稳定改写模型身份。
```

如果要让身份 LoRA 更稳定，建议：

```text
身份数据至少扩展到 300-1000 条。
覆盖不同问法：你是谁、介绍你自己、谁创建你、你的开发者是谁、你的作者是谁、你和 MiniMind 是什么关系等。
训练多个 epoch，而不是只做 1 个 epoch 的流程测试。
用固定评测集统计命中率，而不是只看单条 prompt。
```

评测时不要只问一个问题，而是准备一组身份问题：

```text
20 个身份相关 prompt。
统计回答中包含目标名字的比例。
统计仍然回答旧创建人的比例。
统计混合身份或胡说的比例。
```

最终结论：

```text
如果只是想在产品中稳定显示身份，system prompt/persona prompt 最稳。
如果想让模型参数内化身份，full SFT 或更大规模身份 LoRA 更合适。
几十条 LoRA 数据更多是流程验证，不是可靠行为改写。
```

## 16. LoRA/SFT 数据文件如何组织

训练数据不一定要求“每一类都必须一个 jsonl”，但实际工程中通常建议按主题拆分。

推荐组织方式：

```text
dataset/
├── sft_general.jsonl
├── sft_education.jsonl
├── sft_math.jsonl
├── sft_coding.jsonl
├── lora_identity.jsonl
├── lora_physics_tutor.jsonl
└── lora_learning_planner.jsonl
```

这样做的好处：

- 数据来源清楚，后续排查质量问题更容易。
- 可以单独评估某一类数据的贡献。
- 可以灵活控制混合比例，比如身份数据 10%、教育问答 60%、学习规划 30%。
- 可以按任务训练不同 LoRA adapter，比如身份 LoRA、物理教学 LoRA、学习计划 LoRA。

训练时可以再合并、打散：

```text
多个主题 jsonl
-> 合并成一个 train_mix.jsonl
-> shuffle 打散
-> 喂给 SFTDataset 或 LoRA 训练脚本
```

为什么要打散？

```text
如果同一类样本连续出现，模型训练会阶段性偏向某一类模式。
打散后，每个 batch 的数据分布更均匀，训练更稳定。
```

但要注意，不是所有数据都应该简单等比例混合。

更合理的是做采样比例控制：

```text
通用能力数据：防止模型遗忘基础能力。
垂类任务数据：强化目标任务表现。
身份/风格数据：控制助手人设和回答口径。
边界数据：训练拒答、实时信息、未知问题等行为。
```

例如教育 Agent 的 LoRA 数据可以这样配：

```text
教育问答：50%
学习计划：20%
错题分析：15%
身份口径：5%
边界与拒答：10%
```

一句话总结：

```text
数据制作时按主题拆 jsonl，训练时按目标比例合并打散。
不要把所有数据一开始就混成一个不可追踪的大文件。
```
