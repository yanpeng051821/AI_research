# 优化：从梯度到参数变化

本目录解释梯度算出后，怎样组织参数更新。先连接已理解的计算图与梯度，再研究优化器如何使用当前梯度和历史状态。

当前正文是[从梯度到参数更新：SGD 与 AdamW](01_SGD_ADAMW_AND_PARAMETER_UPDATES.md)。它从一个标量实验出发，依次说明普通 SGD、历史梯度统计、偏差修正、更新尺度和权重衰减，让每组公式都有明确输入与输出。

阅读前可回顾[自动求导与参数更新](../01_model_architecture/foundations/08_AUTOGRAD_AND_PARAMETER_UPDATES.md)，其中解释梯度的范围、链式法则、叶子张量、梯度保存与计算图关系。

通用原理在这里沉淀；具体脚本、报错、运行数值和待完成事项在[MiniMind 第五周实践](../../training_engineering/case_studies/minimind/practice_notes/week_05_training_system.md)中维护。阅读正文不等于已完成对应实践。

后续学到学习率调度、梯度累积与裁剪时，按实际内容扩展，不提前创建空白文章或增加另一份学习路线。
