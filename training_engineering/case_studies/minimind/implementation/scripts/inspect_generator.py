
import torch

generator = torch.Generator()
generator.manual_seed(42)

order = torch.randperm(5, generator=generator)
indices = order.tolist()


print("Tensor:", order)
print("列表:", indices)
print("第一个样本索引:", indices[0])
print("索引数量:", len(indices))





