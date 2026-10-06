from minimind_lab.models.feedforward import SwiGLU

import torch


model = SwiGLU(4, 8)
input = torch.rand(2, 3, 4)
output = model(input)
print(output.shape)