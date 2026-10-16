
import torch

from minimind_lab.models.block import DecoderBlock
from torch import nn



decoder_block = DecoderBlock(16, 32, 4, 2)

x= torch.randn(2, 4, 16)
output, weights = decoder_block(x)
print(output.shape)
print(weights.shape)

with torch.no_grad():
    decoder_block.attention.o_proj.weight.zero_()
    decoder_block.feed_forward.swiglu.down_proj.weight.zero_()

output, weights = decoder_block(x)
torch.testing.assert_close(output, x)


blocks = nn.ModuleList([
    DecoderBlock(16, 32, 4, 2) for _ in range(2)
])

x = torch.randn(2, 4, 16)
hidden = x
for index, block in enumerate(blocks):
    hidden, _ = block(hidden)
    print(f"第 {index} 层, hidden.shape:{hidden.shape}")
    



















