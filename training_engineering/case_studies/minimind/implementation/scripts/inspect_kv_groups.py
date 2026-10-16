
import torch

kv = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)

expanded = kv.repeat_interleave(repeats=2, dim=2)
assert expanded.shape == (1, 2, 4, 2)

torch.testing.assert_close(kv[:,:,0,:], expanded[:,:,0,:])
torch.testing.assert_close(kv[:,:,0,:], expanded[:,:,1,:])
torch.testing.assert_close(kv[:,:,1,:], expanded[:,:,2,:])
torch.testing.assert_close(kv[:,:,1,:], expanded[:,:,3,:])










