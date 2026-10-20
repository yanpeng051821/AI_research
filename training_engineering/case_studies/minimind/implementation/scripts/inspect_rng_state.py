import torch

torch.manual_seed(42)

first = torch.rand(3)
state = torch.get_rng_state()
expected_next = torch.rand(3)

# 模拟中间又发生了其他随机操作
torch.rand(10)

torch.set_rng_state(state)
restored_next = torch.rand(3)

print("first:", first)
print("expected next:", expected_next)
print("restored next:", restored_next)

torch.testing.assert_close(
    restored_next, expected_next, rtol=0, atol=0
)