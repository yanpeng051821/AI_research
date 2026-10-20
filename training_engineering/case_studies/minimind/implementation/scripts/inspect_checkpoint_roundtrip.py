"""Check a real checkpoint round trip, including CPU and CUDA RNG states."""

import tempfile
from pathlib import Path

import torch

from minimind_lab.training.checkpoint import load_checkpoint, save_checkpoint


def make_training_objects(device):
    model = torch.nn.Linear(2, 1, bias=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    return model, optimizer, scheduler


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires a CUDA device")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model, optimizer, scheduler = make_training_objects(device)

    # 1. Create real AdamW moments and advance the learning-rate schedule.
    inputs = torch.tensor([[1.0, 2.0]], device=device)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    with tempfile.TemporaryDirectory(prefix="minimind-checkpoint-") as directory:
        path = Path(directory) / "checkpoint.pt"

        # 2. Save, then record the next draws as the reference continuation.
        save_checkpoint(path, model, optimizer, scheduler, global_step=1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        expected_cpu = torch.rand(5)
        expected_cuda = torch.rand(5, device=device)

        # 3. Consume more randomness and rebuild all three training objects.
        torch.rand(17)
        torch.rand(19, device=device)
        restored, restored_optimizer, restored_scheduler = make_training_objects(device)
        assert not torch.equal(restored.weight.detach().cpu(), saved["model"]["weight"])

        # 4. Restore and inspect states before consuming any new random numbers.
        step = load_checkpoint(
            path, restored, restored_optimizer, restored_scheduler
        )
        assert step == 1
        torch.testing.assert_close(
            torch.get_rng_state(), saved["torch_rng_state"], rtol=0, atol=0
        )
        cuda_states = torch.cuda.get_rng_state_all()
        assert len(cuda_states) == len(saved["cuda_rng_state_all"])
        for actual, expected in zip(cuda_states, saved["cuda_rng_state_all"]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # 5. The resumed generators must produce the original next draws.
        actual_cpu = torch.rand(5)
        actual_cuda = torch.rand(5, device=device)
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0)
        torch.testing.assert_close(actual_cuda, expected_cuda, rtol=0, atol=0)

        # 6. Check weights, AdamW moments and the scheduler against the file.
        torch.testing.assert_close(
            restored.weight.detach().cpu(), saved["model"]["weight"],
            rtol=0, atol=0,
        )
        actual_state = restored_optimizer.state[restored.weight]
        saved_state = next(iter(saved["optimizer"]["state"].values()))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                actual_state[key].cpu(), saved_state[key], rtol=0, atol=0
            )
        assert actual_state["exp_avg"].device == device
        assert actual_state["exp_avg_sq"].device == device
        assert restored_optimizer.state_dict()["param_groups"] == saved["optimizer"]["param_groups"]
        assert restored_scheduler.state_dict() == saved["scheduler"]
        assert restored_optimizer.param_groups[0]["lr"] == 0.0005
        assert restored_scheduler.last_epoch == 1

        print("device:", torch.cuda.get_device_name(device))
        print("CPU expected:", expected_cpu)
        print("CPU restored:", actual_cpu)
        print("CUDA expected:", expected_cuda.cpu())
        print("CUDA restored:", actual_cuda.cpu())
        print("PASS: exact CPU/CUDA RNG states and next random values")
        print("PASS: weights, AdamW state/device, learning rate, scheduler, global_step")


if __name__ == "__main__":
    main()
