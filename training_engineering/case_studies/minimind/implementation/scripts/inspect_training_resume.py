
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tempfile import TemporaryDirectory

from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.loop import train_epoch
from minimind_lab.training.step import train_update
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint




def make_loader():
    input_ids = torch.tensor([
        [1, 3, 2],
        [1, 4, 2],
        [1, 5, 2],
        [1, 6, 2],
        [1, 7, 2],
    ])

    dataset = [
        {
            "input_ids": row.clone(),
            "attention_mask": torch.ones_like(row),
            "labels": row.clone()
        }
        for row in input_ids
    ]

    sampler = StatefulSampler(size=len(dataset), seed=42)
    generator = torch.Generator()
    generator.manual_seed(123)

    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )

    return loader, sampler

def make_training_objects():
    config = ModelConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
    )
    
    model = MinimindForCausalLM(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.02,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.5,
    )

    return model, optimizer, scheduler


if __name__ == "__main__":
    torch.manual_seed(42)

    # 一次性跑完
    model_a, optimizer_a, scheduler_a = make_training_objects()
    loader_a, sampler_a = make_loader()

    history_a = train_epoch(
        model_a,
        optimizer_a,
        loader_a,
        scheduler_a,
        gradient_accumulation_steps=2,
        sampler=sampler_a,
    )

    print(f"history_a: {history_a}")
    print(f"committed_position: {sampler_a.committed_position}")
    print(f"schedule step: {scheduler_a.last_epoch}")

    # 中断恢复
    torch.manual_seed(42)
    model_b, optimizer_b, scheduler_b = make_training_objects()
    loader_b, sampler_b = make_loader()

    iterator_b = iter(loader_b)
    pendding_batches_b = [next(iterator_b), next(iterator_b)]

    first_result_b = train_update(
        model_b,
        optimizer_b,
        pendding_batches_b,
        scheduler_b,
    )

    sample_count = sum(
        batch["input_ids"].shape[0] 
        for batch in pendding_batches_b
    )
    sampler_b.commit(sample_count)

    print("B first update:", first_result_b)
    print("B committed:", sampler_b.committed_position)

    with TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"

        save_checkpoint(
            checkpoint_path,
            model_b,
            optimizer_b,
            scheduler_b,
            global_step=1,
            sampler=sampler_b,
        )

        # 模拟重新启动
        torch.manual_seed(999)
        model_c, optimizer_c, scheduler_c = make_training_objects()
        loader_c, sampler_c = make_loader()

        restored_step = load_checkpoint(
            checkpoint_path,
            model_c,
            optimizer_c,
            scheduler_c,
            sampler=sampler_c,
        )

        print("restored step:", restored_step)
        print("restored position:", sampler_c.position)
        print("restored committed:", sampler_c.committed_position)
        print("restored lr:", optimizer_c.param_groups[0]["lr"])

        history_c = train_epoch(
            model_c,
            optimizer_c,
            loader_c,
            scheduler_c,
            gradient_accumulation_steps=2,
            sampler=sampler_c,
        )

        print("resumed history:", history_c)
        print("final committed:", sampler_c.committed_position)
        print("final scheduler step:", scheduler_c.last_epoch)

        final_step = restored_step + len(history_c)
        print("final global step:", final_step)

        parameters_a = dict(model_a.named_parameters())
        parameters_c = dict(model_c.named_parameters())

        assert parameters_a.keys() == parameters_c.keys()

        for name, parameter_a in parameters_a.items():
            torch.testing.assert_close(
                parameter_a,
                parameters_c[name],
                rtol=1e-5,
                atol=1e-7,
                msg=lambda message, name=name: f"{name}: {message}",
            )

        print("final model parameters match")

        for name, parameter_a in parameters_a.items():
            parameter_c = parameters_c[name]

            state_a = optimizer_a.state[parameter_a]
            state_c = optimizer_c.state[parameter_c]

            assert state_a.keys() == state_c.keys()

            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    state_a[key],
                    state_c[key],
                    rtol=1e-5,
                    atol=1e-7,
                    msg=lambda message, name=name, key=key: (
                        f"{name}.{key}: {message}"
                    ),
                )

        print("optimizer states match")

        assert optimizer_a.state_dict()["param_groups"] == (
            optimizer_c.state_dict()["param_groups"]
        )
        assert scheduler_a.state_dict() == scheduler_c.state_dict()
        assert sampler_a.state_dict() == sampler_c.state_dict()
        assert final_step == len(history_a) == 2

        print("optimizer groups, scheduler, sampler and global step match")

