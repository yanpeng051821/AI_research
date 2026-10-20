

from pathlib import Path

import torch

def save_checkpoint(path, model, optimizer, scheduler, global_step, sampler=None):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "sampler_state": sampler.state_dict() if sampler is not None else None,
    }

    # 临时文件路径
    tmp_path = save_path.with_name(save_path.name + ".tmp")

    # 保存权重到临时文件
    torch.save(checkpoint, tmp_path)

    # 将完整文件替换到正式路径
    tmp_path.replace(save_path)

    return save_path

def load_checkpoint(path, model, optimizer, scheduler, sampler=None):
    load_path = Path(path)

    checkpoint = torch.load(
        load_path,
        map_location="cpu",
        weights_only=True,
    )

    cuda_states = checkpoint["cuda_rng_state_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA state but CUDA is unavailable")
        if len(cuda_states) > torch.cuda.device_count():
            raise ValueError("CUDA device count does not match checkpoint")

    if sampler is not None:
        sampler_state = checkpoint.get("sampler_state")
        if sampler_state is None:
            raise ValueError("cannot restore sampler: checkpoint has no sampler state")
        sampler.load_state_dict(sampler_state)

    # gpu检查条件满足，才能开启权重的加载
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = checkpoint["global_step"]
    torch.set_rng_state(checkpoint["torch_rng_state"])


    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    
    return global_step




