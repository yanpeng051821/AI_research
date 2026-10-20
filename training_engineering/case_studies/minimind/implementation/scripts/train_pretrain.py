
import argparse
import torch
import json
import traceback

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import PretrainDataset, PretrainCollator
from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.loop import train_epoch
from minimind_lab.training.evaluate import evaluate_nll
from minimind_lab.training.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small single-epoch pretraining experiment."
    )

    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)

    return parser.parse_args()

def build_data(train_path, validation_path):
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(
        root / "assets/minimind_tokenizer",
        local_files_only=True,
    )
    max_length = 64
    collator = PretrainCollator(tokenizer, max_length=max_length)

    train_dataset = PretrainDataset(train_path, tokenizer, max_length)
    validation_dataset = PretrainDataset(
        validation_path, tokenizer, max_length
    )
    sampler = StatefulSampler(size=len(train_dataset), seed=42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(123),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(456),
    )
    return tokenizer, train_loader, validation_loader, sampler

def build_training_objects(vocab_size):
    config = ModelConfig(
        vocab_size=vocab_size,
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

def run_training(args):
    torch.manual_seed(42)

    tokenizer, train_loader, validation_loader, sampler = build_data(
        args.train_data,
        args.validation_data,
    )
    model, optimizer, scheduler = build_training_objects(len(tokenizer))

    print("train samples:", len(train_loader.dataset))
    print("validation samples:", len(validation_loader.dataset))
    print("vocab size:", len(tokenizer))
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    print("sampler position:", sampler.position)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            "output checkpoint already exists; use a new output directory"
        )

    global_step = 0
    if args.resume is not None:
        global_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            sampler=sampler,
        )

    history = train_epoch(
        model,
        optimizer,
        train_loader,
        scheduler,
        gradient_accumulation_steps=2,
        sampler=sampler,
    )
    global_step += len(history)

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        global_step=global_step,
        sampler=sampler,
    )

    validation = evaluate_nll(model, validation_loader)

    print("global step:", global_step)
    print("committed samples:", sampler.committed_position)
    print("validation:", validation)
    print("checkpoint:", checkpoint_path)

    return {
        "global_step": global_step,
        "committed_samples": sampler.committed_position,
        "updates_this_run": len(history),
        "history": history,
        "validation": validation,
        "checkpoint": str(checkpoint_path),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if any(args.output_dir.iterdir()):
        raise FileExistsError("use a new or empty output directory")

    try:
        result = run_training(args)
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        error_path = args.output_dir / "error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise

if __name__ == "__main__":
    main()


