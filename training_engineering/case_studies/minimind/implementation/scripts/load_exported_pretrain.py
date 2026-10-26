"""Load a local native MiniMind bundle; this is not a Transformers AutoModel adapter."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_model, save_file
from transformers import AutoTokenizer


PROBE_TEXTS = ["春天的风吹过田野，", "人工智能"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "export_manifest.json")
    if manifest["format"] != "minimind-lab-native-fp32-v1":
        raise ValueError("unsupported bundle format")
    names = set()
    for item in manifest["files"]:
        name = item["path"]
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or name in names:
            raise ValueError("invalid or duplicate bundle path")
        names.add(name)
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"bundle file identity mismatch: {name}")
    required = {"model.safetensors", "config.json", "load_model.py",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                "runtime/minimind_lab/__init__.py", "runtime/minimind_lab/models/model.py"}
    if not required <= names:
        raise ValueError("bundle manifest lacks required files")
    # Imported Python must be covered too, not only the tensor file.
    if any(p.relative_to(directory).as_posix() not in names
           for p in (directory / "runtime").rglob("*.py")):
        raise ValueError("unlisted runtime source")
    return manifest


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = verify_bundle(directory)
    runtime = directory / "runtime"
    for name, module in list(sys.modules.items()):
        if name == "minimind_lab" or name.startswith("minimind_lab."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(runtime):
                raise RuntimeError("another minimind_lab is imported; load this bundle in a fresh process")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime))
    config_type = importlib.import_module("minimind_lab.models.config").ModelConfig
    model_type = importlib.import_module("minimind_lab.models.model").MinimindForCausalLM
    config = read_json(directory / "config.json")
    if config["format"] != manifest["format"] or config["weights_dtype"] != "float32":
        raise ValueError("configuration does not describe this FP32 bundle")
    model = model_type(config_type(**config["model"])).float().cpu()
    load_model(model, directory / "model.safetensors", strict=True, device="cpu")
    if model.lm_head.weight is not model.backbone.embedding.weight:
        raise ValueError("embedding/lm_head sharing was lost")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    ids = {key: getattr(tokenizer, key) for key in config["special_token_ids"]}
    if ids != config["special_token_ids"] or len(tokenizer) != config["model"]["vocab_size"]:
        raise ValueError("tokenizer is inconsistent with the model configuration")
    return model, tokenizer, config


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def probe_inputs(tokenizer):
    rows = [[tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]
            for text in PROBE_TEXTS]
    width = max(map(len, rows))
    ids = torch.tensor([r + [tokenizer.pad_token_id] * (width - len(r)) for r in rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
    return ids, mask


def run_probe(model, tokenizer):
    from minimind_lab.models.generation import greedy_generate

    model.eval()
    ids, mask = probe_inputs(tokenizer)
    with torch.inference_mode():
        logits = model(ids, attention_mask=mask)
        generated = greedy_generate(model, ids[:1], max_new_tokens=8,
                                    eos_token_id=tokenizer.eos_token_id)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite probe logits")
    return {"input_ids": ids, "attention_mask": mask, "logits": logits,
            "generated_ids": generated}


def main():
    # Isolated mode ignores PYTHONIOENCODING; configure Windows stdout explicitly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prompt", default=PROBE_TEXTS[0])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--verify-output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, tokenizer, config = load_bundle(args.bundle_dir)
    if args.verify_output_dir is not None:
        args.verify_output_dir.mkdir(parents=True, exist_ok=False)
        tensors = run_probe(model, tokenizer)
        save_file(tensors, args.verify_output_dir / "probe.safetensors")
        write_json(args.verify_output_dir / "probe.json", {
            "state_sha256": state_digest(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "embedding_lm_head_same_parameter": model.lm_head.weight is model.backbone.embedding.weight,
            "model_runtime": str(Path(importlib.import_module("minimind_lab.models.model").__file__).resolve()),
            "texts": PROBE_TEXTS, "torch": torch.__version__,
        })
        return
    if args.max_new_tokens < 0:
        parser.error("max-new-tokens must be nonnegative")
    ids = [tokenizer.bos_token_id, *tokenizer.encode(args.prompt, add_special_tokens=False)]
    if len(ids) + args.max_new_tokens > config["trained_sequence_length"]:
        parser.error("this entry conservatively limits prompt + generation to the trained length")
    from minimind_lab.models.generation import greedy_generate

    generated = greedy_generate(model, torch.tensor([ids]), args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(generated[0, len(ids):].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
