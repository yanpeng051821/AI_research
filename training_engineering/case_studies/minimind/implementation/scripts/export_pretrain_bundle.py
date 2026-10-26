"""Export final pretraining weights and verify them in a fresh, offline CPU process."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_model
from transformers import AutoTokenizer

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.recovery import inspect_checkpoint
from train_frozen_pretrain import ROOT, verify_frozen_recipe
from load_exported_pretrain import (run_probe, sha256_file, state_digest, write_json)


FORMAT = "minimind-lab-native-fp32-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def save_fp32_weights(model, destination):
    if Path(destination).exists():
        raise FileExistsError("refuse to overwrite exported weights")
    for name, value in model.state_dict().items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"export requires finite, unchanged FP32 tensors: {name}")
    if model.lm_head.weight is not model.backbone.embedding.weight:
        raise ValueError("export expects tied embedding and lm_head")
    save_model(model, str(destination), metadata={"format": FORMAT, "dtype": "float32"})


def create_bundle(model, config, tokenizer_dir, code_hashes, provenance, output, root=ROOT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new bundle directory")
    output.mkdir(parents=True)
    save_fp32_weights(model, output / "model.safetensors")
    write_json(output / "config.json", config)
    (output / "tokenizer").mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "manifest.json"):
        shutil.copyfile(tokenizer_dir / name, output / "tokenizer" / name)
    for name, expected in code_hashes.items():
        if name == "src/minimind_lab/__init__.py" or name.startswith("src/minimind_lab/models/"):
            source = root / name
            if sha256_file(source) != expected:
                raise ValueError(f"frozen model source changed: {name}")
            target = output / "runtime" / Path(name).relative_to("src")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != expected:
                raise ValueError(f"runtime copy mismatch: {name}")
    shutil.copyfile(Path(__file__).with_name("load_exported_pretrain.py"), output / "load_model.py")
    versions = {key: importlib.metadata.version(key)
                for key in ("torch", "transformers", "tokenizers", "safetensors")}
    with (output / "requirements.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# Select the PyTorch build/index appropriate to your platform.\n")
        stream.write("\n".join(f"{k}=={v.split('+')[0]}" for k, v in versions.items()) + "\n")
    with (output / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# MiniMind mini pretraining: native FP32 inference bundle\n\n"
            "Local export candidate; not yet a published Hugging Face release.\n"
            "This package uses the included minimind_lab runtime, not AutoModel.\n\n"
            "From this directory, with the recorded dependencies installed:\n\n"
            "```shell\npython -I -B load_model.py --prompt \"The weather today\" --max-new-tokens 16\n```\n\n"
            "Weights are FP32 with tied embedding/lm_head. No optimizer, scheduler, sampler, "
            "or RNG recovery state is included; this is not a resumable checkpoint.\n"
            "This is a pretrained continuation model, not an instruction-tuned chat model.\n"
            "Training used length 768; this loader conservatively caps prompt plus continuation "
            "at that length. Longer-context quality has not been evaluated.\n"
            "License, redistribution review, model card, and HF download/reload verification "
            "remain pending. Hash checks are not signatures or a sandbox for Python code.\n"
        )
    manifest = {
        "schema_version": 1, "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance, "runtime_versions": versions,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_state_sha256": state_digest(model),
        "weights_dtype": "float32", "tied_weights": ["backbone.embedding.weight", "lm_head.weight"],
        "export_script_sha256": sha256_file(Path(__file__)),
        "loader_source_sha256": sha256_file(Path(__file__).with_name("load_exported_pretrain.py")),
        "files": [describe(p, output) for p in sorted(output.rglob("*")) if p.is_file()],
        "scope": "Native model-only bundle; local verification is recorded separately. Not an AutoModel adapter.",
    }
    write_json(output / "export_manifest.json", manifest)
    return manifest


def verify_in_fresh_process(output, model, tokenizer):
    reference = run_probe(model, tokenizer)
    with tempfile.TemporaryDirectory(prefix="minimind-export-check-") as temporary:
        temporary = Path(temporary)
        child_dir = temporary / "fresh_process"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        # -I ignores PYTHONPATH/cwd, and the child explicitly imports the copied runtime.
        command = [sys.executable, "-I", "-B", str((output / "load_model.py").resolve()),
                   "--verify-output-dir", str(child_dir)]
        child = subprocess.run(command, cwd=temporary, env=env, capture_output=True,
                               text=True, encoding="utf-8", timeout=300)
        if child.returncode != 0:
            raise RuntimeError(f"fresh-process loader failed: {child.stderr[-4000:]}")
        actual = load_file(child_dir / "probe.safetensors")
        state = read_json(child_dir / "probe.json")
        if not Path(state["model_runtime"]).is_relative_to((output / "runtime").resolve()):
            raise ValueError("child used project code instead of the packaged runtime")
        if state["state_sha256"] != state_digest(model) or not state["embedding_lm_head_same_parameter"]:
            raise ValueError("loaded state or sharing differs from checkpoint")
        equality = {key: torch.equal(value, actual[key]) for key, value in reference.items()}
        if not all(equality.values()):
            raise ValueError(f"export probe differs: {equality}")
        cli = subprocess.run(
            [sys.executable, "-I", "-B", str((output / "load_model.py").resolve()),
             "--max-new-tokens", "8"], cwd=temporary, env=env, capture_output=True,
            text=True, encoding="utf-8", timeout=300,
        )
        if cli.returncode != 0 or not cli.stdout.strip():
            raise RuntimeError(f"standalone continuation failed: {cli.stderr[-4000:]}")
        return {"status": "passed", "device": "cpu", "dtype": "float32",
                "cpu_threads": 1, "autocast": False, "fresh_process": True,
                "offline": True, "runtime_loaded_from_bundle": True,
                "parameter_state_exact": True, "parameter_state_sha256": state["state_sha256"],
                "embedding_lm_head_same_parameter": True, "parameters": state["parameters"],
                "standalone_cli_utf8_success": True,
                "exact_probe_equality": equality, "max_abs_logit_difference": 0.0,
                "input_shape": list(reference["input_ids"].shape),
                "logits_shape": list(reference["logits"].shape), "probe_texts": state["texts"],
                "generated_ids": reference["generated_ids"].tolist(),
                "scope": "Two short right-padded prefixes and one 8-token-budget greedy continuation; "
                         "not full validation NLL, CUDA/BF16 parity, or downstream ability evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("use new output and report directories; do not overwrite evidence")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checked, freeze = verify_frozen_recipe(args.experiment_dir / "training_recipe/training_freeze.json")
    contract = read_json(args.experiment_dir / "training_recipe/contract.json")
    final = read_json(args.experiment_dir / "results/final_run_v1/final_checkpoint_inspection.json")
    inspection = inspect_checkpoint(args.checkpoint, checked["contract_sha256"])
    if (inspection["status"] != "valid" or inspection["step"] != freeze["expected"]["max_updates"]
            or inspection["checkpoint_sha256"] != final["checkpoint_sha256"]
            or inspection["committed_samples"] != freeze["expected"]["train_records"]
            or inspection["valid_tokens_seen"] != freeze["expected"]["train_valid_targets"]
            or inspection["initialization_sha256"] != checked["initialization_sha256"]):
        raise ValueError("checkpoint differs from the accepted final training state")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("source checkpoint must already contain FP32 model tensors")
    if not torch.equal(state["backbone.embedding.weight"], state["lm_head.weight"]):
        raise ValueError("checkpoint contains conflicting tied-weight values")
    resolved = contract["resolved_config"]
    model = MinimindForCausalLM(ModelConfig(**resolved["model"]))
    model.load_state_dict(state, strict=True)
    model.eval()
    original_digest = state_digest(model)
    del state, checkpoint
    tokenizer_dir = ROOT / resolved["data"]["tokenizer_dir"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    config = {"format": FORMAT, "model": resolved["model"], "weights_dtype": "float32",
              "trained_sequence_length": resolved["data"]["max_length"],
              "special_token_ids": {k: getattr(tokenizer, k)
                                    for k in ("bos_token_id", "eos_token_id", "pad_token_id")}}
    provenance = {"checkpoint_file": args.checkpoint.name,
                  "checkpoint_sha256": inspection["checkpoint_sha256"],
                  "checkpoint_bytes": args.checkpoint.stat().st_size,
                  "contract_sha256": checked["contract_sha256"],
                  "freeze_sha256": checked["freeze_sha256"],
                  "initialization_sha256": checked["initialization_sha256"],
                  "global_step": inspection["step"], "stage": "pretrain_from_random_initialization"}
    output = args.output_dir.resolve()
    print("Source identities verified; exporting unchanged FP32 weights.", flush=True)
    manifest = create_bundle(model, config, tokenizer_dir, contract["identity"]["code_sha256"],
                             provenance, output)
    print("Bundle written; checking offline loading in a fresh process.", flush=True)
    verification = verify_in_fresh_process(output, model, tokenizer)
    if state_digest(model) != original_digest or sha256_file(args.checkpoint) != inspection["checkpoint_sha256"]:
        raise ValueError("source model or checkpoint changed during export")
    args.report_dir.mkdir(parents=True)
    shutil.copyfile(output / "export_manifest.json", args.report_dir / "export_manifest.json")
    write_json(args.report_dir / "verification.json", {
        **verification, "bundle_location": str(output),
        "export_manifest_sha256": sha256_file(output / "export_manifest.json"),
        "model_safetensors_sha256": sha256_file(output / "model.safetensors"),
        "weights_bytes": (output / "model.safetensors").stat().st_size,
        "bundle_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "source_checkpoint_unchanged": True, "hf_upload": "not_performed",
        "independent_resumable_checkpoint_backup": "not_verified",
    })
    print(json.dumps({"status": "exported_and_locally_verified", "bundle": str(output),
                      "report": str(args.report_dir), "parameters": manifest["parameters"],
                      "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
