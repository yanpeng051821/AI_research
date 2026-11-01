"""Check SFT evaluation readiness, or run an explicitly authorized paired evaluation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from freeze_sft_data_recipe import check as check_data_freeze, write_new
from inspect_sft_eval_draft import render_probe, sha
from load_exported_pretrain import state_digest, verify_bundle
from minimind_lab.data.sft import SFTDataset, SFTCollator
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.models.generation import greedy_generate
from minimind_lab.training.evaluate import evaluate_nll


ROOT = Path(__file__).resolve().parents[1]


class CheckedForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise ValueError("invalid generation logits shape")
        if not torch.isfinite(logits[:, -1, :]).all():
            raise FloatingPointError("non-finite next-token logits")
        return logits


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_case(model, tokenizer, case, config, device):
    text, ids, _ = render_probe(case, config, tokenizer)
    generation = config["generation"]
    if not ids or len(ids) + generation["max_new_tokens"] > generation["context_budget"]:
        raise ValueError("generation exceeds agreed context budget")
    was_training = model.training
    model.eval()
    try:
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = greedy_generate(CheckedForward(model), torch.tensor([ids], device=device),
                                     generation["max_new_tokens"], tokenizer.eos_token_id)
        synchronize(device)
        elapsed = time.perf_counter() - start
        new_ids = result[0, len(ids):].tolist()
    finally:
        model.train(was_training)
    return {"id": case["id"], "category": case["category"], "tier": case["tier"],
            "rendered_prompt": text, "input_ids": ids, "generated_ids": new_ids,
            "raw_text": tokenizer.decode(new_ids, skip_special_tokens=False),
            "display_text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "input_tokens": len(ids), "generated_tokens": len(new_ids),
            "stop_reason": "eos" if new_ids and new_ids[-1] == tokenizer.eos_token_id else "length_limit",
            "elapsed_seconds": elapsed, "scoring_status": "pending_manual_review"}


def load_weights(bundle, contract, device):
    manifest = verify_bundle(bundle)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"].startswith("runtime/minimind_lab/models/") and entry["path"].endswith(".py"):
            path = ROOT / "src" / entry["path"].removeprefix("runtime/")
            if sha(path) != entry["sha256"]:
                raise ValueError(f"current model code differs from exported architecture: {path}")
    for name, expected in contract["pipeline_identity"]["tokenizer_files_sha256"].items():
        if sha(bundle / "tokenizer" / name) != expected:
            raise ValueError("exported tokenizer differs from the SFT data tokenizer")
    if config["weights_dtype"] != "float32" or config["model"]["vocab_size"] != contract["encoding"]["vocab_size"]:
        raise ValueError("incompatible model configuration")
    model = MinimindForCausalLM(ModelConfig(**config["model"])).float()
    load_model(model, bundle / "model.safetensors", strict=True, device="cpu")
    digest = state_digest(model)
    if digest != manifest["model_state_sha256"] or model.lm_head.weight is not model.backbone.embedding.weight:
        raise ValueError("loaded state identity or tied weights mismatch")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/minimind_tokenizer", local_files_only=True)
    special = {name: getattr(tokenizer, name) for name in config["special_token_ids"]}
    if special != config["special_token_ids"]:
        raise ValueError("special-token identity mismatch")
    return model.to(device).eval(), tokenizer, manifest, config


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    generation = config["generation"]
    if (generation["strategy"] != "greedy" or generation["do_sample"] is not False
            or generation["batch_size"] != 1 or generation["max_new_tokens"] <= 0
            or generation["model_eval_mode"] is not True):
        raise ValueError("this entry requires positive-budget, batch-one greedy eval")
    if args.run and (config["status"] != "frozen"
                     or config["acceptance"]["evaluation_run_authorized"] is not True
                     or generation["device_and_precision"] != f"{args.device}_fp32"):
        raise ValueError("formal evaluation requires an approved frozen configuration and matching device/FP32")
    if args.output_dir.exists():
        raise FileExistsError("use a new evaluation output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
    inspection = json.loads(args.inspection.read_text(encoding="utf-8"))
    for path in (args.config, args.cases, Path(__file__).with_name("inspect_sft_eval_draft.py")):
        if inspection["draft_inputs_sha256"].get(str(path.resolve())) != sha(path):
            raise ValueError("inspection no longer matches evaluation inputs")
    if any(v["matched_user_messages"] for v in inspection["normalized_user_text_overlap"].values()):
        raise ValueError("review exact overlaps before using these probes")
    cases = [json.loads(raw) for raw in args.cases.read_text(encoding="utf-8").splitlines()]
    contract_path = (args.config.parent / config["data_contract"]).resolve()
    if sha(contract_path) != config["data_contract_sha256"]:
        raise ValueError("data contract identity mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model_files = list((ROOT / "src/minimind_lab/models").rglob("*.py"))
    dependencies = [ROOT / "src/minimind_lab/data/sft.py", ROOT / "src/minimind_lab/data/sft_protocol.py",
                    ROOT / "src/minimind_lab/data/validator.py", ROOT / "src/minimind_lab/models/loss.py",
                    ROOT / "src/minimind_lab/training/evaluate.py", ROOT / "pyproject.toml", ROOT / "uv.lock",
                    Path(__file__), Path(__file__).with_name("inspect_sft_eval_draft.py"),
                    Path(__file__).with_name("load_exported_pretrain.py"),
                    args.config, args.cases, args.inspection, args.bundle / "export_manifest.json"]
    identities = {str(p.resolve()): sha(p) for p in [*model_files, *dependencies]}
    args.output_dir.mkdir(parents=True)
    manifest = {"status": "running", "mode": "run" if args.run else "check_only",
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_started": False,
                "inputs_and_runtime_sha256": identities}
    manifest_path = args.output_dir / "run_manifest.json"

    def update_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    update_manifest()
    try:
        data_check = check_data_freeze(contract_path.parent / "data_freeze.json")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        model, tokenizer, exported, model_config = load_weights(args.bundle, contract, device)
        if (generation["eos_token_id"] != tokenizer.eos_token_id
                or generation["context_budget"] > model_config["trained_sequence_length"]):
            raise ValueError("generation settings differ from model/tokenizer limits")
        for case in cases:
            _, ids, _ = render_probe(case, config, tokenizer)
            if len(ids) + generation["max_new_tokens"] > generation["context_budget"]:
                raise ValueError(f"over-budget probe: {case['id']}")
        report = {"status": "ready_check_passed_not_evaluated", "device": str(device),
                  "precision": "fp32_no_autocast_tf32_disabled", "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "model_state_sha256": exported["model_state_sha256"],
                  "weight_file_sha256": sha(args.bundle / "model.safetensors"),
                  "pretrain_source": exported["source"], "data_check": data_check,
                  "probes_checked": len(cases), "generation_performed": False,
                  "nll_computed": False, "model_eval_mode": not model.training,
                  "tied_embedding_lm_head": model.lm_head.weight is model.backbone.embedding.weight,
                  "training_started": False}
        if args.run:
            with (args.output_dir / "generations.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for case in cases:
                    row = generate_case(model, tokenizer, case, config, device)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            dataset = SFTDataset(contract["artifacts"]["validation"]["path"], tokenizer)
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                collate_fn=SFTCollator(tokenizer, contract["encoding"]["max_length"]),
                                generator=torch.Generator().manual_seed(42))
            start = time.perf_counter()
            synchronize(device)
            nll = evaluate_nll(model, loader, device=device)
            synchronize(device)
            if nll["valid_tokens"] != contract["split_statistics"]["validation"]["total_targets_after_shift"]:
                raise ValueError("validation target count differs from frozen data")
            write_new(args.output_dir / "validation.json", {**nll, "records": len(dataset), "batch_size": 4,
                                                            "elapsed_seconds": time.perf_counter() - start})
            if state_digest(model) != exported["model_state_sha256"]:
                raise ValueError("model state changed during evaluation")
            report.update(status="evaluation_completed_scoring_pending", generation_performed=True,
                          nll_computed=True, validation=nll)
        if any(sha(Path(path)) != value for path, value in identities.items()):
            raise ValueError("evaluation inputs or runtime changed during execution")
        write_new(args.output_dir / "result.json", report)
        manifest.update(status="completed", result_status=report["status"],
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        update_manifest()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        update_manifest()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    run(parser.parse_args())
