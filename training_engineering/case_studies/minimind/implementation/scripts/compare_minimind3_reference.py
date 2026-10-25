"""Compare the hand-written model with a pinned official CPU FP32 reference."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

import torch
import transformers

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.loss import causal_lm_loss
from minimind_lab.models.model import MinimindForCausalLM


ROOT = Path(__file__).resolve().parents[1]
REVISION = "f659b55761b754d306bd140573493a6543cafd7f"
SOURCE_SHA256 = "548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/jingyaogong/minimind/{REVISION}"
    "/model/model_minimind.py"
)
ATOL = 1e-6
RTOL = 1e-4


def load_reference():
    path = ROOT / "data" / "reference" / REVISION / "model_minimind.py"
    if not path.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("downloaded reference source hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("cached reference source hash mismatch")
    spec = importlib.util.spec_from_file_location("minimind_official_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def local_parameter_name(name):
    roots = {
        "model.embed_tokens.weight": "backbone.embedding.weight",
        "model.norm.weight": "backbone.final_norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    if name in roots:
        return roots[name]
    if not name.startswith("model.layers."):
        raise ValueError(f"unmapped official parameter: {name}")
    mapped = name.replace("model.layers.", "backbone.blocks.")
    for old, new in (
        (".self_attn.", ".attention."),
        (".input_layernorm.", ".attention_norm."),
        (".post_attention_layernorm.", ".feed_forward.rms_norm."),
        (".mlp.", ".feed_forward.swiglu."),
    ):
        mapped = mapped.replace(old, new)
    return mapped


def compare(profile):
    official = load_reference()
    torch.manual_seed(42)
    hidden, intermediate, layers, vocab, heads, kv_heads = (
        (16, 32, 2, 32, 4, 2) if profile == "tiny"
        else (768, 2432, 8, 6400, 8, 4)
    )
    head_dim = hidden // heads
    reference = official.MiniMindForCausalLM(official.MiniMindConfig(
        hidden_size=hidden, num_hidden_layers=layers, intermediate_size=intermediate,
        vocab_size=vocab, num_attention_heads=heads, num_key_value_heads=kv_heads,
        head_dim=head_dim, rms_norm_eps=1e-6, max_position_embeddings=32,
        flash_attn=False, dropout=0.0, use_moe=False,
    ))
    actual = MinimindForCausalLM(ModelConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
        num_layers=layers, num_heads=heads, kv_heads=kv_heads, rms_norm_eps=1e-6,
        qk_norm=True,
    ))
    # Non-unit scales catch missing Q/K normalization weights in the local path.
    with torch.no_grad():
        for layer in reference.model.layers:
            layer.self_attn.q_norm.weight.copy_(
                torch.tensor([0.8, 1.1, 0.9, 1.2]).repeat(head_dim // 4),
            )
            layer.self_attn.k_norm.weight.copy_(
                torch.tensor([1.2, 0.7, 1.1, 0.9]).repeat(head_dim // 4),
            )
    actual.load_state_dict({
        local_parameter_name(name): value
        for name, value in reference.state_dict().items()
    }, strict=True)
    ref_parameters = dict(reference.named_parameters())
    local_parameters = dict(actual.named_parameters())
    if {local_parameter_name(name) for name in ref_parameters} != set(local_parameters):
        raise ValueError("reference and local parameter sets differ")
    if not (actual.lm_head.weight is actual.backbone.embedding.weight
            and reference.lm_head.weight is reference.model.embed_tokens.weight):
        raise ValueError("both models must tie embedding and LM head weights")
    actual.eval()
    reference.eval()
    results = []
    for case, ids, mask in (
        ("no_padding", [[1, 3, 4, 2], [1, 5, 6, 2]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
        ("right_padding", [[1, 3, 4, 2], [1, 5, 2, 0]], [[1, 1, 1, 1], [1, 1, 1, 0]]),
    ):
        actual.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        input_ids, attention_mask = torch.tensor(ids), torch.tensor(mask)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        expected = reference(input_ids, attention_mask=attention_mask, labels=labels)
        logits = actual(input_ids, attention_mask=attention_mask)
        loss = causal_lm_loss(logits, labels)
        torch.testing.assert_close(logits, expected.logits, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(loss, expected.loss, atol=ATOL, rtol=RTOL)
        loss.backward()
        expected.loss.backward()
        max_gradient_error = 0.0
        for name, parameter in ref_parameters.items():
            local = local_parameters[local_parameter_name(name)]
            if parameter.grad is None or local.grad is None:
                raise ValueError(f"missing gradient: {name}")
            torch.testing.assert_close(
                local.grad, parameter.grad, atol=ATOL, rtol=RTOL, msg=name,
            )
            max_gradient_error = max(
                max_gradient_error, (local.grad - parameter.grad).abs().max().item(),
            )
        results.append({
            "case": case,
            "valid_targets": int((labels[:, 1:] != -100).sum()),
            "logits_max_abs_error": (logits - expected.logits).abs().max().item(),
            "local_loss": loss.item(),
            "official_loss": expected.loss.item(),
            "loss_abs_error": abs(loss.item() - expected.loss.item()),
            "gradient_max_abs_error": max_gradient_error,
            "status": "passed",
        })
    model_directory = ROOT / "src" / "minimind_lab" / "models"
    return {
        "reference_revision": REVISION,
        "reference_source_sha256": SOURCE_SHA256,
        "reference_source_url": SOURCE_URL,
        "local_model_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(model_directory.glob("*.py"))
        },
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu", "dtype": "float32", "seed": 42,
        "profile": profile,
        "model_dimensions": {
            "layers": layers, "hidden_size": hidden, "intermediate_size": intermediate,
            "vocab_size": vocab, "query_heads": heads, "kv_heads": kv_heads,
        },
        "local_unique_parameters": sum(p.numel() for p in actual.parameters()),
        "compared_parameter_tensors": len(ref_parameters),
        "flash_attention": False,
        "atol": ATOL, "rtol": RTOL,
        "cases": results,
        "scope": "Forward, loss and gradients; no optimizer update or training.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/minimind3_alignment")
    parser.add_argument("--profile", choices=("tiny", "dense"), default="tiny")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use a new or empty output directory")
    report = compare(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
