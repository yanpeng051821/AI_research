"""Export boundaries: tied weights, isolated loading, and altered/incomplete packages."""

from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import AutoTokenizer

from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def export_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (importlib.import_module("export_pretrain_bundle"),
            importlib.import_module("load_exported_pretrain"))


@pytest.fixture
def tiny_bundle(tmp_path, export_tools):
    exporter, loader = export_tools
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=6400, hidden_size=16, intermediate_size=32,
                         num_layers=1, num_heads=4, kv_heads=2, qk_norm=True)
    model = MinimindForCausalLM(config).eval()
    tokenizer_dir = ROOT / "assets/minimind_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    package_config = {"format": exporter.FORMAT, "model": asdict(config),
                      "weights_dtype": "float32", "trained_sequence_length": 768,
                      "special_token_ids": {key: getattr(tokenizer, key)
                                            for key in ("bos_token_id", "eos_token_id", "pad_token_id")}}
    paths = [ROOT / "src/minimind_lab/__init__.py",
             *sorted((ROOT / "src/minimind_lab/models").glob("*.py"))]
    hashes = {p.relative_to(ROOT).as_posix(): loader.sha256_file(p) for p in paths}
    output = tmp_path / "bundle"
    exporter.create_bundle(model, package_config, tokenizer_dir, hashes, {"test_fixture": True}, output)
    return output, model, tokenizer


def test_native_bundle_matches_in_fresh_offline_process(tiny_bundle, export_tools):
    exporter, _ = export_tools
    output, model, tokenizer = tiny_bundle
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        report = exporter.verify_in_fresh_process(output, model, tokenizer)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
    assert report["parameter_state_exact"]
    assert report["embedding_lm_head_same_parameter"]
    assert report["runtime_loaded_from_bundle"]
    assert all(report["exact_probe_equality"].values())
    assert report["max_abs_logit_difference"] == 0.0
    assert not (output / "runtime/minimind_lab/training").exists()


def test_bundle_rejects_config_tampering(tiny_bundle, export_tools):
    _, loader = export_tools
    output, _, _ = tiny_bundle
    (output / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch: config.json"):
        loader.verify_bundle(output)


def test_standalone_cli_emits_utf8_in_isolated_mode(tiny_bundle, tmp_path):
    output, _, _ = tiny_bundle
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(output / "load_model.py"), "--max-new-tokens", "8"],
        cwd=tmp_path, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stdout.decode("utf-8").strip()


def test_incomplete_bundle_is_not_loadable(tmp_path, export_tools):
    _, loader = export_tools
    with pytest.raises(FileNotFoundError):
        loader.load_bundle(tmp_path)


def test_bundle_rejects_path_escape(tmp_path, export_tools):
    _, loader = export_tools
    manifest = {"format": "minimind-lab-native-fp32-v1",
                "files": [{"path": "../outside", "bytes": 0, "sha256": "unused"}]}
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or duplicate"):
        loader.verify_bundle(tmp_path)


@pytest.mark.parametrize("change", ["dtype", "nonfinite", "untied"])
def test_export_rejects_inappropriate_model_state(tmp_path, export_tools, change):
    exporter, _ = export_tools
    model = MinimindForCausalLM(ModelConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                                          num_layers=1, num_heads=2, kv_heads=1))
    if change == "dtype":
        model.double()
    elif change == "nonfinite":
        with torch.no_grad():
            model.backbone.embedding.weight[0, 0] = float("nan")
    else:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    with pytest.raises(ValueError):
        exporter.save_fp32_weights(model, tmp_path / "weights.safetensors")


def test_export_refuses_existing_bundle(tiny_bundle, export_tools):
    exporter, _ = export_tools
    output, model, _ = tiny_bundle
    with pytest.raises(FileExistsError):
        exporter.create_bundle(model, {}, ROOT / "assets/minimind_tokenizer", {}, {}, output)
