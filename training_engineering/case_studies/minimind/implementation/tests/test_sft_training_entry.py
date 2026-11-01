from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import AutoTokenizer

from minimind_lab.data.sampler import StatefulSampler
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CANDIDATE = ROOT.parent / "experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json"
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("sft_entry", SCRIPTS / "train_sft.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


def test_candidate_config_preserves_agreed_settings_and_requires_explicit_mode(tmp_path):
    cfg = entry.read_config(CANDIDATE)
    assert cfg["training"]["learning_rate"] == 1e-5
    assert cfg["training"]["micro_batch_size"] * cfg["training"]["gradient_accumulation_steps"] == 16
    assert cfg["operations"]["checkpoint_interval"] is None
    cfg["training"]["learning_rate"] = float("nan")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        entry.read_config(bad)
    result = subprocess.run([sys.executable, str(SCRIPTS / "train_sft.py"),
                             "--config", str(bad), "--output-dir", str(tmp_path / "no_mode")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "no_mode").exists()


@pytest.fixture
def tiny_probe(tmp_path, monkeypatch):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/minimind_tokenizer", local_files_only=True)
    row = {"conversations": [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "hello"}]}
    data = {"encoding": {"max_length": 64}, "artifacts": {}}
    for name, count in (("train", 9), ("validation", 2)):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for _ in range(count)), encoding="utf-8")
        data["artifacts"][name] = {"path": str(path)}
    cfg = entry.read_config(CANDIDATE)
    cfg["training"].update(device="cpu", precision="fp32", micro_batch_size=1,
                           gradient_accumulation_steps=2)

    def load_tiny(*args):
        model = MinimindForCausalLM(ModelConfig(vocab_size=6400, hidden_size=8,
                                     intermediate_size=16, num_layers=1, num_heads=2, kv_heads=1))
        return model, tokenizer, {}, {}

    monkeypatch.setattr(entry, "load_weights", load_tiny)
    contract = {"contract_sha256": "tiny-engineering-test",
                "identity": {"candidate": cfg, "train_records": 9, "validation_records": 2,
                             "planned_updates": 5, "effective_batch_records": 2,
                             "sample_order_sha256": entry.canonical_hash(StatefulSampler(9, 42).indices)}}
    return contract, data


def test_probe_continuous_and_resume_match_without_resetting_scheduler(tiny_probe, tmp_path):
    contract, data = tiny_probe
    folders = [tmp_path / name for name in ("continuous", "paused", "resumed")]
    for p in folders:
        p.mkdir()
    before = deepcopy(contract)
    a = entry.run_probe(contract, data, folders[0], stop_after=4, validation_records=2)
    b = entry.run_probe(contract, data, folders[1], stop_after=2, validation_records=2)
    c = entry.run_probe(contract, data, folders[2], stop_after=4, validation_records=2,
                        resume=folders[1] / "checkpoint-2.pt")
    assert b["committed_samples"] == 4
    assert a["committed_samples"] == c["committed_samples"] == 8
    assert a["valid_tokens_seen"] == c["valid_tokens_seen"]
    assert a["final_validation"] == c["final_validation"]
    left = torch.load(folders[0] / "checkpoint-4.pt", weights_only=True)
    right = torch.load(folders[2] / "checkpoint-4.pt", weights_only=True)
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
    for key, state in left["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, right["optimizer"]["state"][key][name], rtol=0, atol=0)
    assert left["scheduler"] == right["scheduler"]
    assert left["sampler_state"] == right["sampler_state"]
    torch.testing.assert_close(left["torch_rng_state"], right["torch_rng_state"], rtol=0, atol=0)
    assert contract == before


def test_candidate_cannot_run_full_epoch_or_oversized_validation(tiny_probe, tmp_path):
    contract, data = tiny_probe
    with pytest.raises(ValueError, match="before the full"):
        entry.run_probe(contract, data, tmp_path, stop_after=5, validation_records=2)
    with pytest.raises(ValueError, match="exceeds"):
        entry.run_probe(contract, data, tmp_path, stop_after=2, validation_records=3)


def test_prepare_only_never_loads_model_or_runs_updates(tiny_probe, tmp_path, monkeypatch):
    contract, data = tiny_probe
    monkeypatch.setattr(entry, "prepare", lambda path: (contract, data))

    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not load model or train")

    monkeypatch.setattr(entry, "run_probe", forbidden)
    monkeypatch.setattr(entry, "load_weights", forbidden)
    output = tmp_path / "prepare"
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--config", "unused.json",
                                     "--output-dir", str(output), "--prepare-only"])
    entry.main()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "prepared_not_trained"
    assert result["training_started"] is False
    assert not list(output.glob("*.pt"))


def test_prepare_failure_records_failed_without_claiming_training_started(tmp_path, monkeypatch):
    def rejected(path):
        raise ValueError("data identity mismatch")

    monkeypatch.setattr(entry, "prepare", rejected)
    output = tmp_path / "rejected"
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--config", "unused.json",
                                     "--output-dir", str(output), "--prepare-only"])
    with pytest.raises(ValueError, match="identity"):
        entry.main()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["training_started"] is False
    assert "data identity mismatch" in (output / "error.txt").read_text(encoding="utf-8")
