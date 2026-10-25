import copy
import importlib.metadata
import json
from pathlib import Path

import pytest
import torch

from minimind_lab.training import experiment
from minimind_lab.training.step import train_update


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment_config(tmp_path):
    split = tmp_path / "split"
    split.mkdir()
    files = {}
    for name, filename, texts, start in [
        ("train", "train.jsonl", ["alpha", "beta", "gamma", "delta", "epsilon"], 1),
        ("validation", "validation.jsonl", ["zeta", "eta"], 6),
        ("excluded", "exclusions.jsonl", [], 8),
    ]:
        path = split / filename
        path.write_text("".join(json.dumps({"source_line": i, "text": text}) + "\n" for i, text in enumerate(texts, start)), encoding="utf-8")
        files[name] = {"file": filename, "sha256": experiment.sha256_file(path)}
    manifest = split / "manifest.json"
    experiment.write_json(manifest, {"status": "text_split_complete_encoding_not_checked", "files": files,
                                     "counts": {"train": 5, "validation": 2, "excluded": 0}})
    tokenizer = ROOT / "assets/minimind_tokenizer"
    audit = split / "encoding_audit.json"
    experiment.write_json(audit, {
        "status": "passed", "max_length": 16,
        "split_manifest_sha256": experiment.sha256_file(manifest),
        "split_files_sha256": {name: info["sha256"] for name, info in files.items()},
        "tokenizer_files_sha256": {path.name: experiment.sha256_file(path) for path in tokenizer.glob("*.json")},
        "code_sha256": {"dataset_and_collator": experiment.sha256_file(ROOT / "src/minimind_lab/data/pretrain.py")},
        "special_token_ids": {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
        "runtime": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
    })
    config = tmp_path / "config.json"
    experiment.write_json(config, {
        "schema_version": 1, "purpose": "engineering_probe",
        "data": {"split_manifest": str(manifest), "split_manifest_sha256": experiment.sha256_file(manifest),
                 "encoding_report": str(audit), "encoding_report_sha256": experiment.sha256_file(audit),
                 "tokenizer_dir": str(tokenizer), "max_length": 16},
        "model": {"vocab_size": 6400, "hidden_size": 16, "intermediate_size": 32,
                  "num_layers": 1, "num_heads": 4, "kv_heads": 2, "qk_norm": True},
        "training": {"device": "cpu", "precision": "fp32", "micro_batch_size": 1,
                     "gradient_accumulation_steps": 2, "max_updates": 3, "checkpoint_interval": 2},
        "evaluation": {"batch_size": 2, "sample_limit": 2, "interval": 2},
    })
    return config


def test_pretrain_probe_continuous_and_resumed_updates_match(experiment_config, tmp_path):
    continuous = tmp_path / "continuous"
    paused = tmp_path / "paused"
    resumed = tmp_path / "resumed"
    a = experiment.run_experiment(experiment_config, continuous, ROOT)
    b = experiment.run_experiment(experiment_config, paused, ROOT, stop_after_updates=1)
    c = experiment.run_experiment(experiment_config, resumed, ROOT, resume=paused / "checkpoint-1.pt")
    assert b["status"] == "paused" and b["committed_samples"] == 2
    assert a["status"] == c["status"] == "completed"
    assert a["global_step"] == c["global_step"] == 3
    assert a["committed_samples"] == c["committed_samples"] == 5
    assert a["valid_tokens_seen"] == c["valid_tokens_seen"]
    assert a["final_validation"] == c["final_validation"]
    left = torch.load(continuous / "checkpoint-3.pt", weights_only=True)
    right = torch.load(resumed / "checkpoint-3.pt", weights_only=True)
    for name in left["model"]:
        torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
    for key, state in left["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, right["optimizer"]["state"][key][name], rtol=0, atol=0)
    assert left["scheduler"] == right["scheduler"]
    assert left["sampler_state"] == right["sampler_state"]
    torch.testing.assert_close(left["torch_rng_state"], right["torch_rng_state"], rtol=0, atol=0)


def test_prepare_only_verifies_without_training(experiment_config, tmp_path):
    output = tmp_path / "prepared"
    result = experiment.run_experiment(experiment_config, output, ROOT, prepare_only=True)
    assert result["status"] == "prepared_not_trained"
    assert (output / "contract.json").exists()
    assert not list(output.glob("*.pt"))


def test_checkpoint_can_be_stored_outside_log_directory(experiment_config, tmp_path):
    output = tmp_path / "logs"
    checkpoint_dir = tmp_path / "checkpoints"
    result = experiment.run_experiment(
        experiment_config,
        output,
        ROOT,
        checkpoint_dir=checkpoint_dir,
    )
    assert result["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert result["last_checkpoint"] == "checkpoint-3.pt"
    assert not list(output.glob("*.pt"))
    assert (checkpoint_dir / "checkpoint-3.pt").exists()
    assert (checkpoint_dir / "checkpoint-3.pt.receipt.json").exists()


def test_checkpoint_refuses_to_overwrite_existing_external_artifact(experiment_config, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    existing = checkpoint_dir / "checkpoint-1.pt"
    existing.write_bytes(b"existing checkpoint")

    with pytest.raises(FileExistsError, match="checkpoint destination already exists"):
        experiment.run_experiment(
            experiment_config,
            tmp_path / "logs",
            ROOT,
            stop_after_updates=1,
            checkpoint_dir=checkpoint_dir,
        )

    assert existing.read_bytes() == b"existing checkpoint"
    state = json.loads((tmp_path / "logs/run_manifest.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


def test_resume_rejects_changed_training_semantics(experiment_config, tmp_path):
    output = tmp_path / "paused"
    experiment.run_experiment(experiment_config, output, ROOT, stop_after_updates=1)
    config = json.loads(experiment_config.read_text(encoding="utf-8"))
    config["training"]["learning_rate"] = 1e-5
    experiment.write_json(experiment_config, config)
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="identity mismatch"):
        experiment.run_experiment(experiment_config, rejected, ROOT, resume=output / "checkpoint-1.pt")
    assert json.loads((rejected / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not list(rejected.glob("*.pt"))


def test_prepare_rejects_data_changed_at_same_path(experiment_config, tmp_path):
    path = tmp_path / "split/train.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="train data identity mismatch"):
        experiment.prepare_contract(experiment_config, ROOT)


def test_failed_save_records_failure_after_committed_update(experiment_config, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(experiment, "save_checkpoint", fail)
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="disk failure"):
        experiment.run_experiment(experiment_config, output, ROOT, stop_after_updates=1)
    state = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["global_step"] == 1
    assert "last_checkpoint" not in state
    assert not (output / "result.json").exists()


@pytest.mark.parametrize("field,value", [("micro_batch_size", 0), ("gradient_accumulation_steps", -1), ("max_updates", True), ("precision", "fp16")])
def test_invalid_training_settings_are_rejected(experiment_config, field, value):
    config = json.loads(experiment_config.read_text(encoding="utf-8"))
    config["training"][field] = value
    experiment.write_json(experiment_config, config)
    with pytest.raises(ValueError):
        experiment.resolve_config(experiment_config)


def test_update_scheduler_uses_original_plan_after_resume():
    assert experiment.lr_multiplier(0, 9, 0, 0.1) == pytest.approx(1.0)
    assert experiment.lr_multiplier(4, 9, 0, 0.1) == pytest.approx(0.55)
    assert experiment.lr_multiplier(8, 9, 0, 0.1) == pytest.approx(0.1)


def test_warmup_and_cosine_use_every_planned_update():
    multipliers = [experiment.lr_multiplier(step, 6, 2, 0.1) for step in range(6)]
    assert multipliers[:2] == pytest.approx([0.5, 1.0])
    assert multipliers[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(multipliers[1:], multipliers[2:]))
