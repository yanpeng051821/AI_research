import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_frozen_pretrain.py"
spec = importlib.util.spec_from_file_location("frozen_pretrain_under_test", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def frozen_recipe(tmp_path, monkeypatch):
    config = {
        "training": {"seed": 42},
        "model": {"vocab_size": 8, "hidden_size": 8, "intermediate_size": 16,
                  "num_layers": 1, "num_heads": 2, "kv_heads": 1},
    }
    identity = {"max_updates_resolved": 3, "runtime_test_identity": "original"}
    contract = {
        "identity": identity,
        "contract_sha256": launcher.canonical_hash(identity),
        "resolved_config": config,
    }
    config_path, saved_path = tmp_path / "training_config.json", tmp_path / "contract.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    saved_path.write_text(json.dumps(contract), encoding="utf-8")
    doc = tmp_path / "TRAINING_CONTRACT.md"
    doc.write_text("Accepted training objective and evaluation rules.\n", encoding="utf-8")
    digest, parameters = launcher.initialization_identity(config)
    artifacts = {"training_config": config_path, "resolved_contract": saved_path,
                 "training_contract": doc, "launcher": SCRIPT}
    freeze = {
        "schema_version": 1, "freeze_id": "test", "status": "frozen",
        "files": {name: {"file": str(path), "sha256": launcher.sha256_file(path)}
                  for name, path in artifacts.items()},
        "expected": {"contract_sha256": contract["contract_sha256"],
                     "initialization_sha256": digest, "model_parameters": parameters},
        "environment": {"test_environment": "same"},
    }
    path = tmp_path / "training_freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(launcher, "prepare_contract", lambda *args: (copy.deepcopy(contract), {}, {}))
    monkeypatch.setattr(launcher, "environment_identity", lambda: {"test_environment": "same"})
    return path


def test_frozen_preflight_preserves_rng_and_performs_no_training(frozen_recipe, monkeypatch):
    def unexpected_training(*args, **kwargs):
        raise AssertionError("preflight must not start training")

    monkeypatch.setattr(launcher, "run_experiment", unexpected_training)
    before = torch.get_rng_state().clone()
    report, _ = launcher.verify_frozen_recipe(frozen_recipe)
    assert report["status"] == "frozen_recipe_verified_not_trained"
    assert report["max_updates"] == 3
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("filename", ["training_config.json", "TRAINING_CONTRACT.md"])
def test_frozen_preflight_rejects_changed_artifacts(frozen_recipe, filename):
    with (frozen_recipe.parent / filename).open("a", encoding="utf-8") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        launcher.verify_frozen_recipe(frozen_recipe)


def test_frozen_preflight_rejects_changed_initialization(frozen_recipe):
    freeze = json.loads(frozen_recipe.read_text(encoding="utf-8"))
    freeze["expected"]["initialization_sha256"] = "0" * 64
    frozen_recipe.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization differs"):
        launcher.verify_frozen_recipe(frozen_recipe)


def test_frozen_preflight_rejects_runtime_drift(frozen_recipe, monkeypatch):
    original = launcher.prepare_contract

    def changed(*args):
        contract, paths, split = original(*args)
        contract["identity"]["runtime_test_identity"] = "changed"
        contract["contract_sha256"] = launcher.canonical_hash(contract["identity"])
        return contract, paths, split

    monkeypatch.setattr(launcher, "prepare_contract", changed)
    with pytest.raises(ValueError, match="runtime/configuration differs"):
        launcher.verify_frozen_recipe(frozen_recipe)


def test_frozen_preflight_rejects_device_environment_drift(frozen_recipe, monkeypatch):
    monkeypatch.setattr(launcher, "environment_identity", lambda: {"test_environment": "changed"})
    with pytest.raises(ValueError, match="device/driver/Python environment differs"):
        launcher.verify_frozen_recipe(frozen_recipe)
