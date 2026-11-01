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
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import sft_formal_runtime as runtime
    import freeze_sft_training as freezing
finally:
    sys.path.pop(0)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/minimind_tokenizer", local_files_only=True)
    row = {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    data = {"encoding": {"max_length": 64}, "artifacts": {}, "split_statistics": {"train": {"total_targets_after_shift": 0}}}
    for name, size in (("train", 5), ("validation", 2)):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(row) + "\n" for _ in range(size)), encoding="utf-8")
        data["artifacts"][name] = {"path": str(p)}
    from minimind_lab.data.sft import encode_sft_conversation
    encoded = encode_sft_conversation(row["conversations"], tokenizer)
    data["split_statistics"]["train"]["total_targets_after_shift"] = sum(v != -100 for v in encoded["labels"][1:]) * 5
    cfg = json.loads((ROOT.parent / "experiments/sft_mini_v1/training_recipe/sft_mini_candidate.json").read_text())
    cfg["training"].update(device="cpu", precision="fp32", micro_batch_size=1, gradient_accumulation_steps=2)
    core = {"candidate": cfg, "planned_updates": 3, "train_records": 5, "validation_records": 2,
            "effective_batch_records": 2, "sample_order_sha256": runtime.canonical_hash(StatefulSampler(5, 42).indices)}
    operations = {"checkpoint_dir": str(tmp_path / "disk"), "checkpoint_interval": 2,
                  "validation_interval": 1, "validation_batch_size": 2, "wall_clock_budget_hours": 14,
                  "minimum_checkpoint_free_gib": 0.001}
    contract = {"identity": {"core": core, "operations": operations}, "contract_sha256": "tiny-formal-test"}

    def load(*args):
        model = MinimindForCausalLM(ModelConfig(vocab_size=6400, hidden_size=8, intermediate_size=16,
                                               num_layers=1, num_heads=2, kv_heads=1))
        return model, tokenizer, {}, {}

    monkeypatch.setattr(runtime, "load_weights", load)
    return contract, data


def test_full_tail_periodic_validation_save_and_completion(tiny, tmp_path):
    contract, data = tiny
    output = tmp_path / "full"
    output.mkdir()
    result = runtime.run(contract, data, output)
    assert result["status"] == "completed"
    assert result["global_step"] == 3 and result["committed_samples"] == 5
    rows = [json.loads(s) for s in (output / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in rows if r["event"] == "update"]
    assert [r["micro_batches"] for r in updates] == [2, 2, 1]
    assert result["valid_tokens_seen"] == data["split_statistics"]["train"]["total_targets_after_shift"]
    assert [r["global_step"] for r in rows if r["event"] == "checkpoint_saved"] == [2, 3]
    assert [r["global_step"] for r in rows if r["event"] == "periodic_validation"] == [1, 2]
    assert result["final_validation"] is not None


def test_budget_pauses_at_update_boundary_and_preserves_resume_plan(tiny, tmp_path):
    contract, data = tiny
    output = tmp_path / "timed"
    output.mkdir()
    clock_value = [0.0]
    original = runtime.train_update

    def update(*args, **kwargs):
        result = original(*args, **kwargs)
        clock_value[0] = 14 * 3600 + 1
        return result

    # Use a temporary replacement only within this test's try/finally.
    runtime.train_update = update
    try:
        result = runtime.run(contract, data, output, clock=lambda: clock_value[0])
    finally:
        runtime.train_update = original
    assert result["status"] == "paused" and result["stop_reason"] == "time_budget"
    assert result["global_step"] == 1 and result["committed_samples"] == 2
    assert result["planned_updates"] == 3 and result["final_validation"] is None
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert saved["scheduler"]["last_epoch"] == 1


def test_formal_gate_restore_exact_and_no_gate_checkpoint_as_formal_resume(tiny, tmp_path):
    contract, data = tiny
    folders = [tmp_path / n for n in ("continuous", "paused", "resumed", "illegal")]
    for p in folders:
        p.mkdir()
    a = runtime.run(contract, data, folders[0], gate_stop=2, gate_validation_records=2)
    b = runtime.run(contract, data, folders[1], gate_stop=1, gate_validation_records=2)
    c = runtime.run(contract, data, folders[2], gate_stop=2, gate_validation_records=2, resume=Path(b["checkpoint"]))
    left, right = [torch.load(r["checkpoint"], weights_only=True) for r in (a, c)]
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
    assert left["sampler_state"] == right["sampler_state"]
    assert left["scheduler"] == right["scheduler"]
    assert a["final_validation"] == c["final_validation"]
    with pytest.raises(ValueError, match="dedicated"):
        runtime.run(contract, data, folders[3], resume=Path(b["checkpoint"]))


def test_cli_rejects_formal_run_without_freeze_before_creating_output(tmp_path):
    output = tmp_path / "must_not_exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_frozen_sft.py"), "--run", "--config", "absent.json",
                             "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    assert "requires --freeze" in result.stderr


def test_freeze_check_rejects_changed_bound_file_before_preparation(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("original")
    marker = tmp_path / "freeze.json"
    marker.write_text(json.dumps({"status": "frozen", "formal_training_started": False,
                                 "files": {"training_config": {"file": "config.json", "sha256": "wrong"}}}))
    monkeypatch.setattr(freezing, "build_contract", lambda p: pytest.fail("must reject before loading runtime"))
    with pytest.raises(ValueError, match="frozen input changed"):
        freezing.check(marker)
