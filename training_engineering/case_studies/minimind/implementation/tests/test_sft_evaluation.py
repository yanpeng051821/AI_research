import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_sft", ROOT / "scripts/evaluate_sft.py")
evaluation = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(evaluation)


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens):
        return ",".join(str(i) for i in ids if not skip_special_tokens or i != self.eos_token_id)


class FakeModel(torch.nn.Module):
    def __init__(self, sequence, finite=True):
        super().__init__()
        self.sequence = sequence
        self.finite = finite
        self.modes = []

    def forward(self, input_ids, attention_mask):
        self.modes.append(self.training)
        logits = torch.zeros(1, input_ids.shape[1], 10)
        logits[:, -1, self.sequence[input_ids.shape[1] - 2]] = 1 if self.finite else float("nan")
        return logits


@pytest.mark.parametrize("sequence,reason", [([4, 2], "eos"), ([4, 5], "length_limit")])
def test_case_reuses_greedy_records_untrimmed_tokens_and_restores_mode(monkeypatch, sequence, reason):
    monkeypatch.setattr(evaluation, "render_probe", lambda *args: ("input only", [1, 3], []))
    model = FakeModel(sequence)
    result = evaluation.generate_case(model, FakeTokenizer(),
        {"id": "example", "category": "instruction", "tier": "core", "checks": ["must not render"]},
        {"generation": {"max_new_tokens": 2, "context_budget": 4}}, torch.device("cpu"))
    assert result["generated_ids"] == sequence
    assert result["input_ids"] == [1, 3] and result["stop_reason"] == reason
    assert result["raw_text"] == ",".join(map(str, sequence))
    assert result["scoring_status"] == "pending_manual_review"
    assert model.modes == [False, False] and model.training


def test_bad_budget_and_nonfinite_logits_fail_without_leaking_eval_mode(monkeypatch):
    monkeypatch.setattr(evaluation, "render_probe", lambda *args: ("input only", [1, 3], []))
    model = FakeModel([4], finite=False)
    case = {"id": "example", "category": "instruction", "tier": "core"}
    with pytest.raises(ValueError, match="context budget"):
        evaluation.generate_case(model, FakeTokenizer(), case,
                                 {"generation": {"max_new_tokens": 2, "context_budget": 3}}, torch.device("cpu"))
    assert not model.modes
    with pytest.raises(FloatingPointError):
        evaluation.generate_case(model, FakeTokenizer(), case,
                                 {"generation": {"max_new_tokens": 1, "context_budget": 3}}, torch.device("cpu"))
    assert model.training


def test_draft_cannot_run_or_create_a_result_directory(tmp_path):
    config = ROOT.parent / "experiments/sft_mini_v1/evaluation_recipe/evaluation_config.draft.json"
    output = tmp_path / "must_not_exist"
    args = argparse.Namespace(config=config, run=True, device="cpu", output_dir=output)
    with pytest.raises(ValueError, match="approved frozen configuration"):
        evaluation.run(args)
    assert not output.exists()
