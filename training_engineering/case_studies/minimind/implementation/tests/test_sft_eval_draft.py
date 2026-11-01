from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_sft_eval_draft", ROOT / "scripts/inspect_sft_eval_draft.py")
inspector = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(inspector)


@pytest.mark.parametrize("case_id", ["context-01", "tool-03"])
def test_rendering_preserves_prefix_and_never_includes_rubric_or_protocol_placeholder(case_id):
    directory = ROOT.parent / "experiments/sft_mini_v1/evaluation_recipe"
    config = json.loads((directory / "evaluation_config.draft.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (directory / "probes.draft.jsonl").read_text(encoding="utf-8").splitlines()]
    case = next(row for row in cases if row["id"] == case_id)
    before = deepcopy(case)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "assets/minimind_tokenizer", local_files_only=True)
    text, ids, _ = inspector.render_probe(case, config, tokenizer)
    changed = deepcopy(case)
    changed["checks"] = ["RUBRIC_MUST_NOT_ENTER_PROMPT"]
    changed["id"] = "DIFFERENT_ID"
    other_text, other_ids, _ = inspector.render_probe(changed, config, tokenizer)
    assert case == before
    assert (text, ids) == (other_text, other_ids)
    assert "protocol audit placeholder" not in text
    assert "RUBRIC_MUST_NOT_ENTER_PROMPT" not in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(ids) + config["generation"]["max_new_tokens"] <= config["generation"]["context_budget"]
    if case_id == "tool-03":
        assert '"temperature_c":18' in text
        assert "<tool_response>" in text
