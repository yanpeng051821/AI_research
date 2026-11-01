from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("sft_scoring_patch", SCRIPTS / "apply_sft_scoring_patch.py")
patching = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(patching)
finally:
    sys.path.pop(0)


def example():
    patch = {"version": "scoring_v2", "status": "user_approved_before_sft",
             "change_scope": "append_one_check_only", "case_id": "location",
             "append_check": "location only", "prompts_changed": False,
             "generation_changed": False, "other_checks_changed": False}
    addition = {"version": "scoring_v2", "id": "location", "index": 2,
                "criterion": "location only", "status": "fail", "reason": "extra time",
                "evidence_quotes": ["14:10"], "case_line_sha256": "a",
                "generation_line_sha256": "b"}
    case = {"id": "location", "checks": ["correct location"]}
    generation = {"id": "location", "raw_text": "east floor 14:10"}
    original = {"checks": [{"status": "pass"}], "case_line_sha256": "a",
                "generation_line_sha256": "b"}
    return patch, addition, case, generation, original


def test_extra_check_changes_outcome_without_mutating_original():
    args = example()
    before = deepcopy(args)
    assert patching.validate_addition(*args) == "fail"
    assert args == before


@pytest.mark.parametrize("change,expected", [
    ("scope", "scope"), ("criterion", "criterion"),
    ("source", "source"), ("quote", "evidence"),
])
def test_patch_rejects_scope_criterion_source_and_evidence_changes(change, expected):
    patch, addition, case, generation, original = example()
    if change == "scope":
        patch["prompts_changed"] = True
    elif change == "criterion":
        addition["criterion"] = "different rule"
    elif change == "source":
        addition["generation_line_sha256"] = "another output"
    else:
        addition["evidence_quotes"] = ["invented"]
    with pytest.raises(ValueError, match=expected):
        patching.validate_addition(patch, addition, case, generation, original)
