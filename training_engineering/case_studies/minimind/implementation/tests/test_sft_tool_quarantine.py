from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("apply_sft_tool_quarantine", ROOT / "scripts/apply_sft_tool_quarantine.py")
quarantine = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(quarantine)


def inputs():
    flags = [{"source_line": 10, "source_line_sha256": "1" * 64,
              "issues": [{"code": "undeclared_function_call", "called_name": "weather"}]},
             {"source_line": 20, "source_line_sha256": "2" * 64,
              "issues": [{"code": "undeclared_function_call", "called_name": "clock"}]},
             {"source_line": 30, "source_line_sha256": "3" * 64,
              "issues": [{"code": "duplicate_declared_name", "name": "clock"}]}]
    payload = "".join(json.dumps(flag) + "\n" for flag in flags).encode()
    candidate = {"status": "confirmed_exclusions_applied_not_final_training_data",
                 "dedup_report_sha256": "d" * 64, "original_source": {"sha256": "a" * 64},
                 "applied_decisions": [{"source_line": 10, "source_line_sha256": "1" * 64}]}
    report = {"dedup_report_sha256": "d" * 64, "original_source_sha256": "a" * 64,
              "issues_file": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 3}}
    rule = {"status": "user_confirmed", "action": "exclude_whole_record_to_quarantine",
            "issue_code": "undeclared_function_call", "policy_id": "test_rule", "source_sha256": "a" * 64}
    return candidate, report, payload, rule


def test_plan_intersects_flags_with_prior_exact_exclusions_and_ignores_other_codes():
    decisions, previous = quarantine.build_exclusion_plan(*inputs())
    assert previous == [10]
    assert set(decisions) == {20}
    assert decisions[20]["source_line_sha256"] == "2" * 64
    assert decisions[20]["status"] == "user_confirmed"


def test_unconfirmed_rule_is_rejected():
    candidate, report, payload, rule = inputs()
    rule["status"] = "assistant_proposal"
    with pytest.raises(ValueError, match="unconfirmed"):
        quarantine.build_exclusion_plan(candidate, report, payload, rule)


def test_different_candidate_lineage_is_rejected():
    candidate, report, payload, rule = inputs()
    candidate["dedup_report_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source lineage"):
        quarantine.build_exclusion_plan(candidate, report, payload, rule)


def test_changed_issue_file_and_prior_record_hash_are_rejected():
    candidate, report, payload, rule = inputs()
    with pytest.raises(ValueError, match="issue file identity"):
        quarantine.build_exclusion_plan(candidate, report, payload + b"\n", rule)
    candidate["applied_decisions"][0]["source_line_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="previously excluded"):
        quarantine.build_exclusion_plan(candidate, report, payload, rule)


def test_duplicate_flags_and_incomplete_issue_counts_are_rejected():
    candidate, report, payload, rule = inputs()
    duplicated = payload + payload.splitlines(keepends=True)[0]
    report["issues_file"]["sha256"] = hashlib.sha256(duplicated).hexdigest()
    with pytest.raises(ValueError, match="duplicate source"):
        quarantine.build_exclusion_plan(candidate, report, duplicated, rule)
    report["issues_file"]["sha256"] = hashlib.sha256(payload).hexdigest()
    report["issues_file"]["records"] = 4
    with pytest.raises(ValueError, match="issue count"):
        quarantine.build_exclusion_plan(candidate, report, payload, rule)


def test_plan_does_not_mutate_reports_or_rule():
    arguments = inputs()
    before = deepcopy(arguments)
    quarantine.build_exclusion_plan(*arguments)
    assert arguments == before
