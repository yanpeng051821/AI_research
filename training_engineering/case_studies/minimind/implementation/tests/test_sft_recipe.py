import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_sft_recipe.py"
spec = importlib.util.spec_from_file_location("prepare_sft_recipe", SCRIPT)
recipe = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(recipe)


def record(question, answer="answer"):
    return {"conversations": [{"role": "user", "content": question},
                              {"role": "assistant", "content": answer}]}


def policy():
    return json.loads((ROOT.parent / "experiments/sft_mini_v1/data_recipe/processing_policy.json")
                      .read_text(encoding="utf-8"))


def test_identity_rewrite_is_bounded_and_does_not_alias_input():
    original = record("你叫什么名字？", "我是MiniMind。")
    original["conversations"][1]["reasoning_content"] = "由jingyaogong开发"
    known = {recipe.digest("我是MiniMind。".encode())}
    result, action, issues = recipe.identity_action(original, policy(), known)
    assert action == "rewrite_identity" and not issues
    assert "yanpeng" in result["conversations"][1]["content"]
    assert result["conversations"][1]["reasoning_content"] == ""
    assert original["conversations"][1]["reasoning_content"] == "由jingyaogong开发"
    assert recipe.identity_action(record("请分析代码", "我是MiniMind。"), policy(), known)[1] == "hold"
    assert recipe.identity_action(record("什么是MiniMind", "第三方项目介绍"), policy(), known)[1] == "hold"
    assert recipe.identity_action(record("你是谁以及你是怎么工作的", "我是MiniMind。"), policy(), known)[1] == "hold"


def test_grouping_connects_shared_prompts_transitively_and_protects_development():
    groups = recipe.Groups()
    a = record(" SAME  prompt ")
    b = record("same prompt", "different answer")
    b["conversations"] += record("next")["conversations"]
    groups.add(a, 1, True)
    groups.add(b, 2, False)
    groups.add(record("next"), 3, False)
    groups.add(record("independent"), 4, False)
    assert groups.find(0) == groups.find(1) == groups.find(2)
    selected = groups.validation_roots(1, 42)
    assert selected == {groups.find(3)}
    assert groups.development[groups.find(0)]


def test_content_revision_binds_source_and_message_and_preserves_context():
    original = record("question", "unsupported answer")
    raw = (recipe.canonical(original) + "\n").encode()
    config = {"exact_content_revisions": {"12": {
        "source_line_sha256": recipe.digest(raw), "assistant_message_index": 1,
        "original_message_sha256": recipe.digest(recipe.canonical(original["conversations"][1]).encode()),
        "replacement_message": {"role": "assistant", "content": "bounded answer", "reasoning_content": ""},
    }}}
    changed = recipe.content_revision(original, raw, 12, config)
    assert changed["conversations"][0] == original["conversations"][0]
    assert changed["conversations"][1]["content"] == "bounded answer"
    assert original["conversations"][1]["content"] == "unsupported answer"
    with pytest.raises(ValueError, match="source bytes changed"):
        recipe.content_revision(original, raw + b" ", 12, config)
    config["exact_content_revisions"]["12"]["original_message_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="assistant message changed"):
        recipe.content_revision(original, raw, 12, config)
    assert recipe.content_revision(original, raw, 13, config) is None


def test_recipe_cli_accounts_for_every_source_and_keeps_groups_together(tmp_path):
    rows = [record("你是谁", "我是MiniMind。"), record("解释物理", "我是MiniMind。"),
            record("一加一", "二"), record("一加一", "2"), record("hi", "hello"),
            record("你是谁", "我是MiniMind。")]
    raw_rows = [(recipe.canonical(r) + "\n").encode() for r in rows]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(raw_rows))
    lineage = tmp_path / "input_lineage.jsonl"
    lineage.write_text("".join(recipe.canonical({"input_line": i, "action": "keep", "source_line": i,
                         "retained_source_line": i, "retained_output_line": i,
                         "source_line_sha256": recipe.digest(raw)}) + "\n"
                               for i, raw in enumerate(raw_rows, 1)), encoding="utf-8")

    def descriptor(path, records):
        return {"path": str(path), "records": records, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    input_desc = descriptor(source, len(rows))
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"outputs": {"candidate": input_desc, "lineage": descriptor(lineage, len(rows))},
                              "original_source": input_desc}), encoding="utf-8")
    group_ref = tmp_path / "groups.json"
    group_ref.write_text(json.dumps({"input": input_desc, "top_groups": {"upstream_reply": [
        {"group_sha256": recipe.digest("我是MiniMind。".encode())}]}}), encoding="utf-8")
    config = policy()
    config["input_candidate_sha256"] = input_desc["sha256"]
    config["exact_review_holds"] = {}
    config["collection"]["validation_records_target"] = 1
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(config), encoding="utf-8")
    samples = tmp_path / "development.jsonl"
    samples.write_text('{"source_line":1}\n', encoding="utf-8")
    report_dir = tmp_path / "report"
    contract_path = tmp_path / "contract.json"
    cmd = [sys.executable, str(SCRIPT), "--candidate-report", str(ref), "--groups-report", str(group_ref),
           "--policy", str(policy_path), "--development-samples", str(samples), "--output-dir", str(tmp_path / "data"),
           "--report-dir", str(report_dir), "--contract", str(contract_path), "--workers", "1", "--chunk-size", "2"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert report["actions"] == {"rewrite_identity": 1, "hold": 1, "keep": 3, "duplicate": 1}
    assert report["outputs"]["accepted"]["records"] == 4
    assert report["outputs"]["validation"]["records"] == 1
    assert report["outputs"]["train"]["records"] == 3
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["ready_for_training"] is False
    assert contract["status"] == "candidate_pending_review_not_frozen"
    assert not contract["approval"]["new_processing_and_split_choices_reviewed"]
    split_rows = [json.loads(line) for line in Path(report["outputs"]["split_index"]["path"]).read_text(encoding="utf-8").splitlines()]
    assert split_rows[0]["split"] == "train"
    assert split_rows[1]["split"] == split_rows[2]["split"]
    assert split_rows[1]["group_id"] == split_rows[2]["group_id"]
    verification = tmp_path / "verification.json"
    verify_cmd = [sys.executable, str(ROOT / "scripts/verify_sft_recipe.py"), "--contract", str(contract_path),
                  "--output", str(verification), "--load-datasets"]
    verified = subprocess.run(verify_cmd, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert verified.returncode == 0, verified.stderr
    evidence = json.loads(verification.read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["normalized_prompt_cross_split_overlap"] == 0
    assert evidence["validation_records_independently_reencoded"] == 1
    assert evidence["dataset_checks"]["train"]["records_loaded"] == 3
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(json.dumps({
        "status": "user_authorized_bounded_data_freeze",
        "candidate_sha256": recipe.digest(contract_path.read_bytes()),
        "policy_sha256": contract["policy"]["sha256"], "accepted_limits": contract["limits"],
        "training_execution_authorized": False,
    }), encoding="utf-8")
    frozen_dir = tmp_path / "frozen"
    freeze_cmd = [sys.executable, str(ROOT / "scripts/freeze_sft_data_recipe.py"),
                  "--candidate", str(contract_path), "--acceptance", str(acceptance_path),
                  "--recipe-dir", str(frozen_dir), "--verification-output", str(tmp_path / "final_verification.json")]
    frozen = subprocess.run(freeze_cmd, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert frozen.returncode == 0, frozen.stderr
    final_contract = json.loads((frozen_dir / "DATA_CONTRACT.json").read_text(encoding="utf-8"))
    assert final_contract["status"] == "frozen" and final_contract["ready_for_training"]
    assert not final_contract["training_execution_authorized"]
    assert subprocess.run(freeze_cmd, cwd=ROOT, capture_output=True, timeout=90).returncode != 0
    check_cmd = [sys.executable, str(ROOT / "scripts/freeze_sft_data_recipe.py"), "--check", str(frozen_dir / "data_freeze.json")]
    assert subprocess.run(check_cmd, cwd=ROOT, capture_output=True, timeout=90).returncode == 0
    verification.unlink()
    tampered_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered_contract["encoding"]["assistant_only"] = False
    tampered_path = tmp_path / "tampered_contract.json"
    tampered_path.write_text(json.dumps(tampered_contract), encoding="utf-8")
    tampered_cmd = [sys.executable, str(ROOT / "scripts/verify_sft_recipe.py"), "--contract", str(tampered_path),
                    "--output", str(verification)]
    tampered_result = subprocess.run(tampered_cmd, cwd=ROOT, capture_output=True, timeout=90)
    assert tampered_result.returncode != 0
    assert b"encoding contract mismatch" in tampered_result.stderr
    source.write_bytes(source.read_bytes() + b"\n")
    assert subprocess.run(check_cmd, cwd=ROOT, capture_output=True, timeout=90).returncode != 0
    assert subprocess.run(verify_cmd, cwd=ROOT, capture_output=True, timeout=90).returncode != 0
    assert not verification.exists()
    assert subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=90).returncode != 0
