import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_sft_identity", ROOT / "scripts/inspect_sft_identity.py")
identity = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(identity)


def test_upstream_self_introduction_has_message_and_field_locations():
    result = identity.inspect_messages([{"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是由jingyaogong开发的MiniMind。"}])
    assert result["review_bucket"] == "upstream_with_identity_cue"
    assert {"upstream_author", "upstream_model", "user_identity_question", "assistant_identity_phrase"} <= set(result["signals"])
    assert result["locations"][1]["message_index"] == 1


def test_objective_upstream_reference_is_a_separate_bucket_not_a_rewrite():
    result = identity.inspect_messages([{"role": "user", "content": "介绍MiniMind项目。"},
        {"role": "assistant", "content": "MiniMind项目的作者是jingyaogong。"}])
    assert result["review_bucket"] == "upstream_without_identity_cue"


def test_reasoning_training_claim_and_target_name_are_visible():
    result = identity.inspect_messages([{"role": "assistant", "content": "可以。",
        "reasoning_content": "我的训练数据来源由yanpeng说明。"}])
    assert result["review_bucket"] == "target_name_present"
    assert "training_data_self_claim" in result["signals"]
    assert result["locations"][0]["field"] == "reasoning_content"


def test_other_model_names_and_plain_records_do_not_become_upstream_identities():
    result = identity.inspect_messages([{"role": "assistant", "content": "I am ChatGPT."}])
    assert result["review_bucket"] == "other_name_with_identity_cue"
    assert identity.inspect_messages([{"role": "assistant", "content": "2 + 2 = 4。"}])["review_bucket"] is None


def test_roleplay_phrase_is_only_a_lexical_cue():
    result = identity.inspect_messages([{"role": "assistant", "content": "我是故事里的小明。"}])
    assert result["review_bucket"] == "identity_cue_without_named_model"
    assert "action" not in result


def test_inventory_preserves_source_and_reproduces_bucket_sampling(tmp_path):
    rows = [(json.dumps({"conversations": [{"role": "assistant", "content": f"我是MiniMind，编号{i}。"}]}, ensure_ascii=False) + "\n").encode()
            for i in range(12)]
    source = tmp_path / "candidate.jsonl"
    raw = b"".join(rows)
    source.write_bytes(raw)
    ledger = b"".join((json.dumps({"input_line": i, "source_line": i + 10,
        "source_line_sha256": hashlib.sha256(row).hexdigest(), "action": "keep",
        "retained_output_line": i, "retained_source_line": i + 10}) + "\n").encode()
        for i, row in enumerate(rows, 1))
    lineage = tmp_path / "lineage.jsonl"
    lineage.write_bytes(ledger)
    reference = {"outputs": {"candidate": {"path": str(source), "records": 12, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()}, "lineage": {"path": str(lineage), "records": 12,
        "bytes": len(ledger), "sha256": hashlib.sha256(ledger).hexdigest()}}}
    one = identity.inventory(reference, tmp_path / "one", sample_size=3, seed=123)
    two = identity.inventory(reference, tmp_path / "two", sample_size=3, seed=123)
    assert one["review_hit_records"] == one["exclusive_review_buckets"]["upstream_with_identity_cue"] == 12
    assert one["files"]["samples"]["sha256"] == two["files"]["samples"]["sha256"]
    assert one["files"]["samples"]["records"] == 3
    assert source.read_bytes() == raw
    with pytest.raises(FileExistsError):
        identity.inventory(reference, tmp_path / "one")
