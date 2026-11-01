import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deduplicate_sft_subset", ROOT / "scripts/deduplicate_sft_subset.py")
dedup = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(dedup)


def fixture_data(tmp_path):
    messages = [{"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}]
    records = [{"conversations": messages, "id": 1},
               {"conversations": messages, "id": "long"},
               {"id": 2, "conversations": [dict(reversed(list(m.items()))) for m in messages]},
               {"conversations": [messages[0], {"role": "assistant", "content": "other"}]}]
    rows = [(json.dumps(record) + "\r\n").encode() for record in records]
    # Source line 2 is absent from the length candidate, but present in the audit.
    candidate = b"".join(rows[index] for index in (0, 2, 3))
    source = tmp_path / "length_eligible.jsonl"
    source.write_bytes(candidate)
    details = tmp_path / "details.jsonl"
    detail_bytes = b"".join((json.dumps({"source_line": i, "issues": [],
                                         "source_line_sha256": hashlib.sha256(row).hexdigest(),
                                         "input_tokens": length, "targets_after_shift": 2}) + "\n").encode()
                            for i, (row, length) in enumerate(zip(rows, [768, 769, 12, 10]), 1))
    details.write_bytes(detail_bytes)
    length_report = {"policy": {"max_length": 768},
                     "encoding_details_sha256": hashlib.sha256(detail_bytes).hexdigest(),
                     "outputs": {"length_eligible": {"records": 3, "bytes": len(candidate),
                                                     "sha256": hashlib.sha256(candidate).hexdigest(),
                                                     "targets_after_shift": 6}}}
    return rows, source, details, length_report, {"overall": {"records": 4}}


def test_keep_first_preserves_bytes_and_traces_original_source_lines(tmp_path):
    rows, source, details, length_report, encoding_report = fixture_data(tmp_path)
    result = dedup.deduplicate(source, details, tmp_path / "out", length_report, encoding_report)
    assert result["input_records"] == 3
    assert result["unique_conversations"] == 2
    assert result["duplicate_groups"] == 1
    assert result["records_in_duplicate_groups"] == result["max_group_size"] == 2
    assert Path(result["outputs"]["unique"]["path"]).read_bytes() == rows[0] + rows[3]
    assert Path(result["outputs"]["duplicates"]["path"]).read_bytes() == rows[2]
    trace = [json.loads(line) for line in Path(result["outputs"]["lineage"]["path"]).read_text().splitlines()]
    assert [row["source_line"] for row in trace] == [1, 3, 4]
    assert [row["retained_source_line"] for row in trace] == [1, 1, 4]
    assert [row["retained_output_line"] for row in trace] == [1, 1, 2]
    assert [row["action"] for row in trace] == ["keep", "exclude_exact_duplicate", "keep"]
    assert source.read_bytes() == rows[0] + rows[2] + rows[3]
    assert result["outputs"]["unique"]["targets_after_shift"] == 4


def test_changed_reasoning_is_not_an_exact_duplicate():
    base = [{"role": "assistant", "content": "answer"}]
    changed = [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}]
    assert dedup.conversation_sha256(base) != dedup.conversation_sha256(changed)


def test_wrong_candidate_identity_does_not_publish_final_files(tmp_path):
    _, source, details, length_report, encoding_report = fixture_data(tmp_path)
    length_report["outputs"]["length_eligible"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity/count/target"):
        dedup.deduplicate(source, details, tmp_path / "out", length_report, encoding_report)
    assert not list((tmp_path / "out").glob("*.jsonl"))


def test_detail_hash_checks_even_excluded_tail_records(tmp_path):
    _, source, details, length_report, encoding_report = fixture_data(tmp_path)
    length_report["encoding_details_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="detail identity/count"):
        dedup.deduplicate(source, details, tmp_path / "out", length_report, encoding_report)
    assert not list((tmp_path / "out").glob("*.jsonl"))


def test_reordered_candidate_is_rejected(tmp_path):
    rows, source, details, length_report, encoding_report = fixture_data(tmp_path)
    source.write_bytes(rows[3] + rows[2] + rows[0])
    with pytest.raises(ValueError, match="candidate/detail mismatch"):
        dedup.deduplicate(source, details, tmp_path / "out", length_report, encoding_report)


def test_existing_output_is_not_overwritten(tmp_path):
    _, source, details, length_report, encoding_report = fixture_data(tmp_path)
    with pytest.raises(FileExistsError):
        dedup.deduplicate(source, details, tmp_path, length_report, encoding_report)
