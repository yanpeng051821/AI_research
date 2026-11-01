import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_sft_duplicates.py"
spec = importlib.util.spec_from_file_location("inspect_sft_duplicates", SCRIPT)
duplicates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duplicates)


def conversation(answer="answer"):
    return [{"role": "user", "content": "question"},
            {"role": "assistant", "content": answer}]


def test_key_ignores_object_key_order_but_preserves_message_content_and_order():
    messages = conversation()
    reordered_keys = [dict(reversed(list(message.items()))) for message in messages]
    key = duplicates.conversation_sha256(messages)
    assert duplicates.conversation_sha256(reordered_keys) == key
    assert duplicates.conversation_sha256(conversation("answer ")) != key
    assert duplicates.conversation_sha256(conversation("other")) != key
    assert duplicates.conversation_sha256(messages[::-1]) != key
    messages[1]["reasoning_content"] = "reason"
    assert duplicates.conversation_sha256(messages) != key


def test_scan_counts_repeated_groups_without_modifying_source(tmp_path):
    records = [{"conversations": conversation(), "id": 1},
               {"conversations": conversation(), "id": 2},
               {"conversations": conversation()},
               {"conversations": conversation("other")}]
    raw = "".join(json.dumps(record) + "\n" for record in records).encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    report = duplicates.scan_duplicates(source, hashlib.sha256(raw).hexdigest(), len(raw), 4)
    assert report["counts"] == {"unique_conversations": 2, "duplicate_groups": 1,
                                "records_in_duplicate_groups": 3,
                                "duplicates_beyond_first": 2, "max_group_size": 3}
    assert [pair["duplicate_source_line"] for pair in report["example_pairs"]] == [2, 3]
    assert source.read_bytes() == raw


def test_source_identity_mismatch_rejects_result(tmp_path):
    raw = (json.dumps({"conversations": conversation()}) + "\n").encode()
    source = tmp_path / "source.jsonl"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="identity/count"):
        duplicates.scan_duplicates(source, "0" * 64, len(raw), 1)


def test_invalid_record_is_not_silently_omitted(tmp_path):
    source = tmp_path / "source.jsonl"
    raw = b'{"conversations": []}\n'
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="invalid record at line 1"):
        duplicates.scan_duplicates(source, hashlib.sha256(raw).hexdigest(), len(raw), 1)
