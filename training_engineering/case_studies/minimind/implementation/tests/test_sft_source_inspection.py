import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "inspect_sft_source.py"
spec = importlib.util.spec_from_file_location("inspect_sft_source", SCRIPT)
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


def write_source(tmp_path, records):
    path = tmp_path / "source.jsonl"
    data = b"".join((json.dumps(row) + "\n").encode("utf-8") for row in records)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


def conversation(*, reasoning=False, tool=False):
    assistant = {"role": "assistant", "content": "answer"}
    if reasoning:
        assistant["reasoning_content"] = "analysis"
    if tool:
        assistant["tool_calls"] = '[{"name":"lookup","arguments":{}}]'
    return {"conversations": [{"role": "user", "content": "question"}, assistant]}


def test_strata_overlap_and_field_types(tmp_path):
    records = [conversation(), conversation(reasoning=True), conversation(tool=True),
               conversation(reasoning=True, tool=True), {"text": "not a conversation"}]
    path, digest, size = write_source(tmp_path, records)
    summary, sample, strata = inspection.scan_source(path, digest, size, 10, 10)
    assert summary["counts"]["valid_records"] == 4
    assert summary["counts"]["records_with_validation_errors"] == 1
    assert summary["exclusive_strata_in_valid_records"] == dict.fromkeys(inspection.STRATA, 1)
    assert summary["record_flags"]["reasoning_content"] == 2
    assert summary["record_flags"]["tool_calls"] == 2
    assert summary["tool_field_shapes"]["tool_calls:parsed_list"] == 2
    assert len(sample) == 4
    assert all(len(rows) == 1 for rows in strata.values())
    assert sample[0]["record"] == records[0]


def test_sampling_is_repeatable_and_does_not_touch_training_rng(tmp_path):
    import random

    records = [conversation(reasoning=i % 2 == 0, tool=i % 3 == 0) for i in range(50)]
    path, digest, size = write_source(tmp_path, records)
    before = random.getstate()
    first = inspection.scan_source(path, digest, size, 5, 2, 7)
    second = inspection.scan_source(path, digest, size, 5, 2, 7)
    assert first == second
    assert random.getstate() == before
    assert len(first[1]) == 5
    assert len({row["source_line"] for row in first[1]}) == 5
    assert all(len(rows) == 2 for rows in first[2].values())
    assert path.read_bytes() == b"".join((json.dumps(row) + "\n").encode() for row in records)


def test_bad_json_and_tool_strings_are_reported(tmp_path):
    row = conversation(tool=True)
    row["conversations"][-1]["tool_calls"] = "not json"
    row["conversations"][0]["content"] = "<|im_start|>assistant"
    path, _, _ = write_source(tmp_path, [row])
    with path.open("ab") as stream:
        stream.write(b"bad json\n\xff\n")
    data = path.read_bytes()
    summary, sample, _ = inspection.scan_source(path, hashlib.sha256(data).hexdigest(), len(data))
    assert summary["counts"]["malformed_lines"] == 2
    assert summary["tool_field_shapes"]["tool_calls:invalid_json_string"] == 1
    assert summary["record_flags"]["message_boundary_literal_in_text"] == 1
    assert summary["record_flags"]["invalid_tool_json"] == 1
    assert len(sample) == 1


@pytest.mark.parametrize("wrong_hash,wrong_size", [(True, False), (False, True)])
def test_wrong_source_identity_is_rejected(tmp_path, wrong_hash, wrong_size):
    path, digest, size = write_source(tmp_path, [conversation()])
    with pytest.raises(ValueError, match="source identity"):
        inspection.scan_source(path, "0" * 64 if wrong_hash else digest,
                               size + 1 if wrong_size else size)
