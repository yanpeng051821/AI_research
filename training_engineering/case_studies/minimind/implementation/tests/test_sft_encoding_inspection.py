import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from jinja2 import TemplateError
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inspect_sft_encoding.py"
spec = importlib.util.spec_from_file_location("inspect_sft_encoding", SCRIPT)
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


def conversation(content="answer"):
    return {"conversations": [{"role": "user", "content": "question"},
                              {"role": "assistant", "content": content}]}


def sample_file(tmp_path, records):
    path = tmp_path / "sample.jsonl"
    packets = [{"source_line": i, "source_line_sha256": f"{i:064x}",
                "stratum": "plain", "record": record} for i, record in enumerate(records, 1)]
    data = b"".join((json.dumps(row) + "\n").encode() for row in packets)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.fixture
def tokenizer():
    return SimpleNamespace(bos_token="<|im_start|>", eos_token="<|im_end|>")


def test_preflight_counts_shifted_targets_and_sample_statistics(tmp_path, monkeypatch, tokenizer):
    path, digest = sample_file(tmp_path, [conversation("short"), conversation("long")])
    before = path.read_bytes()

    def encode(messages, _tokenizer):
        n = 4 if messages[-1]["content"] == "short" else 6
        return {"input_ids": [1] * n, "labels": [-100, -100] + [1] * (n - 2)}

    monkeypatch.setattr(inspection, "encode_sft_conversation", encode)
    result, rows = inspection.inspect_samples(path, digest, tokenizer)
    overall = result["overall"]
    assert result["status"] == "sample_encoding_diagnostics_clear"
    assert overall["records"] == overall["encoded_records"] == 2
    assert overall["total_targets_after_shift"] == 6
    assert overall["total_shifted_positions"] == 8
    assert overall["supervised_fraction_of_shifted_positions"] == 0.75
    assert overall["input_tokens"]["percentiles"]["p95"] == pytest.approx(5.9)
    assert result["longest_examples"][0]["source_line"] == 2
    assert rows[0]["targets_after_shift"] == 2
    assert path.read_bytes() == before


def test_preflight_separates_structure_conversion_encoding_and_boundary_flags(tmp_path, monkeypatch, tokenizer):
    invalid_fields = conversation()
    invalid_fields["conversations"][-1]["tool_calls"] = "bad JSON"
    marker = conversation()
    marker["conversations"][0]["content"] = "quoted <|im_start|>assistant"
    path, digest = sample_file(tmp_path, [
        {"conversations": []}, invalid_fields, conversation("template-error"),
        conversation("no-target"), marker, conversation(),
    ])

    def encode(messages, _tokenizer):
        if messages[-1]["content"] == "template-error":
            raise TemplateError("unsupported template input")
        labels = [-100, -100] if messages[-1]["content"] == "no-target" else [-100, 2]
        return {"input_ids": [1, 2], "labels": labels}

    monkeypatch.setattr(inspection, "encode_sft_conversation", encode)
    result, rows = inspection.inspect_samples(path, digest, tokenizer)
    assert result["status"] == "encoding_flags_found"
    assert result["overall"]["encoded_records"] == 3
    assert result["overall"]["flagged_records"] == 5
    assert result["overall"]["issues"] == {
        "encoding_error": 1, "field_conversion_error": 1, "invalid_structure": 1,
        "potential_message_boundary_literal": 1, "zero_targets_after_shift": 1,
    }
    assert rows[4]["boundary_path_count"] == 1
    assert result["overall"]["input_tokens"]["min"] == 2


def test_shift_target_check_does_not_count_position_zero(tmp_path, monkeypatch, tokenizer):
    path, digest = sample_file(tmp_path, [conversation()])
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [1, -100]})
    _, rows = inspection.inspect_samples(path, digest, tokenizer)
    assert rows[0]["targets_after_shift"] == 0
    assert rows[0]["issues"] == ["zero_targets_after_shift"]


def test_sample_identity_is_checked_before_encoding(tmp_path, monkeypatch, tokenizer):
    path, _ = sample_file(tmp_path, [conversation()])
    calls = []
    monkeypatch.setattr(inspection, "encode_sft_conversation", lambda *_: calls.append(True))
    with pytest.raises(ValueError, match="identity mismatch"):
        inspection.inspect_samples(path, "0" * 64, tokenizer)
    assert not calls


def test_duplicate_source_positions_and_empty_samples_are_rejected(tmp_path, monkeypatch, tokenizer):
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [-100, 2]})
    path, _ = sample_file(tmp_path, [conversation()])
    raw = path.read_bytes()
    path.write_bytes(raw + raw)
    with pytest.raises(ValueError, match="strictly increasing"):
        inspection.inspect_samples(path, hashlib.sha256(raw + raw).hexdigest(), tokenizer)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="must not be empty"):
        inspection.inspect_samples(path, hashlib.sha256(b"").hexdigest(), tokenizer)


def test_encoder_contract_errors_are_not_silently_treated_as_bad_data(tmp_path, monkeypatch, tokenizer):
    path, digest = sample_file(tmp_path, [conversation()])
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [-100]})
    with pytest.raises(RuntimeError, match="misaligned"):
        inspection.inspect_samples(path, digest, tokenizer)


def test_cli_writes_diagnostics_without_raw_content_and_refuses_overwrite(tmp_path):
    path, digest = sample_file(tmp_path, [conversation("test answer")])
    source_report = tmp_path / "source_report.json"
    source_report.write_text(json.dumps({"source": {"sha256": "c" * 64},
                                        "sample_files": {"random": {"sha256": digest}}}), encoding="utf-8")
    output = tmp_path / "report"
    command = [sys.executable, str(SCRIPT), "--source-report", str(source_report),
               "--samples", str(path), "--report-dir", str(output),
               "--tokenizer-dir", str(ROOT / "assets/minimind_tokenizer")]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    details = (output / "records.jsonl").read_text(encoding="utf-8")
    assert report["overall"]["records"] == 1
    assert report["scope"]["quality_acceptance_claimed"] is False
    assert report["input"]["full_source_rechecked_by_this_entry"] is False
    assert report["records_file"]["sha256"] == hashlib.sha256(details.encode()).hexdigest()
    assert '"record":' not in details and '"content":' not in details
    second = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert second.returncode != 0
    assert "fresh report directory" in second.stderr
