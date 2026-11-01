import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inspect_sft_full_encoding.py"
spec = importlib.util.spec_from_file_location("inspect_sft_full_encoding", SCRIPT)
inspection = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(inspection)


def conversation(answer="answer"):
    return {"conversations": [{"role": "user", "content": "question"},
                              {"role": "assistant", "content": answer}]}


def test_raw_record_diagnostics_and_source_hash(monkeypatch):
    tokenizer = SimpleNamespace(bos_token="<|im_start|>", eos_token="<|im_end|>")
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [-100, 2]})
    raw = (json.dumps(conversation()) + "\n").encode()
    row = inspection.inspect_raw_record(12, raw, tokenizer)
    assert row["source_line_sha256"] == hashlib.sha256(raw).hexdigest()
    assert row["stratum"] == "plain" and row["targets_after_shift"] == 1
    assert inspection.inspect_raw_record(13, b"bad JSON\n", tokenizer)["issues"] == ["invalid_utf8_or_json"]
    assert inspection.inspect_raw_record(14, b"{}\n", tokenizer)["issues"] == ["invalid_structure"]
    broken = conversation()
    broken["conversations"][-1]["tool_calls"] = "bad JSON"
    assert inspection.inspect_raw_record(15, json.dumps(broken).encode(), tokenizer)["issues"] == ["field_conversion_error"]
    marker = conversation()
    marker["conversations"][0]["content"] = "quoted <|im_start|>assistant"
    row = inspection.inspect_raw_record(16, json.dumps(marker).encode(), tokenizer)
    assert row["issues"] == ["potential_message_boundary_literal"]


def test_raw_record_preserves_shift_and_program_error_checks(monkeypatch):
    tokenizer = SimpleNamespace(bos_token="<|im_start|>", eos_token="<|im_end|>")
    raw = json.dumps(conversation()).encode()
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [1, -100]})
    assert inspection.inspect_raw_record(1, raw, tokenizer)["issues"] == ["zero_targets_after_shift"]
    monkeypatch.setattr(inspection, "encode_sft_conversation",
                        lambda *_: {"input_ids": [1, 2], "labels": [-100]})
    with pytest.raises(RuntimeError, match="misaligned"):
        inspection.inspect_raw_record(1, raw, tokenizer)


def test_compact_aggregation_matches_known_lengths():
    stats = inspection.Statistics()
    for length, targets, issues in [(4, 2, []), (6, 4, ["potential_message_boundary_literal"]),
                                    (None, None, ["invalid_structure"])]:
        stats.add({"input_tokens": length, "targets_after_shift": targets, "issues": issues})
    result = stats.summary()
    assert result["records"] == 3 and result["encoded_records"] == 2
    assert result["flagged_records"] == 2
    assert result["input_tokens"]["percentiles"]["p95"] == pytest.approx(5.9)
    assert result["supervised_fraction_of_shifted_positions"] == 0.75


def test_parallel_cli_keeps_order_reports_bad_rows_and_checks_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    raw = b"".join((json.dumps(conversation(f"answer {i}")) + "\n").encode() for i in range(2)) + b"bad JSON\n"
    source.write_bytes(raw)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"source": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                                     "counts": {"total_lines": 3}}), encoding="utf-8")
    report_dir, details_dir = tmp_path / "summary", tmp_path / "large_details"
    command = [sys.executable, str(SCRIPT), "--source", str(source), "--source-report", str(reference),
               "--report-dir", str(report_dir), "--details-dir", str(details_dir),
               "--tokenizer-dir", str(ROOT / "assets/minimind_tokenizer"), "--workers", "2", "--chunk-size", "1"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 2, result.stderr
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((report_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (details_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["source_line"] for row in rows] == [1, 2, 3]
    assert report["overall"]["encoded_records"] == 2
    assert report["overall"]["issues"] == {"invalid_utf8_or_json": 1}
    assert manifest["status"] == "completed" and manifest["diagnostic_status"] == "encoding_flags_found"
    assert source.read_bytes() == raw
    for info in report["files"].values():
        assert hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() == info["sha256"]
    again = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert again.returncode != 0 and "fresh output directory" in again.stderr
    source.write_bytes(raw + b"changed")
    command[command.index("--report-dir") + 1] = str(tmp_path / "wrong_summary")
    command[command.index("--details-dir") + 1] = str(tmp_path / "wrong_details")
    mismatch = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert mismatch.returncode != 0 and "source identity mismatch" in mismatch.stderr
    assert not (tmp_path / "wrong_summary").exists()
