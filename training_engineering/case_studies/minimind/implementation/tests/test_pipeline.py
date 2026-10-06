
import pytest, json

from minimind_lab.data.pipeline import audit_parsed_line,iter_audited_jsonl, audit_jsonl
from minimind_lab.data import pipeline

def test_pretrain_return_full_dict():
    parse_line =  {
        "source_path": "/tmp/test.jsonl",
        "raw_line": '{\"text\": \"hello\"}\n',
        "line_number": 1,
        "record": {"text": "hello"},
        "error_code": None,
            }  
        
    result = audit_parsed_line(parse_line, "pretrain")
        
    assert result == {
            "source_path": "/tmp/test.jsonl",
            "line_number": 1,
            "raw_line": '{\"text\": \"hello\"}\n',
            "record": {"text": "hello"},
            "issues": [],
            "accepted": True,
        }

    
def test_pretrain_missing_text():
    parse_line =  {
        "source_path": "/tmp/test.jsonl",
        "raw_line": '{\"source\": \"wiki\"}',
        "line_number": 1,
        "record": {"source": "wiki"},
        "error_code": None,
            }  

    result = audit_parsed_line(parse_line, "pretrain")

    assert result == {
        "source_path": "/tmp/test.jsonl",
        "line_number": 1,
        "raw_line": '{\"source\": \"wiki\"}',
        "record": {"source": "wiki"},
        "issues": ["missing_text"],
        "accepted": False,
    }


def test_sft_return_full_dict():
    parse_line =  {
        "source_path": "/tmp/test.jsonl",
        "raw_line": '{\"conversations\":[{\"role\":\"user\", \"content\": \"hello\"}, {\"role\":\"assistant\", \"content\": \"hello\"}]}',
        "line_number": 1,
        "record": {"conversations":[{"role":"user", "content": "hello"}, {"role":"assistant", "content": "hello"}]},
        "error_code": None,
        }  
    
    result = audit_parsed_line(parse_line, "sft")
    
    assert result == {
        "source_path": "/tmp/test.jsonl",
        "line_number": 1,
        "raw_line": '{\"conversations\":[{\"role\":\"user\", \"content\": \"hello\"}, {\"role\":\"assistant\", \"content\": \"hello\"}]}',
        "record": {"conversations":[{"role":"user", "content": "hello"}, {"role":"assistant", "content": "hello"}]},
        "issues": [],
        "accepted": True,
    }

def test_unknown_dataset_type():
    parse_line =  {
        "source_path": "/tmp/test.jsonl",
        "raw_line": '{\"conversations\":[{\"role\":\"user\", \"content\": \"hello\"}, {\"role\":\"assistant\", \"content\": \"hello\"}]}',
        "line_number": 1,
        "record": {"conversations":[{"role":"user", "content": "hello"}, {"role":"assistant", "content": "hello"}]},
        "error_code": None,
        }  
    
    with pytest.raises(ValueError, match="unsupported dataset type: unknown"):
        audit_parsed_line(parse_line, "unknown")


def validator_must_not_run(record):
    raise AssertionError("validator should not run")

def test_validator_not_run_for_failed_parse(monkeypatch):
    parsed_line = {
        "source_path": "train.jsonl",
        "line_number": 2,
        "raw_line": "\n",
        "record": None,
        "error_code": "blank_line",
    }
    
    monkeypatch.setitem(pipeline.VALIDATORS, "pretrain", validator_must_not_run)

    result = audit_parsed_line(parsed_line, "pretrain")

    assert result == {
        "source_path": "train.jsonl",
        "line_number": 2,
        "raw_line": "\n",
        "record": None,
        "issues": ["blank_line"],
        "accepted": False,
    }

def test_integration_parse_and_audit(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"text": "valid"}\n'
        '{"source": "missing text"}\n'
        '\n'
        '{"text": }',
        encoding="utf-8"
    )

    results = list(iter_audited_jsonl(path, "pretrain"))

    assert [item["line_number"] for item in results] == [1, 2, 3, 4]

    assert [item["accepted"] for item in results] == [
        True,
        False,
        False,
        False,
    ]

    assert [item["issues"] for item in results] == [
        [],
        ["missing_text"],
        ["blank_line"],
        ["invalid_json"],
    ]



def test_audit_jsonl_writes_expected_artifacts(tmp_path):
    input_path = tmp_path / "test.jsonl"
    input_path.write_text(
        '{"text": "valid"}\n'
        '{"source": "missing text"}\n'
        '\n'
        '{"text": }',
        encoding="utf-8"
    )

    output_path = tmp_path / "output"
    report = audit_jsonl(input_path, "pretrain", output_path)

    assert report == {
        "source_path": str(input_path),
        "dataset_type": "pretrain",
        "total_lines": 4,
        "accepted_lines": 1,
        "rejected_lines": 3,
        "errors_by_code": {
            "blank_line": 1,
            "invalid_json": 1,
            "missing_text": 1,
        },
    }
    report_path = output_path / "report.json"
    loaded_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert loaded_report == report

    accepted_records = [json.loads(line) for line in (output_path / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert accepted_records == [{"text": "valid"}]

    rejected_records = [json.loads(line) for line in (output_path / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rejected_records) == 3
    assert [record["line_number"] for record in rejected_records] == [2, 3, 4]
    assert [record["issues"] for record in rejected_records] == [
        ["missing_text"],
        ["blank_line"],
        ["invalid_json"],
    ]


def test_audit_jsonl_rejects_unknown_type_before_creating_output(tmp_path):
    input_path = tmp_path / "test.jsonl"
    input_path.write_text('{"text": "valid"}\n', encoding="utf-8")

    output_dir = tmp_path / "output"
    
    with pytest.raises(ValueError, match="unsupported dataset type: unknown"):
        audit_jsonl(input_path, "unknown", output_dir)
    
    assert not output_dir.exists()


def test_audit_jsonl_overwrites_previous_outputs(tmp_path):
    input_path = tmp_path / "test.jsonl"
    input_path.write_text('{"source_wiki": "invalid"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    report_path = output_dir / "report.json"
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"

    first_report = audit_jsonl(input_path,"pretrain", output_dir)
    assert first_report["total_lines"] == 1
    assert first_report["accepted_lines"] == 0
    assert first_report["rejected_lines"] == 1

    first_rejected_lines = (output_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first_rejected_lines) == 1

    input_path.write_text('{"text": "hello"}\n', encoding='utf-8',)
    second_report = audit_jsonl(input_path, "pretrain", output_dir)
    assert second_report["total_lines"] == 1
    assert second_report["accepted_lines"] == 1
    assert second_report["rejected_lines"] == 0

    accepted_lines = accepted_path.read_text(encoding='utf-8').splitlines()
    assert len(accepted_lines) == 1
    
    accepted_record = json.loads(accepted_lines[0])
    assert accepted_record == {"text": "hello"}
    assert rejected_path.read_text(encoding='utf-8') == ""

    assert second_report["total_lines"] == 1
    assert second_report["accepted_lines"] == 1
    assert second_report["rejected_lines"] == 0
    assert second_report["errors_by_code"] == {}

    saved_report = json.loads(
        report_path.read_text(encoding='utf-8')
    )
    assert saved_report == second_report



def test_audit_jsonl_missing_input_does_not_create_output(tmp_path):
    input_path = tmp_path / "missing.jsonl"
    output_dir = tmp_path / "output"

    with pytest.raises(FileNotFoundError):
        audit_jsonl(input_path, "pretrain", output_dir)

    assert not output_dir.exists()