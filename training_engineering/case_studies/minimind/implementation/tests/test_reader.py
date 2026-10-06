import pytest

from minimind_lab.data.reader import parse_json_line, iter_jsonl



def test_parse_blank_line():
    result = parse_json_line(" \n", 1)

    assert result == {
        "line_number": 1,
        "record": None,
        "error_code": "blank_line"
    }


def test_parse_invalid_json():
    result = parse_json_line('{"text":}',1)

    assert result == {
        "line_number": 1,
        "record": None,
        "error_code": "invalid_json"
    }


def test_parse_valid_json():
    result = parse_json_line('{"text": "hello"}', 1)

    assert result == {
        "line_number": 1,
        "record": {"text": "hello"},
        "error_code": None
    }

def test_iter_jsonl_preserves_physical_lines(tmp_path):
    # 创建临时测试文件
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"text": "first"}\n'
        '\n'
        '{"text":}\n'
        '{"test": "last"}',
        encoding="utf-8"
    )

    result = list(iter_jsonl(path))

    assert len(result) == 4
    assert result == [
        {"source_path": str(path),"raw_line": '{"text": "first"}\n',"line_number": 1, "record": {"text": "first"}, "error_code": None},
        {"source_path": str(path),"raw_line": '\n',"line_number": 2,"record": None, "error_code": "blank_line"},
        {"source_path": str(path),"raw_line": '{"text":}\n',"line_number": 3, "record": None, "error_code": "invalid_json"},
        {"source_path": str(path),"raw_line": '{"test": "last"}',"line_number": 4, "record": {"test": "last"}, "error_code": None}
    ]



def test_iter_jsonl_raises_when_file_does_not_exist(tmp_path):
    missing_path = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(missing_path))



