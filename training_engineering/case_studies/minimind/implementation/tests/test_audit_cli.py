import json

from minimind_lab.audit_cli import main


def test_main_returns_zero_for_valid_data(tmp_path, capsys):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "output"

    input_path.write_text('{"text":"nihao"}', encoding='utf-8')


    exit_code = main([
        str(input_path),
        "--dataset-type","pretrain",
        "--output-dir", str(output_dir)
    ])

    assert exit_code == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert report["accepted_lines"] == 1
    assert report["rejected_lines"] == 0
    assert captured.err == ""


def test_main_returns_one_when_records_are_rejected(tmp_path, capsys):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "output"

    input_path.write_text(
        '{"text":"nihao"}\n'
        '{"source": "wiki"}'
        , encoding='utf-8')


    exit_code = main([
        str(input_path),
        "--dataset-type","pretrain",
        "--output-dir", str(output_dir)
    ])

    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 1
    assert report["accepted_lines"] == 1
    assert report["rejected_lines"] == 1
    assert captured.err == ""


def test_main_returns_two_for_missing_input(tmp_path, capsys):
    input_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "output"

    exit_code = main([
        str(input_path),
        "--dataset-type","pretrain",
        "--output-dir", str(output_dir)
    ])

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "Error:" in captured.err
    assert str(input_path) in captured.err  