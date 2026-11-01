import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_sft_length_subset", ROOT / "scripts/prepare_sft_length_subset.py")
selection = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(selection)


def inputs(tmp_path):
    rows = [b'{"conversations": [{"content": "a"}]}\r\n',
            b'{"conversations": [{"content": "b"}]}\n',
            b'{"conversations": [{"content": "c"}]}']
    raw = b"".join(rows)
    details = b"".join((json.dumps({"source_line": i, "issues": [],
                                    "source_line_sha256": hashlib.sha256(row).hexdigest(),
                                    "input_tokens": length, "targets_after_shift": 2}) + "\n").encode()
                       for i, (row, length) in enumerate(zip(rows, [767, 769, 768]), 1))
    source, detail = tmp_path / "source.jsonl", tmp_path / "details.jsonl"
    source.write_bytes(raw)
    detail.write_bytes(details)
    return rows, source, detail, {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_select_preserves_bytes_order_and_inclusive_length_boundary(tmp_path):
    rows, source, detail, identity = inputs(tmp_path)
    outputs = selection.select_records(source, detail, tmp_path / "out", 768,
                                       identity, hashlib.sha256(detail.read_bytes()).hexdigest(), 3)
    assert Path(outputs["length_eligible"]["path"]).read_bytes() == rows[0] + rows[2]
    assert Path(outputs["overlength_excluded"]["path"]).read_bytes() == rows[1]
    assert outputs["length_eligible"]["records"] == 2
    assert outputs["length_eligible"]["targets_after_shift"] == 4
    assert source.read_bytes() == b"".join(rows)


def test_wrong_identity_does_not_publish_final_files(tmp_path):
    _, source, detail, identity = inputs(tmp_path)
    with pytest.raises(ValueError, match="final identity"):
        selection.select_records(source, detail, tmp_path / "out", 768, identity, "0" * 64, 3)
    assert not list((tmp_path / "out").glob("*.jsonl"))


def test_misaligned_details_rejected(tmp_path):
    _, source, detail, identity = inputs(tmp_path)
    detail.write_bytes(b"\n".join(detail.read_bytes().splitlines()[1:]) + b"\n")
    with pytest.raises(ValueError, match="source/detail mismatch"):
        selection.select_records(source, detail, tmp_path / "out", 768, identity,
                                 hashlib.sha256(detail.read_bytes()).hexdigest(), 3)


def test_existing_output_directory_is_not_overwritten(tmp_path):
    _, source, detail, identity = inputs(tmp_path)
    with pytest.raises(FileExistsError):
        selection.select_records(source, detail, tmp_path, 768, identity, "unused", 3)
