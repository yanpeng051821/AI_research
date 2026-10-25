import importlib.util
import json
from pathlib import Path

import pytest
from transformers import BatchEncoding


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("encoding_audit", ROOT / "scripts/audit_pretrain_encoding.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    truncation_side = "right"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, padding=False):
        def encode(value):
            ids = [ord(char) + 3 for char in value]
            return ids[:max_length] if truncation else ids

        return BatchEncoding({"input_ids": [encode(value) for value in text] if isinstance(text, list) else encode(text)})


def make_split(root, train_text="abc123", validation_text="abc456"):
    root.mkdir()
    files = {}
    for name, filename, rows in [
        ("train", "train.jsonl", [{"source_line": 1, "text": train_text}]),
        ("validation", "validation.jsonl", [{"source_line": 2, "text": validation_text}]),
        ("excluded", "exclusions.jsonl", []),
    ]:
        path = root / filename
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        files[name] = {"file": filename, "sha256": audit.sha256_file(path)}
    manifest = {
        "status": "text_split_complete_encoding_not_checked",
        "counts": {"train": 1, "validation": 1, "excluded": 0}, "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_audit_detects_different_texts_with_identical_truncated_inputs(tmp_path):
    split = make_split(tmp_path / "split")
    result = audit.audit_encoding(split, CharacterTokenizer(), 5, batch_size=1)
    assert result["status"] == "blocked_exact_input_overlap"
    assert result["train_records_with_exact_validation_input"] == 1
    assert result["overlap_examples"][0]["train_source_line"] == 1
    assert result["overlap_examples"][0]["validation_source_line"] == 2
    stats = result["statistics"]["train"]
    assert stats["body_tokens"] == 6
    assert stats["retained_body_tokens"] == 3
    assert stats["dropped_tail_tokens"] == 3
    assert stats["active_input_positions"] == 5
    assert stats["valid_causal_targets"] == 4


def test_audit_longer_length_retains_distinct_inputs_without_changing_files(tmp_path):
    split = make_split(tmp_path / "split")
    before = {p.name: p.read_bytes() for p in split.iterdir()}
    result = audit.audit_encoding(split, CharacterTokenizer(), 20)
    assert result["status"] == "passed"
    assert result["statistics"]["train"]["truncated_records"] == 0
    assert result["statistics"]["train"]["valid_causal_targets"] == 7
    assert {p.name: p.read_bytes() for p in split.iterdir()} == before


def test_audit_refuses_changed_split_file(tmp_path):
    split = make_split(tmp_path / "split")
    with (split / "train.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.audit_encoding(split, CharacterTokenizer(), 5)


def test_audit_refuses_changed_counts(tmp_path):
    split = make_split(tmp_path / "split")
    path = split / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["counts"]["train"] = 2
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        audit.audit_encoding(split, CharacterTokenizer(), 5)
