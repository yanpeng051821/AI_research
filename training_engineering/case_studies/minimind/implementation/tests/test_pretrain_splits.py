import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pretrain_split_builder", ROOT / "scripts/build_pretrain_splits.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def split_inputs(tmp_path, monkeypatch):
    base = tmp_path / "candidates"
    analysis = base / "analysis"
    analysis.mkdir(parents=True)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    class CharacterTokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text, add_special_tokens, truncation):
            assert not add_special_tokens and not truncation
            return {"input_ids": list(range(len(text)))}

    monkeypatch.setattr(
        builder.AutoTokenizer, "from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    records = [
        {"source_line": n * 10, "text": chr(96 + n) * n}
        for n in range(1, 8)
    ] + [{"source_line": 80, "text": "x" * 600}]
    builder.write_jsonl(base / "candidates.jsonl", records)
    candidate_hash = builder.sha256_file(base / "candidates.jsonl")
    source_manifest = {
        "source_repo": "fixture/corpus",
        "source_revision": "fixed-revision",
        "source_path": "raw/source.jsonl",
        "source_sha256": "fixture-source-sha256",
        "candidate_sha256": candidate_hash,
        "record_count": len(records),
    }
    (base / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    stats = [
        {"source_line": r["source_line"], "token_count": len(r["text"])}
        for r in records
    ]
    builder.write_jsonl(analysis / "retained_rows.jsonl", stats[:-1])
    builder.write_jsonl(analysis / "length_filtered_rows.jsonl", stats[-1:])
    report = {
        "max_length": 512,
        "content_budget": 510,
        "sample_count": 8,
        "retained_count": 7,
        "length_filtered_count": 1,
        "total_tokens": sum(len(r["text"]) for r in records),
    }
    (analysis / "report.json").write_text(json.dumps(report), encoding="utf-8")
    reviews = []
    for record, decision, allowed in (
        (records[0], "keep", "train_only"),
        (records[1], "exclude", "exclude_all"),
    ):
        reviews.append({
            "source_line": record["source_line"],
            "source_repo": source_manifest["source_repo"],
            "source_revision": source_manifest["source_revision"],
            "source_file": "source.jsonl",
            "candidate_sha256": candidate_hash,
            "text_sha256": builder.text_sha256(record["text"]),
            "decision": decision,
            "allowed_split": allowed,
            "reason_code": "fixture_review",
        })
    builder.write_jsonl(analysis / "review_decisions.jsonl", reviews)
    return base, tokenizer_dir, records


def test_partition_preserves_text_and_reviews_and_is_byte_reproducible(split_inputs, tmp_path):
    base, tokenizer_dir, original = split_inputs
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = builder.build_splits(base, first, tokenizer_dir, validation_size=2)
    builder.build_splits(base, second, tokenizer_dir, validation_size=2)
    train = builder.read_records(first / "train.jsonl")
    validation = builder.read_records(first / "validation.jsonl")
    excluded = builder.read_records(first / "exclusions.jsonl")
    a = {r["source_line"] for r in train}
    b = {r["source_line"] for r in validation}
    c = {r["source_line"] for r in excluded}
    assert len(train) == 4 and len(validation) == 2
    assert c == {20, 80}
    assert 10 in a and 10 not in b
    assert not a & b and not a & c and not b & c
    assert a | b | c == {r["source_line"] for r in original}
    expected = {r["source_line"]: r["text"] for r in original}
    for row in train + validation:
        assert row["text"] == expected[row["source_line"]]
    assert manifest["train"]["valid_tokens_after_shift"] == sum(
        len(r["text"]) + 1 for r in train
    )
    for filename in ("train.jsonl", "validation.jsonl", "exclusions.jsonl", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_changed_candidate_fails_before_creating_output(split_inputs, tmp_path):
    base, tokenizer_dir, _ = split_inputs
    with (base / "candidates.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="candidate SHA256"):
        builder.build_splits(base, output, tokenizer_dir, validation_size=2)
    assert not output.exists()


def test_stale_review_identity_is_rejected(split_inputs, tmp_path):
    base, tokenizer_dir, _ = split_inputs
    path = base / "analysis/review_decisions.jsonl"
    reviews = builder.read_records(path)
    reviews[0]["text_sha256"] = "wrong-record-identity"
    builder.write_jsonl(path, reviews)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="review text SHA256"):
        builder.build_splits(base, output, tokenizer_dir, validation_size=2)
    assert not output.exists()


def test_stale_length_statistics_are_rejected(split_inputs, tmp_path):
    base, tokenizer_dir, _ = split_inputs
    path = base / "analysis/retained_rows.jsonl"
    stats = builder.read_records(path)
    stats[0]["token_count"] += 1
    builder.write_jsonl(path, stats)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="stale token count"):
        builder.build_splits(base, output, tokenizer_dir, validation_size=2)
    assert not output.exists()


def test_existing_output_is_not_overwritten(split_inputs, tmp_path):
    base, tokenizer_dir, _ = split_inputs
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "notes.txt"
    sentinel.write_text("keep existing work", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new output directory"):
        builder.build_splits(base, output, tokenizer_dir, validation_size=2)
    assert sentinel.read_text(encoding="utf-8") == "keep existing work"


def test_validation_cannot_consume_reviewed_training_records(split_inputs, tmp_path):
    base, tokenizer_dir, _ = split_inputs
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="exceeds the pool"):
        builder.build_splits(base, output, tokenizer_dir, validation_size=6)
    assert not output.exists()
