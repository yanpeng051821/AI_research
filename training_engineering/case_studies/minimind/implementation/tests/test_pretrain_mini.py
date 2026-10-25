import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pretrain_mini_builder", ROOT / "scripts/build_pretrain_mini.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_collect_text_groups_preserves_all_duplicate_source_lines(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": text}) for text in ("same", "other", "same")) + "\n",
        encoding="utf-8",
    )
    actual = builder.collect_text_groups(path)
    assert actual == {
        hashlib.sha256(b"same").hexdigest(): [1, 3],
        hashlib.sha256(b"other").hexdigest(): [2],
    }


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("not-json\n", "invalid_json"),
        ("\n", "blank_line"),
        ("{}\n", "missing_text"),
        ('{"text": ""}\n', "empty_text"),
        ('{"text": "   "}\n', "empty_text"),
        ('{"text": 7}\n', "text_not_string"),
    ],
)
def test_collect_text_groups_stops_with_source_line_on_invalid_record(tmp_path, raw, error):
    path = tmp_path / "source.jsonl"
    path.write_text('{"text": "valid first record"}\n' + raw, encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        builder.collect_text_groups(path)
    assert error in str(raised.value)
    assert "2" in str(raised.value)


def write_reviews(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def review_record(text_hash, source_line, decision="keep", allowed_split="train_only"):
    return {
        "text_sha256": text_hash,
        "source_line": source_line,
        "decision": decision,
        "allowed_split": allowed_split,
    }


def test_load_review_constraints_uses_original_source_lines_and_group_identity(tmp_path):
    path = tmp_path / "reviews.jsonl"
    write_reviews(path, [
        review_record("excluded", 356, "exclude", "exclude_all"),
        review_record("train", 900),
    ])
    groups = {"excluded": [356, 700], "train": [900, 1000], "other": [42]}
    assert builder.load_review_constraints(path, groups) == ({"excluded"}, {"train"})


def test_load_review_constraints_accepts_empty_review(tmp_path):
    path = tmp_path / "reviews.jsonl"
    write_reviews(path, [])
    assert builder.load_review_constraints(path, {}) == (set(), set())


@pytest.mark.parametrize(
    "record",
    [
        review_record("unknown", 356),
        review_record("known", 1),
        review_record("known", 356, "keep", "exclude_all"),
        review_record("known", 356, "exclude", "train_only"),
        review_record("known", 356, "unknown", "train_only"),
    ],
    ids=["unknown-hash", "wrong-source-line", "keep-excluded", "exclude-training", "unknown-decision"],
)
def test_load_review_constraints_rejects_mismatches(tmp_path, record):
    path = tmp_path / "reviews.jsonl"
    write_reviews(path, [record])
    with pytest.raises(ValueError):
        builder.load_review_constraints(path, {"known": [356]})


def test_load_review_constraints_rejects_conflict_across_duplicate_rows(tmp_path):
    path = tmp_path / "reviews.jsonl"
    write_reviews(path, [
        review_record("same", 356, "exclude", "exclude_all"),
        review_record("same", 700),
    ])
    with pytest.raises(ValueError, match="overlap"):
        builder.load_review_constraints(path, {"same": [356, 700]})


def test_load_review_constraints_rejects_parse_error(tmp_path):
    path = tmp_path / "reviews.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_json"):
        builder.load_review_constraints(path, {})


def test_validation_selection_is_deterministic_and_preserves_global_rng():
    import random

    groups = {name: [index] for index, name in enumerate("abcdef", 1)}
    before = random.getstate()
    selected = builder.select_validation_groups(groups, {"a"}, {"b"}, 2, 43)
    assert random.getstate() == before
    reversed_groups = dict(reversed(list(groups.items())))
    assert selected == builder.select_validation_groups(reversed_groups, {"a"}, {"b"}, 2, 43)
    assert len(selected) == 2
    assert selected <= {"c", "d", "e", "f"}


def test_validation_can_use_entire_pool_when_train_only_group_remains():
    assert builder.select_validation_groups(
        {"excluded": [1], "train": [2], "validation": [3, 4]},
        {"excluded"}, {"train"}, 1, 43,
    ) == {"validation"}


@pytest.mark.parametrize(
    ("excluded", "train_only", "size"),
    [
        ({"unknown"}, set(), 1),
        (set(), {"unknown"}, 1),
        ({"a"}, {"a"}, 1),
        (set(), set(), 0),
        (set(), set(), -1),
        (set(), set(), 4),
        (set(), set(), 3),
        ({"a"}, {"b", "c"}, 1),
    ],
    ids=["unknown-exclusion", "unknown-train-only", "overlap", "zero", "negative",
         "too-many", "no-training-left", "empty-pool"],
)
def test_validation_selection_rejects_invalid_constraints_and_sizes(excluded, train_only, size):
    with pytest.raises(ValueError):
        builder.select_validation_groups({"a": [1], "b": [2], "c": [3]}, excluded, train_only, size, 43)


def test_write_split_records_preserves_text_source_lines_and_duplicate_groups(tmp_path):
    source = tmp_path / "source.jsonl"
    texts = ["train", "validation", "excluded", "train", "validation", "excluded", "\u4f60\u597d\nworld"]
    source.write_text(
        "\n".join(json.dumps({"text": text}) for text in texts),
        encoding="utf-8",
    )
    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    output = tmp_path / "split"
    counts = builder.write_split_records(source, output, {digest("excluded")}, {digest("validation")})
    assert counts == {"train": 3, "validation": 2, "excluded": 2}
    assert {path.name for path in output.iterdir()} == {"train.jsonl", "validation.jsonl", "exclusions.jsonl"}
    for filename, source_lines in {
        "train.jsonl": [1, 4, 7],
        "validation.jsonl": [2, 5],
        "exclusions.jsonl": [3, 6],
    }.items():
        rows = [json.loads(line) for line in (output / filename).read_text(encoding="utf-8").splitlines()]
        assert rows == [{"source_line": number, "text": texts[number - 1]} for number in source_lines]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("not-json\n", "invalid_json"),
        ("\n", "blank_line"),
        ("{}\n", "missing_text"),
        ('{"text": ""}\n', "empty_text"),
        ('{"text": "   "}\n', "empty_text"),
        ('{"text": 7}\n', "text_not_string"),
    ],
)
def test_write_split_records_rejects_bad_input_without_success_manifest(tmp_path, raw, error):
    source = tmp_path / "source.jsonl"
    source.write_text('{"text": "valid"}\n' + raw, encoding="utf-8")
    output = tmp_path / "split"
    with pytest.raises(ValueError) as raised:
        builder.write_split_records(source, output, set(), set())
    assert error in str(raised.value)
    assert "2" in str(raised.value)
    assert not (output / "manifest.json").exists()


def test_write_split_records_rejects_overlapping_groups_before_creating_directory(tmp_path):
    output = tmp_path / "split"
    with pytest.raises(ValueError, match="overlap"):
        builder.write_split_records(tmp_path / "unused.jsonl", output, {"same"}, {"same"})
    assert not output.exists()


def test_write_split_records_does_not_touch_existing_output(tmp_path):
    output = tmp_path / "split"
    output.mkdir()
    sentinel = output / "train.jsonl"
    sentinel.write_bytes(b"existing data\n")
    with pytest.raises(FileExistsError):
        builder.write_split_records(tmp_path / "unused.jsonl", output, set(), set())
    assert sentinel.read_bytes() == b"existing data\n"
    assert len(list(output.iterdir())) == 1


@pytest.fixture
def publish_inputs(tmp_path):
    source = tmp_path / "source.jsonl"
    texts = ["train-only", "validation", "validation", "excluded"]
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    candidate = tmp_path / "candidates.jsonl"
    write_reviews(candidate, [{"source_line": i, "text": text} for i, text in enumerate(texts, 1)])
    anchor = {
        "source_path": source.name, "source_repo": "test/repo", "source_revision": "test-revision",
        "source_sha256": builder.sha256_file(source), "source_total_lines": 4,
        "candidate_sha256": builder.sha256_file(candidate), "record_count": 4,
    }
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(anchor), encoding="utf-8")
    review = tmp_path / "reviews.jsonl"
    reviews = []
    for line, decision, split in [(1, "keep", "train_only"), (4, "exclude", "exclude_all")]:
        record = review_record(hashlib.sha256(texts[line - 1].encode()).hexdigest(), line, decision, split)
        record.update({
            "source_repo": anchor["source_repo"], "source_revision": anchor["source_revision"],
            "source_file": source.name, "candidate_sha256": anchor["candidate_sha256"],
        })
        reviews.append(record)
    write_reviews(review, reviews)
    return source, review, tmp_path / "output", manifest, candidate


def test_publish_full_pipeline_counts_records_not_groups(publish_inputs):
    source, review, output, anchor, candidate = publish_inputs
    manifest = builder.publish_mini_dataset(source, review, output, anchor, candidate, 1, 43)
    assert manifest["counts"] == {"train": 1, "validation": 2, "excluded": 1}
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["status"] == "text_split_complete_encoding_not_checked"
    for info in manifest["files"].values():
        assert info["sha256"] == builder.sha256_file(output / info["file"])
        assert b"\r\n" not in (output / info["file"]).read_bytes()
    assert not (output / "manifest.pending.json").exists()


@pytest.mark.parametrize("which", ["source", "candidate", "review"])
def test_publish_rejects_changed_inputs_before_output(publish_inputs, which):
    source, review, output, anchor, candidate = publish_inputs
    if which == "review":
        records = [json.loads(line) for line in review.read_text(encoding="utf-8").splitlines()]
        records[0]["source_revision"] = "wrong"
        write_reviews(review, records)
    else:
        path = source if which == "source" else candidate
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    with pytest.raises(ValueError):
        builder.publish_mini_dataset(source, review, output, anchor, candidate, 1, 43)
    assert not output.exists()


def test_builder_detects_wrong_writer_count(publish_inputs, monkeypatch):
    source, review, output, _, _ = publish_inputs
    monkeypatch.setattr(builder, "write_split_records", lambda *args: {"train": 0, "validation": 0, "excluded": 0})
    with pytest.raises(ValueError, match="mismatch"):
        builder.build_mini_dataset(source, review, output, 1, 43)


def test_publish_does_not_mark_changed_during_build_as_complete(publish_inputs, monkeypatch):
    source, review, output, anchor, candidate = publish_inputs
    original = builder.build_mini_dataset

    def changing_build(*args):
        result = original(*args)
        with review.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(builder, "build_mini_dataset", changing_build)
    with pytest.raises(ValueError):
        builder.publish_mini_dataset(source, review, output, anchor, candidate, 1, 43)
    assert output.exists()
    assert not (output / "manifest.json").exists()


def test_build_rejects_existing_directory_before_reading_sources(tmp_path):
    with pytest.raises(FileExistsError):
        builder.build_mini_dataset(tmp_path / "missing", tmp_path / "missing-review", tmp_path, 1, 43)
