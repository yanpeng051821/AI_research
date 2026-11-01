import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("apply_sft_review_decisions", ROOT / "scripts/apply_sft_review_decisions.py")
review = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(review)


def fixture(tmp_path):
    rows = [b'{"id":1}\r\n', b'{"id":2}\n']
    source = tmp_path / "candidate.jsonl"
    source.write_bytes(b"".join(rows))
    ledger = [{"input_line": 1, "source_line": 10, "source_line_sha256": hashlib.sha256(rows[0]).hexdigest(),
               "action": "keep", "retained_output_line": 1, "retained_source_line": 10},
              {"input_line": 2, "source_line": 20, "source_line_sha256": hashlib.sha256(rows[0]).hexdigest(),
               "action": "exclude_exact_duplicate", "retained_output_line": 1, "retained_source_line": 10},
              {"input_line": 3, "source_line": 30, "source_line_sha256": hashlib.sha256(rows[1]).hexdigest(),
               "action": "keep", "retained_output_line": 2, "retained_source_line": 30}]
    raw_ledger = "".join(json.dumps(row) + "\n" for row in ledger).encode()
    lineage = tmp_path / "lineage.jsonl"
    lineage.write_bytes(raw_ledger)
    source_spec = {"path": str(source), "records": 2, "bytes": source.stat().st_size,
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    ledger_spec = {"path": str(lineage), "records": 3, "bytes": len(raw_ledger),
                   "sha256": hashlib.sha256(raw_ledger).hexdigest()}
    decision = {"status": "user_confirmed", "action": "exclude", "source_sha256": "a" * 64,
                "source_line": 30, "source_line_sha256": hashlib.sha256(rows[1]).hexdigest(), "reason": "reviewed"}
    return rows, {"outputs": {"unique": source_spec, "lineage": ledger_spec}}, decision


def test_only_exact_confirmed_record_is_excluded_and_lineage_remains_usable(tmp_path):
    rows, reference, decision = fixture(tmp_path)
    result = review.apply_decisions(reference, {30: decision}, tmp_path / "out")
    assert Path(result["outputs"]["candidate"]["path"]).read_bytes() == rows[0]
    assert Path(result["outputs"]["excluded"]["path"]).read_bytes() == rows[1]
    kept = list(review.iter_candidate_rows(Path(result["outputs"]["candidate"]["path"]),
                Path(result["outputs"]["lineage"]["path"]),
                result["outputs"]["candidate"], result["outputs"]["lineage"]))
    assert len(kept) == 1 and kept[0][2]["source_line"] == 10
    assert result["applied_decisions"][0]["candidate_line"] == 2


@pytest.mark.parametrize("change", [{"status": "assistant_proposal"}, {"source_sha256": "b" * 64},
                                    {"action": "revise"}, {"source_line": True}])
def test_unconfirmed_or_incompatible_decisions_are_rejected(tmp_path, change):
    _, _, decision = fixture(tmp_path)
    decision.update(change)
    with pytest.raises(ValueError, match="only confirmed"):
        review.load_decisions(json.dumps(decision).encode(), "a" * 64)


def test_wrong_content_hash_and_unmatched_decisions_do_not_publish(tmp_path):
    _, reference, decision = fixture(tmp_path)
    decision["source_line_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash"):
        review.apply_decisions(reference, {30: decision}, tmp_path / "wrong")
    assert not list((tmp_path / "wrong").glob("*.jsonl"))
    with pytest.raises(ValueError, match="did not match"):
        review.apply_decisions(reference, {100: decision}, tmp_path / "missing")
    assert not list((tmp_path / "missing").glob("*.jsonl"))


def test_corrupted_ledger_fails_even_when_candidate_matches(tmp_path):
    _, reference, decision = fixture(tmp_path)
    reference["outputs"]["lineage"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage identity"):
        review.apply_decisions(reference, {30: decision}, tmp_path / "out")


def test_duplicate_decisions_are_rejected(tmp_path):
    _, _, decision = fixture(tmp_path)
    payload = (json.dumps(decision) + "\n" + json.dumps(decision)).encode()
    with pytest.raises(ValueError, match="duplicate review"):
        review.load_decisions(payload, "a" * 64)
