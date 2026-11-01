"""Publication-only checks; no remote requests or frozen training edits."""

import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("sft_publication", SCRIPTS / "publish_sft_bundle.py")
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def test_public_snapshot_redacts_nested_workstation_paths_without_changing_hashes():
    source = {"path": "E:/private/checkpoint.pt", "hash": "abc", "nested": [
        {"path": "D:\\pythonlearning\\data.jsonl", "records": 3}],
        "source_files": {"D:/private/file": "abc"}, "encoding": {"max_length": 768}}
    result = publication.public_snapshot(source)
    assert result["path"] == "<local artifact path omitted>"
    assert result["nested"][0] == {"path": "<local artifact path omitted>", "records": 3}
    assert result["hash"] == "abc"
    assert result["encoding"] == {"max_length": 768}
    assert "source_files" not in result
    assert source["path"] == "E:/private/checkpoint.pt"


def test_public_cards_keep_negative_results_and_remove_obsolete_draft_status():
    repo = "yanpeng051821/minimind-dense-64m-mini-sft"
    english = publication.publication_card("README.md", repo)
    chinese = publication.publication_card("README_zh.md", repo)
    for card in (english, chinese):
        assert repo in card
        assert "2.3954561617" in card
        assert "1.7210819774" in card
        assert "README" in card
        assert "license:" not in card
        assert publication.EXPECTED_STATE in card
    assert "0 confirmed pass, 31 fail, 1 uncertain" in english
    assert "local draft only" not in english
    assert "尚未上传新HF仓库" not in chinese
