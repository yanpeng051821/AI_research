"""Documentation-only updates must preserve model facts and generated outputs."""

import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent / "experiments/pretrain_mini_one_epoch_v1"


@pytest.fixture
def cards(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("update_model_card_languages")
    original = (ROOT / "exports/pretrain_mini_one_epoch_v1_public_v1/README.md")
    # The committed Chinese source has the same examples and facts, with publication placeholders.
    original = (original if original.exists() else EXPERIMENT / "release/MODEL_CARD_DRAFT.md")
    text = original.read_text(encoding="utf-8")
    english = (EXPERIMENT / "release/model_cards/README.md").read_text(encoding="utf-8")
    return module, text, english


def test_english_and_chinese_preserve_metadata_examples_and_identities(cards):
    module, original, english = cards
    module.check_translation(english, module.add_chinese_navigation(original), original)


def test_translation_rejects_modified_generation(cards):
    module, original, english = cards
    changed = english.replace("树叶在微风中轻轻摇曳。", "模型并未生成此句。")
    with pytest.raises(ValueError, match="verbatim"):
        module.check_translation(changed, module.add_chinese_navigation(original), original)


def test_manifest_keeps_all_non_card_files(cards):
    module, _, _ = cards
    weight = {"path": "model.safetensors", "bytes": 123, "sha256": "unchanged"}
    parent = {"files": [weight, {"path": "README.md", "bytes": 0, "sha256": "old"}]}
    result = module.new_manifest(parent, {"README.md": "English", "README_zh.md": "中文"}, "parent")
    assert next(item for item in result["files"] if item["path"] == "model.safetensors") == weight
    assert parent["files"][1]["sha256"] == "old"
    assert result["documentation_update"]["model_runtime_or_weights_changed"] is False
