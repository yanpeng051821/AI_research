"""Local publication guards; no network or real account credentials are used."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def publisher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("publish_pretrain_bundle")


def test_credential_reader_accepts_existing_lowercase_key_and_rejects_ambiguity(tmp_path, publisher):
    path = tmp_path / ".env"
    token = "hf_" + "a" * 30
    path.write_text("hf=" + token, encoding="utf-8")
    assert publisher.credential(path) == token
    path.write_text("hf=" + token + "\nHF_TOKEN=hf_" + "b" * 30, encoding="utf-8")
    with pytest.raises(ValueError, match="unambiguous"):
        publisher.credential(path)


@pytest.mark.parametrize("case", ["valid", "tampered", "escape"])
def test_public_payload_hash_and_path_guard(tmp_path, publisher, case):
    source = tmp_path / "config.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {"files": [publisher.describe(source, tmp_path)]}
    if case == "valid":
        publisher.check_files(tmp_path, manifest)
    elif case == "tampered":
        source.write_text('{"changed":true}', encoding="utf-8")
        with pytest.raises(ValueError, match="identity mismatch"):
            publisher.check_files(tmp_path, manifest)
    else:
        manifest["files"][0]["path"] = "../outside.json"
        with pytest.raises(ValueError, match="invalid manifest path"):
            publisher.check_files(tmp_path, manifest)
