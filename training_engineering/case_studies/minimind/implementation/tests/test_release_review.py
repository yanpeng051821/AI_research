"""Check the release scanner's narrow guarantees without accessing the network."""

import importlib
from pathlib import Path


def release_tools(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_pretrain_release")


def test_release_scan_reports_locations_without_echoing_secret(tmp_path, monkeypatch):
    tools = release_tools(monkeypatch)
    secret = "hf_" + "a" * 30
    source = tmp_path / "private.txt"
    source.write_text(secret + "\nC:\\Users\\example\\draft.txt", encoding="utf-8")
    report = tools.scan_texts([("candidate.txt", source)])
    assert report["findings"] == [
        {"file": "candidate.txt", "kind": "credential_pattern"},
        {"file": "candidate.txt", "kind": "local_absolute_path"},
    ]
    assert secret not in str(report)


def test_release_scan_accepts_relative_native_usage(tmp_path, monkeypatch):
    tools = release_tools(monkeypatch)
    source = tmp_path / "README.md"
    source.write_text('python -I -B load_model.py --prompt "春天"\nmodel.safetensors', encoding="utf-8")
    report = tools.scan_texts([("README.md", source)])
    assert report["text_files_checked"] == 1
    assert report["findings"] == []
