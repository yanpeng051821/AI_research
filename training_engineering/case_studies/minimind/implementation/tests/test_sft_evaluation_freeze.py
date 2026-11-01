import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("freeze_sft_evaluation", ROOT / "scripts/freeze_sft_evaluation.py")
freezer = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(freezer)


def test_freeze_requires_authorization_and_refuses_existing_outputs(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        freezer.freeze(tmp_path, tmp_path / "missing_bundle", False)
    assert not list(tmp_path.iterdir())
    (tmp_path / "evaluation_config.json").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freezer.freeze(tmp_path, tmp_path / "missing_bundle", True)
    assert (tmp_path / "evaluation_config.json").read_text(encoding="utf-8") == "existing"


def test_frozen_check_rejects_changed_file_and_package_version(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    marker = tmp_path / "freeze.json"
    value = {"status": "frozen", "files": {"source": freezer.descriptor(source)},
             "runtime_versions": {"torch": importlib.metadata.version("torch")}}
    marker.write_text(json.dumps(value), encoding="utf-8")
    assert freezer.check(marker)["status"] == "frozen_evaluation_verified_not_run"
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="file changed"):
        freezer.check(marker)
    source.write_text("{}", encoding="utf-8")
    value["runtime_versions"]["torch"] = "incorrect-version"
    marker.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="package version changed"):
        freezer.check(marker)
