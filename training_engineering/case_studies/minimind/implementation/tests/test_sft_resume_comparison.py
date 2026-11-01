from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "sft_resume_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_sft_resume.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_exact_comparison_includes_rng_optimizer_and_metadata():
    original = {"model": {"weight": torch.tensor([1.0])},
                "optimizer": {0: {"exp_avg": torch.tensor([0.5])}},
                "cuda_rng_state_all": [torch.tensor([1, 2], dtype=torch.uint8)],
                "metadata": {"contract": "same", "tokens": 12}}
    comparison.equal_tree(original, deepcopy(original))
    for key in ("optimizer", "cuda_rng_state_all", "metadata"):
        changed = deepcopy(original)
        if key == "optimizer":
            changed[key][0]["exp_avg"][0] += 0.01
        elif key == "cuda_rng_state_all":
            changed[key][0][0] = 3
        else:
            changed[key]["tokens"] += 1
        with pytest.raises(ValueError, match=key):
            comparison.equal_tree(original, changed)


def test_exact_comparison_rejects_dtype_and_missing_fields():
    with pytest.raises(ValueError, match="tensor"):
        comparison.equal_tree(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="mapping"):
        comparison.equal_tree({"model": {}, "sampler": {}}, {"model": {}})
