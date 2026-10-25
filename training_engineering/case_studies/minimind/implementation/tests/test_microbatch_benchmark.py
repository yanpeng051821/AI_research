import json
from pathlib import Path

import pytest

from minimind_lab.training.microbatch_benchmark import resolve_benchmark_config, select_candidate


def test_benchmark_candidates_must_preserve_effective_batch(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    import hashlib
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    config = tmp_path / "benchmark.json"
    config.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "microbatch_benchmark",
        "source_config": str(source),
        "source_config_sha256": source_sha256,
        "effective_batch_records": 10,
        "candidates": [2, 4],
        "excluded_candidates": [],
        "warmup_updates": 1,
        "measured_updates": 2,
        "minimum_device_free_ratio": 0.1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="divide effective_batch_records"):
        resolve_benchmark_config(config, Path("."))


def test_candidate_selection_requires_headroom_then_uses_throughput():
    results = [
        {"micro_batch_size": 2, "status": "completed", "minimum_device_free_ratio": 0.30,
         "measured_records_per_second": 100.0},
        {"micro_batch_size": 4, "status": "completed", "minimum_device_free_ratio": 0.20,
         "measured_records_per_second": 120.0},
        {"micro_batch_size": 8, "status": "completed", "minimum_device_free_ratio": 0.05,
         "measured_records_per_second": 140.0},
        {"micro_batch_size": 16, "status": "oom"},
    ]
    assert select_candidate(results, 0.1)["micro_batch_size"] == 4
