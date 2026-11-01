from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aggregate_sft_scores", ROOT / "scripts/aggregate_sft_scores.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def test_checklist_outcomes_do_not_reward_negative_guard_only_or_discard_uncertainty():
    assert scoring.outcome(["pass", "pass"]) == "pass"
    assert scoring.outcome(["fail", "pass"]) == "fail"
    assert scoring.outcome(["pass", "uncertain"]) == "uncertain"
    assert scoring.outcome(["uncertain", "fail"]) == "fail"
    with pytest.raises(ValueError):
        scoring.outcome([])


def test_review_binding_and_evidence_validation_preserve_the_frozen_criterion():
    case = {"id": "example", "category": "grounded", "tier": "core", "checks": ["contains the requested place"]}
    generation = {"id": "example", "raw_text": "place repeated place"}
    a, b = json.dumps(case).encode(), json.dumps(generation).encode()
    review = {"id": "example", "category": "grounded", "tier": "core",
              "case_line_sha256": hashlib.sha256(a).hexdigest(),
              "generation_line_sha256": hashlib.sha256(b).hexdigest(),
              "checks": [{"index": 1, "criterion": case["checks"][0], "status": "pass",
                          "reason": "requested place present; repetition is separate", "evidence_quotes": ["place"]}],
              "task_outcome": "pass"}
    assert scoring.validate_review(review, case, generation, a, b) == "pass"
    changed = deepcopy(review)
    changed["checks"][0]["criterion"] = "new post-hoc repetition penalty"
    with pytest.raises(ValueError, match="checklist changed"):
        scoring.validate_review(changed, case, generation, a, b)
    changed = deepcopy(review)
    changed["checks"][0]["evidence_quotes"] = ["invented evidence"]
    with pytest.raises(ValueError, match="evidence quote"):
        scoring.validate_review(changed, case, generation, a, b)
    with pytest.raises(ValueError, match="source bytes changed"):
        scoring.validate_review(review, case, generation, a, b + b" ")
