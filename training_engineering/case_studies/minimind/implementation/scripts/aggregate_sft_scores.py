"""Validate source-bound manual checklists and aggregate them; never judge answers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def outcome(statuses):
    if not statuses or any(s not in ("pass", "fail", "uncertain") for s in statuses):
        raise ValueError("invalid or empty checklist statuses")
    return "fail" if "fail" in statuses else "uncertain" if "uncertain" in statuses else "pass"


def validate_review(review, case, generation, case_raw, generation_raw):
    if (review["id"] != case["id"] or review["id"] != generation["id"]
            or review["category"] != case["category"] or review["tier"] != case["tier"]):
        raise ValueError("review identity/category mismatch")
    if (review["case_line_sha256"] != hashlib.sha256(case_raw).hexdigest()
            or review["generation_line_sha256"] != hashlib.sha256(generation_raw).hexdigest()):
        raise ValueError("reviewed source bytes changed")
    checks = review["checks"]
    if len(checks) != len(case["checks"]):
        raise ValueError("missing or extra checklist items")
    for index, (actual, expected) in enumerate(zip(checks, case["checks"]), 1):
        if actual["index"] != index or actual["criterion"] != expected or not actual["reason"].strip():
            raise ValueError("checklist changed or rationale missing")
        for quote in actual["evidence_quotes"]:
            if not quote or quote not in generation["raw_text"]:
                raise ValueError("evidence quote not in the full raw output")
    derived = outcome([r["status"] for r in checks])
    if review["task_outcome"] != derived:
        raise ValueError("task label disagrees with frozen checklist aggregation")
    return derived


def run(recipe_dir, run_dir, reviews_path, output):
    if output.exists():
        raise FileExistsError("do not overwrite a scoring summary")
    case_path = recipe_dir / "probes.jsonl"
    config_path = recipe_dir / "evaluation_config.json"
    freeze_path = recipe_dir / "evaluation_freeze.json"
    generation_path = run_dir / "generations.jsonl"
    verification_path = run_dir / "verification.json"
    inputs = (case_path, config_path, freeze_path, generation_path, verification_path, reviews_path, Path(__file__))
    identities = {str(p.resolve()): sha(p) for p in inputs}
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen":
        raise ValueError("expected a frozen evaluation")
    for path in (case_path, config_path):
        if freeze["files"][str(path.resolve())]["sha256"] != sha(path):
            raise ValueError("frozen probes or scoring configuration changed")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification["files"]["generations.jsonl"]["sha256"] != sha(generation_path):
        raise ValueError("generation file differs from completed-run verification")
    case_lines = case_path.read_bytes().splitlines(keepends=True)
    generation_lines = generation_path.read_bytes().splitlines(keepends=True)
    reviews = [json.loads(raw) for raw in reviews_path.read_bytes().splitlines()]
    if not (len(case_lines) == len(generation_lines) == len(reviews) == 36):
        raise ValueError("incomplete fixed-probe review coverage")
    by_tier, by_category = {}, {}
    ids, gap_cases, quality = set(), [], Counter()
    for case_raw, generation_raw, review in zip(case_lines, generation_lines, reviews):
        case, generation = json.loads(case_raw), json.loads(generation_raw)
        if review["id"] in ids:
            raise ValueError("duplicate review ID")
        ids.add(review["id"])
        if (review["evaluation_freeze_sha256"] != sha(freeze_path)
                or review["cases_file_sha256"] != sha(case_path)
                or review["generations_file_sha256"] != sha(generation_path)):
            raise ValueError("review file identity mismatch")
        label = validate_review(review, case, generation, case_raw, generation_raw)
        by_tier.setdefault(case["tier"], Counter())[label] += 1
        by_category.setdefault(case["category"], Counter())[label] += 1
        quality.update(set(review["quality_observations"]))
        if any("frozen_checks" in flag for flag in review["quality_observations"]):
            gap_cases.append(case["id"])

    def stats(counts):
        total = sum(counts.values())
        return {"records": total, **{key: counts[key] for key in ("pass", "fail", "uncertain")},
                "pass_rate_lower_bound": counts["pass"] / total,
                "pass_rate_upper_bound": (counts["pass"] + counts["uncertain"]) / total}

    if any(sha(Path(p)) != expected for p, expected in identities.items()):
        raise ValueError("scoring inputs changed during aggregation")
    result = {"status": "assistant_initial_review_complete_not_independently_reviewed",
              "reviewed_records": len(reviews), "by_tier": {k: stats(v) for k, v in by_tier.items()},
              "by_category": {k: stats(v) for k, v in by_category.items()},
              "quality_observation_counts": dict(quality), "rubric_gap_cases": gap_cases,
              "task_pass_is_frozen_checklist_pass_not_overall_response_quality": True,
              "termination_scored_separately": True, "reviewer": "assistant_full_output_manual_review",
              "independently_reviewed": False, "user_reviewed": False,
              "semantic_judgments_automatically_verified": False,
              "source_coverage_hashes_checklists_and_quotes_verified": True,
              "inputs_sha256": identities, "training_started": False}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.recipe_dir.resolve(), args.run_dir.resolve(), args.reviews.resolve(), args.output.resolve())
