"""Apply an approved, source-bound extra check without altering V1 evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

from aggregate_sft_scores import outcome, sha, validate_review


def validate_addition(patch, addition, case, generation, original):
    if (patch["version"] != "scoring_v2"
            or patch["status"] != "user_approved_before_sft"
            or patch["change_scope"] != "append_one_check_only"
            or any(patch[k] for k in ("prompts_changed", "generation_changed", "other_checks_changed"))):
        raise ValueError("unsupported patch scope or approval")
    if not (patch["case_id"] == addition["id"] == case["id"] == generation["id"]):
        raise ValueError("patch case mismatch")
    if (addition["version"] != patch["version"]
            or addition["criterion"] != patch["append_check"]
            or addition["index"] != len(case["checks"]) + 1
            or not addition["reason"].strip()):
        raise ValueError("added criterion mismatch")
    for key in ("case_line_sha256", "generation_line_sha256"):
        if addition[key] != original[key]:
            raise ValueError("added review source mismatch")
    if not addition["evidence_quotes"] or any(
        not q or q not in generation["raw_text"] for q in addition["evidence_quotes"]
    ):
        raise ValueError("added evidence not in original output")
    return outcome([c["status"] for c in original["checks"]] + [addition["status"]])


def run(experiment, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite scoring evidence")
    recipe = experiment / "evaluation_recipe"
    baseline = experiment / "results/b0_v1"
    prior = baseline / "scoring_v1"
    patch_path = recipe / "scoring_v2_patch.json"
    addition_path = baseline / "scoring_v2/added_check.assistant.json"
    summary_path = prior / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs = {Path(p): h for p, h in summary["inputs_sha256"].items()}
    for p in (summary_path, patch_path, addition_path, Path(__file__)):
        inputs[p.resolve()] = sha(p)
    if any(sha(p) != h for p, h in inputs.items()):
        raise ValueError("V1 inputs or aggregator changed")
    for p in (recipe / "probes.jsonl", recipe / "evaluation_freeze.json",
              baseline / "generations.jsonl", prior / "reviews.assistant.jsonl"):
        if p.resolve() not in inputs:
            raise ValueError("requested experiment is not the source-bound V1 run")
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    addition = json.loads(addition_path.read_text(encoding="utf-8"))
    if patch["base_evaluation_freeze_sha256"] != sha(recipe / "evaluation_freeze.json"):
        raise ValueError("patch targets another evaluation freeze")
    cases = (recipe / "probes.jsonl").read_bytes().splitlines(keepends=True)
    generations = (baseline / "generations.jsonl").read_bytes().splitlines(keepends=True)
    reviews = [json.loads(r) for r in (prior / "reviews.assistant.jsonl").read_bytes().splitlines()]
    if not len(cases) == len(generations) == len(reviews) == 36:
        raise ValueError("incomplete review coverage")
    tiers, categories, decisions, changed = {}, {}, [], []
    patched = 0
    for a, b, review in zip(cases, generations, reviews):
        case, generation = json.loads(a), json.loads(b)
        label = validate_review(review, case, generation, a, b)
        previous = label
        if case["id"] == patch["case_id"]:
            label = validate_addition(patch, addition, case, generation, review)
            patched += 1
        if previous != label:
            changed.append({"id": case["id"], "v1": previous, "v2": label})
        decisions.append({"id": case["id"], "category": case["category"],
                          "tier": case["tier"], "task_outcome": label,
                          "checks_source": "v1_plus_patch" if case["id"] == patch["case_id"] else "v1_unchanged"})
        tiers.setdefault(case["tier"], Counter())[label] += 1
        categories.setdefault(case["category"], Counter())[label] += 1
    if patched != 1 or len({d["id"] for d in decisions}) != 36:
        raise ValueError("patch coverage or ID uniqueness failure")

    def stats(counts):
        total = sum(counts.values())
        return {"records": total, **{k: counts[k] for k in ("pass", "fail", "uncertain")},
                "pass_rate": counts["pass"] / total}

    result = {"version": "scoring_v2", "status": "assistant_initial_review_not_independently_reviewed",
              "by_tier": {k: stats(v) for k, v in tiers.items()},
              "by_category": {k: stats(v) for k, v in categories.items()},
              "changed_decisions": changed, "decisions": decisions,
              "inputs_sha256": {str(p.resolve()): h for p, h in inputs.items()},
              "independently_reviewed": False, "user_approved_patch": True,
              "user_reviewed_all_cases": False, "semantic_judgments_automatically_verified": False,
              "generation_rerun": False, "training_started": False}
    if any(sha(p) != h for p, h in inputs.items()):
        raise ValueError("inputs changed during scoring")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"by_tier": result["by_tier"], "changed_decisions": changed}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.experiment_dir.resolve(), args.output.resolve())
