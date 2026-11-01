"""Apply a confirmed undeclared-function rule and recheck the remaining candidate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from apply_sft_review_decisions import apply_decisions
from audit_sft_tool_declarations import audit
from inspect_sft_encoding import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def build_exclusion_plan(candidate, audit_report, issues_bytes, rule):
    if (rule.get("status") != "user_confirmed"
            or rule.get("action") != "exclude_whole_record_to_quarantine"
            or rule.get("issue_code") != "undeclared_function_call"):
        raise ValueError("unsupported or unconfirmed quarantine rule")
    if (candidate["status"] != "confirmed_exclusions_applied_not_final_training_data"
            or candidate["dedup_report_sha256"] != audit_report["dedup_report_sha256"]
            or candidate["original_source"]["sha256"] != audit_report["original_source_sha256"]
            or rule["source_sha256"] != audit_report["original_source_sha256"]):
        raise ValueError("audit and candidate do not share the required source lineage")
    if hashlib.sha256(issues_bytes).hexdigest() != audit_report["issues_file"]["sha256"]:
        raise ValueError("audit issue file identity mismatch")
    previous = {entry["source_line"]: entry for entry in candidate["applied_decisions"]}
    decisions = {}
    already_excluded = []
    seen = set()
    total_flags = 0
    for raw in issues_bytes.decode("utf-8").splitlines():
        packet = json.loads(raw)
        total_flags += 1
        source_line = packet["source_line"]
        if source_line in seen:
            raise ValueError("duplicate source line in audit issues")
        seen.add(source_line)
        relevant = [issue for issue in packet["issues"] if issue["code"] == rule["issue_code"]]
        if not relevant:
            continue
        if source_line in previous:
            if previous[source_line]["source_line_sha256"] != packet["source_line_sha256"]:
                raise ValueError("previously excluded record hash mismatch")
            already_excluded.append(source_line)
            continue
        names = sorted({issue["called_name"] for issue in relevant})
        decisions[source_line] = {
            "status": "user_confirmed", "action": "exclude",
            "source_sha256": rule["source_sha256"], "source_line": source_line,
            "source_line_sha256": packet["source_line_sha256"],
            "reason": f"{rule['policy_id']}: undeclared structured function calls {names}",
        }
    if total_flags != audit_report["issues_file"]["records"]:
        raise ValueError("audit issue count mismatch")
    if not decisions:
        raise ValueError("no new applicable quarantine decisions")
    return decisions, sorted(already_excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("use fresh data and report directories")
    start = time.perf_counter()
    input_paths = {"candidate_report": args.candidate_report, "audit_report": args.audit_report, "rule": args.rule}
    input_bytes = {key: path.read_bytes() for key, path in input_paths.items()}
    candidate, audit_report, rule = (json.loads(input_bytes[key]) for key in input_paths)
    if (hashlib.sha256(input_bytes["candidate_report"]).hexdigest() != rule["candidate_report_sha256"]
            or hashlib.sha256(input_bytes["audit_report"]).hexdigest() != rule["audit_report_sha256"]):
        raise ValueError("reports differ from the confirmed rule's fixed inputs")
    issues_path = Path(audit_report["issues_file"]["path"])
    decisions, already_excluded = build_exclusion_plan(candidate, audit_report, issues_path.read_bytes(), rule)
    code_paths = {"entry": Path(__file__),
                  "decision_application": ROOT / "scripts/apply_sft_review_decisions.py",
                  "tool_audit": ROOT / "scripts/audit_sft_tool_declarations.py",
                  "lineage_reader": ROOT / "scripts/sft_candidate_lineage.py",
                  "sft_adapter": ROOT / "src/minimind_lab/data/sft.py",
                  "validator": ROOT / "src/minimind_lab/data/validator.py"}
    code_hashes = {key: sha256_file(path) for key, path in code_paths.items()}
    # Reuse the exact-record executor with explicit input metadata, not a fake report file.
    execution_input = {"outputs": {"unique": candidate["outputs"]["candidate"],
                                    "lineage": candidate["outputs"]["lineage"]}}
    result = apply_decisions(execution_input, decisions, args.output_dir)
    post_input = {"outputs": {"unique": result["outputs"]["candidate"],
                               "lineage": result["outputs"]["lineage"]}}
    post_path = args.output_dir / "remaining_tool_issues.jsonl.partial"
    post_audit = audit(post_input, post_path)
    if post_audit["counts"]["flagged_records"]:
        raise ValueError("remaining candidate still has tool declaration audit flags")
    if post_audit["counts"]["records_checked"] != result["outputs"]["candidate"]["records"]:
        raise ValueError("post-audit record count mismatch")
    if (any(path.read_bytes() != input_bytes[key] for key, path in input_paths.items())
            or sha256_file(issues_path) != audit_report["issues_file"]["sha256"]
            or any(sha256_file(path) != code_hashes[key] for key, path in code_paths.items())):
        raise ValueError("fixed input or processing code changed during run")
    for output in result["outputs"].values():
        if sha256_file(Path(output["path"])) != output["sha256"]:
            raise ValueError("written output identity mismatch")
    post_final = args.output_dir / "remaining_tool_issues.jsonl"
    post_path.rename(post_final)
    post_audit["issues_file"]["path"] = str(post_final.resolve())
    decision_path = args.output_dir / "rule_exclusions.jsonl"
    decision_path.write_text("".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                                     for entry in result["applied_decisions"]), encoding="utf-8")
    report = {
        "schema_version": 1, "status": "confirmed_tool_quarantine_completed_not_final_training_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": candidate["outputs"]["candidate"], "original_source": candidate["original_source"],
        "rule": rule, "input_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in input_bytes.items()},
        "audit_issues_sha256": audit_report["issues_file"]["sha256"], "code_sha256": code_hashes,
        "previously_excluded_flagged_source_lines": already_excluded,
        "newly_quarantined_records": len(decisions),
        "outputs": result["outputs"], "post_audit": post_audit,
        "exclusion_manifest": {"path": str(decision_path.resolve()), "records": len(decisions),
                               "sha256": sha256_file(decision_path)},
        "scope": {"input_modified": False, "raw_record_bytes_preserved": True,
                  "relative_order_preserved": True, "other_tool_records_blanket_excluded": False,
                  "tool_declarations_invented": False, "full_protocol_accepted": False,
                  "content_quality_accepted": False, "identity_rewrite_applied": False,
                  "split_created": False, "data_recipe_frozen": False},
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"newly_quarantined_records": len(decisions),
                      "retained_records": result["outputs"]["candidate"]["records"],
                      "post_audit": post_audit["counts"]}, indent=2))


if __name__ == "__main__":
    main()
