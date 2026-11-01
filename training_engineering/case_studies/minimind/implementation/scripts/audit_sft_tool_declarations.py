"""Audit function declaration/call consistency; do not filter SFT records."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from minimind_lab.data.sft import prepare_sft_messages
from sft_candidate_lineage import iter_candidate_rows


ROOT = Path(__file__).resolve().parents[1]


def inspect_declarations(messages):
    try:
        prepared, tools = prepare_sft_messages(messages)
    except ValueError as error:
        return {"tool_record": False, "calls": 0,
                "issues": [{"code": "adapter_error", "detail": str(error)}]}
    names = [tool["function"]["name"] for tool in tools]
    declared = set(names)
    issues = [{"code": "duplicate_declared_name", "name": name}
              for name, count in Counter(names).items() if count > 1]
    calls = 0
    for index, message in enumerate(prepared):
        for call_index, call in enumerate(message.get("tool_calls", [])):
            calls += 1
            name = call["function"]["name"]
            if name not in declared:
                issues.append({"code": "undeclared_function_call", "message_index": index,
                               "call_index": call_index, "called_name": name,
                               "declared_names": sorted(declared)})
    return {"tool_record": bool(tools or calls or any(m["role"] == "tool" for m in prepared)),
            "calls": calls, "issues": issues}


def audit(reference, issues_path):
    counts = Counter()
    issue_counts = Counter()
    examples = []
    digest = hashlib.sha256()
    byte_count = 0
    with issues_path.open("xb") as stream:
        for line, raw, trace in iter_candidate_rows(
                Path(reference["outputs"]["unique"]["path"]),
                Path(reference["outputs"]["lineage"]["path"]),
                reference["outputs"]["unique"], reference["outputs"]["lineage"]):
            record = json.loads(raw.decode("utf-8"))
            result = inspect_declarations(record["conversations"])
            counts["records_checked"] += 1
            counts["tool_records"] += int(result["tool_record"])
            counts["function_calls"] += result["calls"]
            if result["issues"]:
                counts["flagged_records"] += 1
                issue_counts.update({issue["code"] for issue in result["issues"]})
                packet = {"candidate_line": line, "source_line": trace["source_line"],
                          "source_line_sha256": trace["source_line_sha256"], **result}
                payload = (json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                stream.write(payload)
                digest.update(payload)
                byte_count += len(payload)
                if len(examples) < 10:
                    examples.append(packet)
            if line % 100000 == 0:
                print(f"Checked declarations in {line:,} candidate records", flush=True)
    counts.setdefault("flagged_records", 0)
    return {"counts": dict(counts), "records_by_issue": dict(sorted(issue_counts.items())),
            "example_flags": examples,
            "issues_file": {"path": str(issues_path.resolve()), "records": counts["flagged_records"],
                            "bytes": byte_count, "sha256": digest.hexdigest()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-report", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.details_dir.exists() or args.report_dir.exists():
        raise FileExistsError("use fresh detail and report directories")
    start = time.perf_counter()
    reference_bytes = args.dedup_report.read_bytes()
    reference = json.loads(reference_bytes)
    code_paths = {"audit_entry": Path(__file__), "lineage_reader": ROOT / "scripts/sft_candidate_lineage.py",
                  "sft_adapter": ROOT / "src/minimind_lab/data/sft.py",
                  "validator": ROOT / "src/minimind_lab/data/validator.py"}
    code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in code_paths.items()}
    if reference["status"] != "exact_deduplication_completed_not_final_training_data":
        raise ValueError("unsupported candidate report status")
    args.details_dir.mkdir(parents=True, exist_ok=False)
    partial = args.details_dir / "issues.jsonl.partial"
    result = audit(reference, partial)
    if (args.dedup_report.read_bytes() != reference_bytes or any(
            hashlib.sha256(path.read_bytes()).hexdigest() != code_hashes[key]
            for key, path in code_paths.items())):
        raise ValueError("candidate report or audit code changed during run")
    final = args.details_dir / "issues.jsonl"
    partial.rename(final)
    result["issues_file"]["path"] = str(final.resolve())
    report = {"schema_version": 1, "status": "declaration_audit_completed_flags_require_review",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": reference["outputs"]["unique"],
              "original_source_sha256": reference["original_source"]["sha256"],
              "dedup_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "code_sha256": code_hashes, **result,
              "scope": {"source_modified": False, "filtering_applied": False,
                        "checked": ["adapter_representational_checks", "duplicate_declared_names",
                                    "structured_call_name_in_declared_names"],
                        "not_checked": ["full_argument_JSON_Schema", "call_result_matching",
                                        "tool_execution_truth", "answer_quality", "tool_calls_embedded_only_in_text"],
                        "data_recipe_frozen": False},
              "elapsed_seconds": time.perf_counter() - start}
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "records_by_issue": result["records_by_issue"]}, indent=2))


if __name__ == "__main__":
    main()
