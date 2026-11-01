"""Apply explicitly confirmed exact-record exclusions to a verified SFT candidate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sft_candidate_lineage import iter_candidate_rows


def load_decisions(payload, source_sha256):
    decisions = {}
    for raw in payload.decode("utf-8").splitlines():
        decision = json.loads(raw)
        line = decision.get("source_line")
        digest = decision.get("source_line_sha256", "")
        if (decision.get("status") != "user_confirmed" or decision.get("action") != "exclude"
                or decision.get("source_sha256") != source_sha256
                or type(line) is not int or line < 1
                or not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not decision.get("reason")):
            raise ValueError("only confirmed, source-bound exact exclusions are supported")
        if line in decisions:
            raise ValueError("duplicate review decision for source line")
        decisions[line] = decision
    if not decisions:
        raise ValueError("no confirmed decisions supplied")
    return decisions


def apply_decisions(reference, decisions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {"candidate": output_dir / "reviewed_candidate.jsonl",
             "excluded": output_dir / "review_excluded.jsonl",
             "lineage": output_dir / "lineage.jsonl"}
    temporary = {key: path.with_suffix(".jsonl.partial") for key, path in paths.items()}
    digests = {key: hashlib.sha256() for key in paths}
    counts = {key: {"records": 0, "bytes": 0} for key in paths}
    matched = []
    with (temporary["candidate"].open("xb") as candidate,
          temporary["excluded"].open("xb") as excluded,
          temporary["lineage"].open("xb") as ledger):
        streams = {"candidate": candidate, "excluded": excluded}
        for line, raw, trace in iter_candidate_rows(
                Path(reference["outputs"]["unique"]["path"]),
                Path(reference["outputs"]["lineage"]["path"]),
                reference["outputs"]["unique"], reference["outputs"]["lineage"]):
            decision = decisions.get(trace["source_line"])
            if decision and decision["source_line_sha256"] != trace["source_line_sha256"]:
                raise ValueError("reviewed record content hash differs from candidate")
            destination = "excluded" if decision else "candidate"
            streams[destination].write(raw)
            digests[destination].update(raw)
            counts[destination]["records"] += 1
            counts[destination]["bytes"] += len(raw)
            if decision:
                matched.append({"source_line": trace["source_line"], "candidate_line": line,
                                "source_line_sha256": trace["source_line_sha256"],
                                "reason": decision["reason"]})
            new_trace = {"input_line": line, "source_line": trace["source_line"],
                         "source_line_sha256": trace["source_line_sha256"],
                         "action": "exclude_review" if decision else "keep",
                         "retained_output_line": None if decision else counts["candidate"]["records"],
                         "retained_source_line": None if decision else trace["source_line"]}
            payload = (json.dumps(new_trace, separators=(",", ":")) + "\n").encode()
            ledger.write(payload)
            digests["lineage"].update(payload)
            counts["lineage"]["records"] += 1
            counts["lineage"]["bytes"] += len(payload)
    if {entry["source_line"] for entry in matched} != set(decisions):
        raise ValueError("some confirmed decisions did not match the actual candidate")
    for key, path in paths.items():
        temporary[key].rename(path)
    outputs = {key: {**counts[key], "path": str(path.resolve()), "sha256": digests[key].hexdigest()}
               for key, path in paths.items()}
    return {"outputs": outputs, "applied_decisions": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup-report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("use fresh data and report directories")
    reference_bytes, decision_bytes = args.dedup_report.read_bytes(), args.decisions.read_bytes()
    reference = json.loads(reference_bytes)
    if reference["status"] != "exact_deduplication_completed_not_final_training_data":
        raise ValueError("unsupported candidate report status")
    decisions = load_decisions(decision_bytes, reference["original_source"]["sha256"])
    code_paths = {"entry": Path(__file__), "lineage_reader": Path(__file__).with_name("sft_candidate_lineage.py")}
    code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in code_paths.items()}
    result = apply_decisions(reference, decisions, args.output_dir)
    if (args.dedup_report.read_bytes() != reference_bytes or args.decisions.read_bytes() != decision_bytes
            or any(hashlib.sha256(path.read_bytes()).hexdigest() != code_hashes[key]
                   for key, path in code_paths.items())):
        raise ValueError("inputs or processing code changed during run")
    for output in result["outputs"].values():
        digest = hashlib.sha256()
        with Path(output["path"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != output["sha256"]:
            raise ValueError("written output hash mismatch")
    report = {"schema_version": 1, "status": "confirmed_exclusions_applied_not_final_training_data",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": reference["outputs"]["unique"],
              "original_source": reference["original_source"],
              "dedup_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "decisions_sha256": hashlib.sha256(decision_bytes).hexdigest(),
              "code_sha256": code_hashes, **result,
              "scope": {"input_modified": False, "confirmed_exclusions_applied": len(decisions),
                        "other_flags_automatically_excluded": False, "identity_rewrite_applied": False,
                        "remaining_records_quality_accepted": False, "split_created": False,
                        "data_recipe_frozen": False}}
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["records"] for key, value in result["outputs"].items()}, indent=2))


if __name__ == "__main__":
    main()
