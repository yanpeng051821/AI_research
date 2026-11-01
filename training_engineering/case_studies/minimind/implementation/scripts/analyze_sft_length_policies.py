"""Compare whole-record exclusion and head slicing without changing SFT data."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from inspect_sft_encoding import sha256_file
from inspect_sft_full_encoding import pipeline_identity
from minimind_lab.data.sft import encode_sft_conversation


ROOT = Path(__file__).resolve().parents[1]


def truncation_effect(input_ids, labels, limit, eos_token_id):
    if limit < 2 or len(input_ids) != len(labels):
        raise ValueError("limit must be at least 2 and inputs/labels must align")
    before = sum(value != -100 for value in labels[1:])
    retained = sum(value != -100 for value in labels[1:limit])
    if len(input_ids) <= limit:
        ending = "not_truncated"
    elif labels[limit - 1] != -100 and labels[limit] != -100:
        ending = "assistant_eos_without_newline" if input_ids[limit - 1] == eos_token_id else "assistant_without_eos"
    else:
        ending = "outside_assistant_supervision"
    return {"targets_before": before, "targets_retained": retained,
            "targets_lost": before - retained, "ending_state": ending,
            "zero_targets_after_cut": retained == 0}


def empty_policy():
    return {"dropped_records": 0, "dropped_input_positions": 0, "dropped_targets_after_shift": 0}


def finish_exclusion(counts, total_records, total_targets):
    return {**counts, "retained_records": total_records - counts["dropped_records"],
            "retained_targets_after_shift": total_targets - counts["dropped_targets_after_shift"],
            "dropped_record_fraction": counts["dropped_records"] / total_records if total_records else None,
            "dropped_target_fraction": counts["dropped_targets_after_shift"] / total_targets if total_targets else None}


def analyze(args):
    limits = sorted(set(args.limits))
    if not limits or min(limits) < 2:
        raise ValueError("candidate limits must be at least 2")
    if args.report_dir.exists():
        raise FileExistsError("use a fresh analysis report directory")
    start = time.perf_counter()
    reference_hash = sha256_file(args.encoding_report)
    reference = json.loads(args.encoding_report.read_text(encoding="utf-8"))
    baseline = reference["overall"]
    if baseline["flagged_records"] or baseline["encoded_records"] != baseline["records"]:
        raise ValueError("analysis requires a fully encoded reference without diagnostic flags")
    details = Path(reference["files"]["records"]["path"])
    if sha256_file(details) != reference["files"]["records"]["sha256"]:
        raise ValueError("encoding detail identity mismatch")
    if (args.source.stat().st_size != reference["source"]["size"]
            or sha256_file(args.source) != reference["source"]["sha256"]):
        raise ValueError("source identity mismatch")
    before_pipeline = pipeline_identity(args.tokenizer_dir)
    if before_pipeline != reference["pipeline"]:
        raise ValueError("encoding pipeline differs from reference")

    exclusion = {limit: empty_policy() for limit in limits}
    by_stratum = {name: {limit: empty_policy() for limit in limits} for name in reference["by_stratum"]}
    selected = {}
    total_records = total_inputs = total_targets = 0
    with details.open("r", encoding="utf-8") as stream:
        for expected_line, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["source_line"] != expected_line or row["issues"]:
                raise ValueError("encoding details are unordered or contain flags")
            total_records += 1
            total_inputs += row["input_tokens"]
            total_targets += row["targets_after_shift"]
            for limit in limits:
                if row["input_tokens"] > limit:
                    for counts in (exclusion[limit], by_stratum[row["stratum"]][limit]):
                        counts["dropped_records"] += 1
                        counts["dropped_input_positions"] += row["input_tokens"]
                        counts["dropped_targets_after_shift"] += row["targets_after_shift"]
            if row["input_tokens"] > min(limits):
                selected[row["source_line"]] = row
    if (total_records != baseline["records"] or total_inputs != baseline["total_input_positions"]
            or total_targets != baseline["total_targets_after_shift"]):
        raise ValueError("encoding detail totals differ from reference")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define EOS")
    slicing = {limit: Counter() for limit in limits}
    examples = {limit: [] for limit in limits}
    source_digest = hashlib.sha256()
    inspected = source_lines = 0
    with args.source.open("rb") as stream:
        for source_lines, raw in enumerate(stream, 1):
            source_digest.update(raw)
            row = selected.get(source_lines)
            if row is None:
                continue
            if hashlib.sha256(raw).hexdigest() != row["source_line_sha256"]:
                raise ValueError(f"source-line identity mismatch at {source_lines}")
            record = json.loads(raw)
            feature = encode_sft_conversation(record["conversations"], tokenizer)
            ids, labels = feature["input_ids"], feature["labels"]
            if (len(ids) != row["input_tokens"]
                    or sum(value != -100 for value in labels[1:]) != row["targets_after_shift"]):
                raise ValueError(f"re-encoding differs from reference at {source_lines}")
            for limit in limits:
                if len(ids) <= limit:
                    continue
                effect = truncation_effect(ids, labels, limit, tokenizer.eos_token_id)
                slicing[limit].update({"truncated_records": 1,
                                      "lost_targets_after_shift": effect["targets_lost"],
                                      "records_losing_targets": int(effect["targets_lost"] > 0),
                                      "records_with_zero_retained_targets": int(effect["zero_targets_after_cut"]),
                                      effect["ending_state"]: 1})
                if len(examples[limit]) < 5:
                    examples[limit].append({"source_line": source_lines,
                                            "source_line_sha256": row["source_line_sha256"],
                                            "stratum": row["stratum"], "input_tokens": len(ids),
                                            **effect})
            inspected += 1
            if inspected % 5000 == 0:
                print(f"Re-encoded {inspected:,}/{len(selected):,} long records", flush=True)
    if (inspected != len(selected) or source_lines != total_records
            or source_digest.hexdigest() != reference["source"]["sha256"]):
        raise ValueError("source content/count changed during analysis")
    if (pipeline_identity(args.tokenizer_dir) != before_pipeline
            or sha256_file(args.encoding_report) != reference_hash
            or sha256_file(details) != reference["files"]["records"]["sha256"]):
        raise ValueError("pipeline or input report changed during analysis")

    comparisons = {}
    for limit in limits:
        counts = slicing[limit]
        if counts["truncated_records"] != exclusion[limit]["dropped_records"]:
            raise RuntimeError("cut and exclusion record counts disagree")
        comparisons[str(limit)] = {
            "whole_record_exclusion": finish_exclusion(exclusion[limit], total_records, total_targets),
            "head_slicing": {
                "truncated_records": counts["truncated_records"],
                "lost_targets_after_shift": counts["lost_targets_after_shift"],
                "retained_targets_after_shift": total_targets - counts["lost_targets_after_shift"],
                "lost_target_fraction": counts["lost_targets_after_shift"] / total_targets,
                "records_losing_targets": counts["records_losing_targets"],
                "records_with_zero_retained_targets": counts["records_with_zero_retained_targets"],
                "cut_ending_states": {name: counts[name] for name in (
                    "assistant_without_eos", "assistant_eos_without_newline", "outside_assistant_supervision")},
            },
            "by_stratum_exclusion": {name: finish_exclusion(by_stratum[name][limit],
                                        reference["by_stratum"][name]["records"],
                                        reference["by_stratum"][name]["total_targets_after_shift"])
                                     for name in by_stratum},
            "cut_examples": examples[limit],
        }
    report = {
        "schema_version": 1, "status": "length_policy_impact_analyzed_no_policy_selected",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"encoding_report_sha256": reference_hash,
                  "details_sha256": reference["files"]["records"]["sha256"],
                  "source_sha256": source_digest.hexdigest(), "records": total_records,
                  "input_positions": total_inputs, "targets_after_shift": total_targets,
                  "long_records_reencoded": inspected},
        "candidate_limits": limits, "comparisons": comparisons,
        "method": {"whole_record_exclusion": "simulate omission when full input length > limit",
                   "head_slicing": "simulate input_ids[:limit], labels[:limit]; no replacement/appended EOS",
                   "target_count": "count non--100 labels[1:]",
                   "ending_state": "use current contiguous supervised span across cut; distinguish retained EOS from lost newline",
                   "no_claim_of_complete_tool_trajectory_after_cut": True},
        "pipeline": before_pipeline,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - start,
        "scope": {"current_unfiltered_candidate_only": True, "quality_acceptance_claimed": False,
                  "truncation_applied": False, "filtering_applied": False, "split_created": False,
                  "final_length_selected": False, "training_started": False},
    }
    args.report_dir.mkdir(parents=True, exist_ok=False)
    with (args.report_dir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report_dir": str(args.report_dir), "long_records_reencoded": inspected,
                      "elapsed_seconds": report["elapsed_seconds"], "comparisons": {
                          key: {"whole_record_exclusion": value["whole_record_exclusion"],
                                "head_slicing": value["head_slicing"]}
                          for key, value in comparisons.items()}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--encoding-report", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--limits", nargs="+", type=int, default=[768, 1024, 1536, 2048])
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
