"""Inspect a fixed SFT source without filtering, rewriting, or tokenizing it."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from minimind_lab.data.validator import validate_sft_record


STRATA = ("plain", "reasoning_only", "tool_only", "reasoning_and_tool")


def field_type(value):
    if value is None:
        return "null"
    return type(value).__name__


def offer_sample(sample, seen, capacity, rng, item):
    if len(sample) < capacity:
        sample.append(item)
    elif capacity:
        position = rng.randrange(seen)
        if position < capacity:
            sample[position] = item


def scan_source(source, expected_sha256, expected_size, sample_size=1000,
                stratum_sample_size=16, seed=20261029):
    if sample_size < 1 or stratum_sample_size < 1:
        raise ValueError("sample sizes must be positive")
    counts = Counter()
    roles = Counter()
    record_keys = Counter()
    field_types = Counter()
    field_shapes = Counter()
    issue_counts = Counter()
    flags = Counter()
    strata_counts = Counter({name: 0 for name in STRATA})
    message_counts = Counter()
    issue_examples = []
    random_sample = []
    strata_samples = {name: [] for name in STRATA}
    rng = random.Random(seed)
    strata_rng = {name: random.Random(f"{seed}:{name}") for name in STRATA}
    digest = hashlib.sha256()

    with source.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            counts["bytes"] += len(raw)
            counts["total_lines"] += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["malformed_lines"] += 1
                issue_counts["invalid_utf8_or_json"] += 1
                if len(issue_examples) < 20:
                    issue_examples.append({"source_line": line_number,
                                           "issues": ["invalid_utf8_or_json"]})
                continue
            counts["parsed_records"] += 1
            issues = validate_sft_record(record)
            if issues:
                counts["records_with_validation_errors"] += 1
                issue_counts.update(issues)
                if len(issue_examples) < 20:
                    issue_examples.append({"source_line": line_number, "issues": issues})
                continue

            counts["valid_records"] += 1
            record_keys.update(record.keys())
            messages = record["conversations"]
            message_counts[len(messages)] += 1
            record_flags = set()
            for message in messages:
                role = message["role"]
                roles[role] += 1
                for key, value in message.items():
                    field_types[f"{key}:{field_type(value)}"] += 1
                content = message["content"]
                counts["content_characters"] += len(content)
                if role == "assistant" and not content.strip():
                    counts["assistant_messages_with_empty_content"] += 1
                    record_flags.add("empty_assistant_content")
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    record_flags.add("reasoning_content")
                    counts["reasoning_characters"] += len(reasoning)
                if "<think>" in content or "</think>" in content:
                    record_flags.add("inline_thinking")
                if role == "tool":
                    record_flags.add("tool_role")
                for key in ("tools", "tool_calls"):
                    value = message.get(key)
                    if not value or (isinstance(value, str) and not value.strip()):
                        continue
                    record_flags.add(key)
                    parsed = value
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                        except json.JSONDecodeError:
                            field_shapes[f"{key}:invalid_json_string"] += 1
                            record_flags.add("invalid_tool_json")
                            continue
                    field_shapes[f"{key}:parsed_{field_type(parsed)}"] += 1
                for key in ("content", "reasoning_content"):
                    value = message.get(key)
                    if isinstance(value, str) and "jingyaogong" in value:
                        record_flags.add(f"author_name_in_{key}")
                    if isinstance(value, str) and any(
                        marker in value for marker in ("<|im_start|>", "<|im_end|>")
                    ):
                        record_flags.add("message_boundary_literal_in_text")

            reasoning = bool(record_flags & {"reasoning_content", "inline_thinking"})
            tool = bool(record_flags & {"tools", "tool_calls", "tool_role"})
            stratum = ("reasoning_and_tool" if reasoning and tool else
                       "reasoning_only" if reasoning else "tool_only" if tool else "plain")
            strata_counts[stratum] += 1
            flags.update(record_flags)
            packet = {"source_line": line_number,
                      "source_line_sha256": hashlib.sha256(raw).hexdigest(),
                      "stratum": stratum, "record": record}
            offer_sample(random_sample, counts["valid_records"], sample_size, rng, packet)
            offer_sample(strata_samples[stratum], strata_counts[stratum], stratum_sample_size,
                         strata_rng[stratum], packet)
            if line_number % 100000 == 0:
                print(f"Scanned {line_number:,} lines", flush=True)

    actual_sha256 = digest.hexdigest()
    if counts["bytes"] != expected_size or actual_sha256 != expected_sha256:
        raise ValueError("source identity differs from the expected size or SHA256")
    summary = {
        "status": "source_structure_scanned_not_accepted_for_training",
        "source": {"path": str(source), "size": counts["bytes"], "sha256": actual_sha256},
        "counts": dict(sorted(counts.items())),
        "roles_in_valid_records": dict(sorted(roles.items())),
        "record_key_presence": dict(sorted(record_keys.items())),
        "message_field_types": dict(sorted(field_types.items())),
        "tool_field_shapes": dict(sorted(field_shapes.items())),
        "record_flags": dict(sorted(flags.items())),
        "exclusive_strata_in_valid_records": dict(strata_counts),
        "message_count_histogram": dict(sorted(message_counts.items())),
        "issues": dict(sorted(issue_counts.items())), "issue_examples": issue_examples,
        "sampling": {"seed": seed, "method": "reservoir over structurally valid records",
                     "random_sample_requested": sample_size,
                     "random_sample_actual": len(random_sample),
                     "per_stratum_requested": stratum_sample_size,
                     "per_stratum_actual": {k: len(v) for k, v in strata_samples.items()},
                     "sample_line_hash_includes_original_newline": True,
                     "stratum_samples_are_not_population_weighted": True},
        "scope": {"token_lengths_computed": False, "quality_audited": False,
                  "identity_rewritten": False, "training_recipe_frozen": False,
                  "flags_are_content_markers_not_verified_task_quality": True},
    }
    return summary, random_sample, strata_samples


def write_jsonl(path, records):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in sorted(records, key=lambda row: row["source_line"]):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--stratum-sample-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.report_dir.resolve() == args.samples_dir.resolve():
        raise ValueError("keep report and raw samples in separate directories")
    if args.report_dir.exists() or args.samples_dir.exists():
        raise FileExistsError("use fresh report and sample directories")
    summary, random_sample, strata_samples = scan_source(
        args.source, args.expected_sha256, args.expected_size,
        args.sample_size, args.stratum_sample_size, args.seed,
    )
    args.samples_dir.mkdir(parents=True, exist_ok=False)
    sample_files = {"random": "random_sample.jsonl"}
    write_jsonl(args.samples_dir / sample_files["random"], random_sample)
    for stratum, records in strata_samples.items():
        name = f"{stratum}_examples.jsonl"
        sample_files[stratum] = name
        write_jsonl(args.samples_dir / name, records)
    summary["sample_files"] = {
        key: {"path": str(args.samples_dir / name),
              "sha256": hashlib.sha256((args.samples_dir / name).read_bytes()).hexdigest()}
        for key, name in sample_files.items()
    }
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    report = args.report_dir / "structure_report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "counts": summary["counts"],
                      "strata": summary["exclusive_strata_in_valid_records"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
