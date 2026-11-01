"""Group exact assistant replies/reasoning in the fixed identity review pool."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from inspect_sft_identity import NAME_PATTERNS, QUESTION
from sft_candidate_lineage import iter_candidate_rows


ROOT = Path(__file__).resolve().parents[1]


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value):
    return digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def assistant_indices(messages, locations):
    selected = set()
    for location in locations:
        index = location["message_index"]
        if messages[index]["role"] == "assistant":
            selected.add(index)
        elif messages[index]["role"] == "user":
            for following in range(index + 1, len(messages)):
                if messages[following]["role"] == "user":
                    break
                if messages[following]["role"] == "assistant":
                    selected.add(following)
                    break
    return sorted(selected)


def reasoning_variant(message):
    if "reasoning_content" not in message:
        return "missing"
    value = message["reasoning_content"]
    if value is None:
        return "null"
    return "text:" + digest_text(value)


def add_group(groups, text, occurrence, counterpart):
    if not isinstance(text, str) or not text:
        return False
    key = digest_text(text)
    if key not in groups:
        groups[key] = {"text": text, "occurrences": 0, "record_lines": set(),
                       "prompt_keys": set(), "context_keys": set(), "counterparts": set(),
                       "message_keys": set(), "cue_counts": Counter(), "examples": defaultdict(list)}
    group = groups[key]
    group["occurrences"] += 1
    group["record_lines"].add(occurrence["candidate_line"])
    group["prompt_keys"].add(occurrence["user_prompt_sha256"])
    group["context_keys"].add(occurrence["context_sha256"])
    group["counterparts"].add(counterpart)
    group["message_keys"].add(occurrence["message_sha256"])
    cue = occurrence["prompt_cue"]
    group["cue_counts"][cue] += 1
    examples = group["examples"][cue]
    if len(examples) < 3 and all(item["context_sha256"] != occurrence["context_sha256"] for item in examples):
        examples.append(occurrence)
    return True


def group_summary(key, group, preview=True):
    result = {"group_sha256": key, "occurrences": group["occurrences"],
              "record_count": len(group["record_lines"]),
              "nearest_user_prompt_variants": len(group["prompt_keys"]),
              "context_variants": len(group["context_keys"]),
              "counterpart_variants": len(group["counterparts"]),
              "full_message_variants": len(group["message_keys"]),
              "prompt_cue_occurrences": dict(sorted(group["cue_counts"].items())),
              "examples": [item for cue in sorted(group["examples"]) for item in group["examples"][cue]]}
    result["text_preview" if preview else "text"] = group["text"][:160] if preview else group["text"]
    return result


def collect_groups(candidate, inventory, index_path):
    groups = {"reply": {}, "reasoning": {}}
    counts = Counter()
    index_hash = hashlib.sha256()
    with index_path.open("rb") as stream:
        index_raw = stream.readline()
        entry = json.loads(index_raw) if index_raw else None
        for line, raw, trace in iter_candidate_rows(
                Path(candidate["outputs"]["candidate"]["path"]),
                Path(candidate["outputs"]["lineage"]["path"]),
                candidate["outputs"]["candidate"], candidate["outputs"]["lineage"]):
            counts["candidate_records_checked"] += 1
            if entry is None or entry["candidate_line"] != line:
                if entry is not None and entry["candidate_line"] < line:
                    raise ValueError("unordered identity index")
                continue
            if (entry["source_line"] != trace["source_line"]
                    or entry["source_line_sha256"] != trace["source_line_sha256"]):
                raise ValueError("identity index source mapping mismatch")
            counts["indexed_records_checked"] += 1
            messages = json.loads(raw)["conversations"]
            indices = assistant_indices(messages, entry["locations"])
            counts["indexed_records_with_selected_assistant_turns"] += bool(indices)
            for index in indices:
                message = messages[index]
                previous_users = [i for i in range(index) if messages[i]["role"] == "user"]
                user_index = previous_users[-1] if previous_users else None
                prompt = messages[user_index]["content"] if user_index is not None else None
                cue = ("identity_question_cue" if QUESTION.search(prompt) else "no_identity_question_cue") if prompt is not None else "no_previous_user"
                occurrence = {"candidate_line": line, "source_line": trace["source_line"],
                              "source_line_sha256": trace["source_line_sha256"], "message_index": index,
                              "user_message_index": user_index, "prompt_cue": cue,
                              "user_prompt_sha256": digest_text(prompt) if prompt is not None else None,
                              "user_prompt_preview": prompt[:160] if prompt is not None else None,
                              "context_sha256": canonical_hash(messages[:index]),
                              "message_sha256": canonical_hash(message),
                              "reasoning_variant": reasoning_variant(message)}
                counts["selected_assistant_turns"] += 1
                body = message["content"]
                if add_group(groups["reply"], body, occurrence, occurrence["reasoning_variant"]):
                    counts["reply_occurrences"] += 1
                else:
                    counts["empty_reply_occurrences_skipped"] += 1
                if add_group(groups["reasoning"], message.get("reasoning_content"), occurrence, digest_text(body)):
                    counts["reasoning_occurrences"] += 1
            index_hash.update(index_raw)
            index_raw = stream.readline()
            entry = json.loads(index_raw) if index_raw else None
        if entry is not None:
            raise ValueError("identity index contains unconsumed records")
    if (index_hash.hexdigest() != inventory["files"]["review_index"]["sha256"]
            or counts["indexed_records_checked"] != inventory["review_hit_records"]):
        raise ValueError("identity index identity/count mismatch")
    return groups, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--inventory-report", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1 or args.details_dir.exists() or args.report_dir.exists():
        raise ValueError("top-k must be positive and output directories must be fresh")
    start = time.perf_counter()
    candidate_bytes, inventory_bytes = args.candidate_report.read_bytes(), args.inventory_report.read_bytes()
    candidate, inventory = json.loads(candidate_bytes), json.loads(inventory_bytes)
    if (hashlib.sha256(candidate_bytes).hexdigest() != inventory["candidate_report_sha256"]
            or candidate["outputs"]["candidate"]["sha256"] != inventory["input"]["sha256"]):
        raise ValueError("candidate and identity inventory refer to different inputs")
    code_paths = {"entry": Path(__file__), "identity_inventory": ROOT / "scripts/inspect_sft_identity.py",
                  "lineage_reader": ROOT / "scripts/sft_candidate_lineage.py"}
    code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in code_paths.items()}
    if (code_hashes["identity_inventory"] != inventory["code_sha256"]["entry"]
            or code_hashes["lineage_reader"] != inventory["code_sha256"]["lineage_reader"]):
        raise ValueError("identity cue rules or lineage reader changed since inventory")
    groups, counts = collect_groups(candidate, inventory, Path(inventory["files"]["review_index"]["path"]))
    args.details_dir.mkdir(parents=True, exist_ok=False)
    ranked = {kind: sorted(values.items(), key=lambda item: (-item[1]["occurrences"], item[0]))
              for kind, values in groups.items()}
    upstream_replies = [(key, group) for key, group in ranked["reply"] if any(
        NAME_PATTERNS[name].search(group["text"]) for name in ("upstream_author", "upstream_model"))]
    files = {}
    for kind, ordered in ranked.items():
        path = args.details_dir / f"{kind}_groups.jsonl"
        digest = hashlib.sha256()
        with path.open("xb") as stream:
            for key, group in ordered:
                payload = (json.dumps(group_summary(key, group, preview=False), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                stream.write(payload)
                digest.update(payload)
        files[kind + "_groups"] = {"path": str(path.resolve()), "records": len(ordered), "sha256": digest.hexdigest()}
    tops = {"reply": ranked["reply"][:args.top_k], "reasoning": ranked["reasoning"][:args.top_k],
            "upstream_reply": upstream_replies[:args.top_k]}
    requested = defaultdict(list)
    for kind, ordered in tops.items():
        for key, group in ordered:
            for example in group_summary(key, group)["examples"]:
                requested[example["candidate_line"]].append({"kind": kind, "group_sha256": key, **example})
    example_path = args.details_dir / "top_group_contexts.jsonl"
    example_digest = hashlib.sha256()
    found = set()
    with example_path.open("xb") as stream:
        for line, raw, trace in iter_candidate_rows(Path(inventory["input"]["path"]),
                Path(candidate["outputs"]["lineage"]["path"]), inventory["input"], candidate["outputs"]["lineage"]):
            if line in requested:
                packet = {"candidate_line": line, "source_line": trace["source_line"],
                          "source_line_sha256": trace["source_line_sha256"],
                          "group_links": requested[line], "record": json.loads(raw)}
                payload = (json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                stream.write(payload)
                example_digest.update(payload)
                found.add(line)
    if found != set(requested):
        raise ValueError("not all contrast examples were found")
    files["top_group_contexts"] = {"path": str(example_path.resolve()), "records": len(found), "sha256": example_digest.hexdigest()}
    if (args.candidate_report.read_bytes() != candidate_bytes or args.inventory_report.read_bytes() != inventory_bytes
            or any(hashlib.sha256(path.read_bytes()).hexdigest() != code_hashes[key] for key, path in code_paths.items())):
        raise ValueError("inputs or grouping code changed during run")
    coverage = {}
    for kind, ordered in tops.items():
        coverage[kind] = {"groups": len(ordered), "occurrences": sum(group["occurrences"] for _, group in ordered),
                          "unique_records": len(set().union(*(group["record_lines"] for _, group in ordered)))}
    report = {"schema_version": 1, "status": "identity_exact_templates_grouped_no_rewrite",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "input": inventory["input"],
              "candidate_report_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
              "inventory_report_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
              "index_sha256": inventory["files"]["review_index"]["sha256"], "code_sha256": code_hashes,
              "counts": {**counts, "exact_reply_groups": len(groups["reply"]),
                         "exact_reasoning_groups": len(groups["reasoning"]), "upstream_reply_groups": len(upstream_replies),
                         "upstream_reply_occurrences": sum(group["occurrences"] for _, group in upstream_replies),
                         "upstream_reply_records": len(set().union(*(group["record_lines"] for _, group in upstream_replies)))},
              "top_k": args.top_k, "top_k_coverage": coverage,
              "top_groups": {kind: [group_summary(key, group) for key, group in ordered] for kind, ordered in tops.items()},
              "files": files,
              "scope": {"source_modified": False, "rewrite_applied": False, "filtering_applied": False,
                        "selection": "indexed assistant turns and first assistant after indexed user cues; not all assistant turns in the corpus",
                        "exact_text_without_whitespace_normalization": True, "approximate_clustering_applied": False,
                        "prompt_cues_are_semantic_verdicts": False, "reasoning_missing_null_empty_distinguished": True,
                        "examples": "first up to three distinct contexts per prompt-cue class, contrastive not random",
                        "data_recipe_frozen": False}, "elapsed_seconds": time.perf_counter() - start}
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "top_k_coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
