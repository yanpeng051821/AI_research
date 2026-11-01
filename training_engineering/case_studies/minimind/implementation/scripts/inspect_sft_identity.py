"""Inventory identity-related lexical cues; do not rewrite or accept SFT records."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

from sft_candidate_lineage import iter_candidate_rows


ROOT = Path(__file__).resolve().parents[1]
NAME_PATTERNS = {
    "upstream_author": re.compile(r"jingyaogong", re.I),
    "upstream_model": re.compile(r"mini[ _-]?mind", re.I),
    "target_trainer": re.compile(r"yanpeng", re.I),
    "other_model_or_provider": re.compile(r"chatgpt|openai|deepseek|qwen|通义千问|claude|anthropic", re.I),
}
QUESTION = re.compile(
    r"你是谁|你叫什么|你的(?:名字|名称|身份|作者|开发者|开发团队|训练者|创造者|真实来源|模型)|"
    r"谁(?:开发|训练|创建|创造)(?:了)?你|你.{0,12}(?:由谁|谁开发|谁训练|哪家公司)|"
    r"who (?:are you|created you|developed you|trained you)|which (?:model|company)", re.I)
SELF = re.compile(
    r"我(?:是|叫|由)|我的(?:名字|身份|开发者|训练者)|作为.{0,16}(?:AI|语言模型|人工智能)|"
    r"\bi am\b|\bi'm\b|\bi was (?:trained|developed|created)\b|\bmy (?:name|developer)\b", re.I)
TRAINING = re.compile(r"我的训练数据|本模型的?训练数据|my training data", re.I)
SYSTEM = re.compile(r"你是|你的名字|you are|your name", re.I)
IDENTITY_SIGNALS = {"user_identity_question", "assistant_identity_phrase", "reasoning_identity_phrase",
                    "system_identity_instruction", "training_data_self_claim"}
BUCKETS = ("target_name_present", "upstream_with_identity_cue", "upstream_without_identity_cue",
           "other_name_with_identity_cue", "identity_cue_without_named_model",
           "other_name_without_identity_cue")


def inspect_messages(messages):
    signals = set()
    locations = []
    for index, message in enumerate(messages):
        role = message["role"]
        for field in ("content", "reasoning_content"):
            text = message.get(field)
            if not isinstance(text, str) or not text:
                continue
            hits = {name for name, pattern in NAME_PATTERNS.items() if pattern.search(text)}
            if role == "user" and QUESTION.search(text):
                hits.add("user_identity_question")
            if role == "assistant" and SELF.search(text):
                hits.add("reasoning_identity_phrase" if field == "reasoning_content" else "assistant_identity_phrase")
            if role == "system" and SYSTEM.search(text):
                hits.add("system_identity_instruction")
            if TRAINING.search(text):
                hits.add("training_data_self_claim")
            if hits:
                signals.update(hits)
                locations.append({"message_index": index, "role": role, "field": field,
                                  "signals": sorted(hits)})
    identity_cue = bool(signals & IDENTITY_SIGNALS)
    upstream = bool(signals & {"upstream_author", "upstream_model"})
    if "target_trainer" in signals:
        bucket = "target_name_present"
    elif upstream:
        bucket = "upstream_with_identity_cue" if identity_cue else "upstream_without_identity_cue"
    elif "other_model_or_provider" in signals:
        bucket = "other_name_with_identity_cue" if identity_cue else "other_name_without_identity_cue"
    elif identity_cue:
        bucket = "identity_cue_without_named_model"
    else:
        bucket = None
    return {"signals": sorted(signals), "review_bucket": bucket, "locations": locations}


def inventory(reference, details_dir, sample_size=8, seed=20261031):
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    details_dir.mkdir(parents=True, exist_ok=False)
    index_path = details_dir / "review_index.jsonl.partial"
    index_digest = hashlib.sha256()
    indexed = total = index_bytes = 0
    signals, fields = Counter(), Counter()
    buckets = Counter({name: 0 for name in BUCKETS})
    samples = {name: [] for name in BUCKETS}
    rngs = {name: random.Random(f"{seed}:{name}") for name in BUCKETS}
    with index_path.open("xb") as index_stream:
        for total, raw, trace in iter_candidate_rows(
                Path(reference["outputs"]["candidate"]["path"]),
                Path(reference["outputs"]["lineage"]["path"]),
                reference["outputs"]["candidate"], reference["outputs"]["lineage"]):
            record = json.loads(raw.decode("utf-8"))
            result = inspect_messages(record["conversations"])
            if result["review_bucket"]:
                indexed += 1
                bucket = result["review_bucket"]
                buckets[bucket] += 1
                signals.update(result["signals"])
                for location in result["locations"]:
                    fields.update(f"{location['role']}.{location['field']}:{signal}"
                                  for signal in location["signals"])
                packet = {"candidate_line": total, "source_line": trace["source_line"],
                          "source_line_sha256": trace["source_line_sha256"], **result}
                payload = (json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                index_stream.write(payload)
                index_digest.update(payload)
                index_bytes += len(payload)
                sample = {**packet, "record": record}
                if len(samples[bucket]) < sample_size:
                    samples[bucket].append(sample)
                else:
                    position = rngs[bucket].randrange(buckets[bucket])
                    if position < sample_size:
                        samples[bucket][position] = sample
            if total % 100000 == 0:
                print(f"Inventoried {total:,} candidate records", flush=True)
    final_index = details_dir / "review_index.jsonl"
    index_path.rename(final_index)
    sample_path = details_dir / "samples.jsonl"
    sample_payload = "".join(json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n"
                             for bucket in BUCKETS for packet in samples[bucket]).encode()
    sample_path.write_bytes(sample_payload)
    return {"records_checked": total, "review_hit_records": indexed, "no_cue_records": total - indexed,
            "records_by_signal": dict(sorted(signals.items())),
            "field_hits_by_signal": dict(sorted(fields.items())), "exclusive_review_buckets": dict(buckets),
            "sampling": {"seed": seed, "per_bucket_capacity": sample_size,
                         "selection": "seeded reservoir within each lexical bucket; not proportional or quality scoring"},
            "files": {
                "review_index": {"path": str(final_index.resolve()), "records": indexed, "bytes": index_bytes,
                                 "sha256": index_digest.hexdigest()},
                "samples": {"path": str(sample_path.resolve()), "records": sum(map(len, samples.values())),
                            "bytes": len(sample_payload), "sha256": hashlib.sha256(sample_payload).hexdigest()}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.details_dir.exists() or args.report_dir.exists():
        raise FileExistsError("use fresh detail and report directories")
    start = time.perf_counter()
    reference_bytes = args.candidate_report.read_bytes()
    reference = json.loads(reference_bytes)
    if reference["status"] != "confirmed_tool_quarantine_completed_not_final_training_data":
        raise ValueError("unsupported candidate report status")
    code_paths = {"entry": Path(__file__), "lineage_reader": ROOT / "scripts/sft_candidate_lineage.py"}
    code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in code_paths.items()}
    result = inventory(reference, args.details_dir, args.sample_size, args.seed)
    if (args.candidate_report.read_bytes() != reference_bytes or any(
            hashlib.sha256(path.read_bytes()).hexdigest() != code_hashes[key]
            for key, path in code_paths.items())):
        raise ValueError("candidate report or inspection code changed during scan")
    report = {"schema_version": 1, "status": "identity_lexical_inventory_completed_no_rewrite",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "input": reference["outputs"]["candidate"], "original_source": reference["original_source"],
              "candidate_report_sha256": hashlib.sha256(reference_bytes).hexdigest(), "code_sha256": code_hashes,
              "patterns": {**{name: pattern.pattern for name, pattern in NAME_PATTERNS.items()},
                           "question": QUESTION.pattern, "self": SELF.pattern,
                           "training": TRAINING.pattern, "system": SYSTEM.pattern},
              **result,
              "scope": {"source_modified": False, "identity_rewrite_applied": False,
                        "filtering_applied": False, "lexical_cues_not_semantic_verdicts": True,
                        "inventory_exhaustive_for_all_identity_expressions": False,
                        "not_scanned": ["top-level metadata", "tools/tool_calls payloads", "unlisted aliases"],
                        "content_quality_accepted": False, "data_recipe_frozen": False},
              "elapsed_seconds": time.perf_counter() - start}
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("records_checked", "review_hit_records", "records_by_signal", "exclusive_review_buckets")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
