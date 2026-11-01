"""Independently verify source accounting, split isolation and a candidate contract."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from minimind_lab.data.sft import encode_sft_conversation, SFTDataset, SFTCollator
from prepare_sft_recipe import canonical, digest, normalize, normalize_tree


def check_descriptor(desc):
    h = hashlib.sha256()
    count = size = 0
    with Path(desc["path"]).open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            h.update(raw)
            size += len(raw)
    if h.hexdigest() != desc["sha256"] or count != desc["records"] or size != desc["bytes"]:
        raise ValueError(f"artifact identity mismatch: {desc['path']}")


def run(contract_path, output, load_datasets=False):
    if not __debug__:
        raise RuntimeError("verification requires Python without -O so checks cannot be disabled")
    if output.exists():
        raise FileExistsError("use a new verification output")
    start = time.perf_counter()
    contract_raw = contract_path.read_bytes()
    contract = json.loads(contract_raw)
    policy_raw = Path(contract["policy"]["path"]).read_bytes()
    if digest(policy_raw) != contract["policy"]["sha256"]:
        raise ValueError("policy identity mismatch")
    policy = json.loads(policy_raw)
    report_raw = Path(contract["preparation_report"]["path"]).read_bytes()
    if digest(report_raw) != contract["preparation_report"]["sha256"]:
        raise ValueError("preparation report identity mismatch")
    report = json.loads(report_raw)
    if report["outputs"] != contract["artifacts"]:
        raise ValueError("report/contract artifact mismatch")
    root = Path(__file__).resolve().parents[1]
    for name, expected in contract["pipeline_identity"]["runtime_files_sha256"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"runtime identity changed: {name}")
    tokenizer_dir = root / "assets/minimind_tokenizer"
    for name, expected in contract["pipeline_identity"]["tokenizer_files_sha256"].items():
        if digest((tokenizer_dir / name).read_bytes()) != expected:
            raise ValueError(f"tokenizer identity changed: {name}")
    for path, expected in report["fixed_inputs_sha256"].items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError(f"preparation input changed: {path}")
    source = contract["input_candidate"]
    check_descriptor(source)
    for desc in contract["artifacts"].values():
        check_descriptor(desc)
    artifacts = contract["artifacts"]
    actions = Counter()
    with ExitStack() as stack:
        original = stack.enter_context(Path(source["path"]).open("rb"))
        ledger = stack.enter_context(Path(artifacts["lineage"]["path"]).open("rb"))
        streams = {name: stack.enter_context(Path(artifacts[name]["path"]).open("rb"))
                   for name in ("accepted", "review_hold", "duplicates")}
        accepted = 0
        for number, pair in enumerate(zip_longest(original, ledger), 1):
            raw, entry = pair
            if raw is None or entry is None:
                raise ValueError("lineage/input length mismatch")
            row = json.loads(entry)
            assert row["candidate_line"] == number
            assert digest(raw) == row["source_line_sha256"]
            action = row["action"]
            actions[action] += 1
            destination = "review_hold" if action == "hold" else "duplicates" if action == "duplicate" else "accepted"
            result = next(streams[destination])
            assert digest(result) == row["output_record_sha256"]
            if action in ("hold", "keep"):
                assert result == raw
            elif action in ("rewrite_identity", "duplicate"):
                original_record, changed = json.loads(raw), json.loads(result)
                if original_record != changed:
                    assert len(original_record["conversations"]) == 2
                    assert original_record["conversations"][0] == changed["conversations"][0]
                    expected = json.loads(raw)
                    expected["conversations"][1]["content"] = policy["identity"]["reply"]
                    expected["conversations"][1]["reasoning_content"] = ""
                    assert changed == expected
            else:
                raise ValueError("unknown action")
            if destination == "accepted":
                accepted += 1
                assert row["accepted_line"] == accepted
        for stream in streams.values():
            assert stream.read() == b""
    assert dict(actions) == report["actions"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    development = set()
    for path in report["fixed_inputs_sha256"]:
        if path.endswith(".jsonl"):
            for raw in Path(path).read_text(encoding="utf-8").splitlines():
                development.add(json.loads(raw)["source_line"])
    prompt_splits, normalized_splits, group_splits = {}, {}, {}
    conversations = set()
    dataset_reference = {}
    selected_indices = {name: {0, contract["split_statistics"][name]["records"] // 2,
                              contract["split_statistics"][name]["records"] - 1}
                        for name in ("train", "validation")}
    counts, totals, strata = Counter(), Counter(), {name: Counter() for name in ("train", "validation")}
    validation_reencoded = 0
    development_records_in_validation = 0
    with ExitStack() as stack:
        inputs = [stack.enter_context(Path(artifacts[name]["path"]).open("rb"))
                  for name in ("accepted", "encoding_index", "split_index")]
        outputs = {name: stack.enter_context(Path(artifacts[name]["path"]).open("rb"))
                   for name in ("train", "validation")}
        for index, packed in enumerate(zip_longest(*inputs), 1):
            raw, metadata_raw, split_raw = packed
            assert all(value is not None for value in packed)
            metadata, split = json.loads(metadata_raw), json.loads(split_raw)
            assert metadata["accepted_line"] == split["accepted_line"] == index
            assert metadata["source_line"] == split["source_line"]
            assert digest(raw) == split["record_sha256"]
            part = split["split"]
            assert next(outputs[part]) == raw
            record = json.loads(raw)
            convo = record["conversations"]
            key = digest(canonical(convo).encode())
            assert key not in conversations
            conversations.add(key)
            for keyspace, key in [(group_splits, split["group_id"]),
                                  (normalized_splits, digest(canonical(normalize_tree(convo)).encode()))]:
                assert key not in keyspace or keyspace[key] == part
                keyspace[key] = part
            for message in convo:
                if message["role"] == "user" and normalize(message["content"]):
                    key = digest(normalize(message["content"]).encode())
                    assert key not in prompt_splits or prompt_splits[key] == part
                    prompt_splits[key] = part
            if split["source_line"] in development:
                development_records_in_validation += part == "validation"
                assert part == "train"
            assert 2 <= metadata["input_tokens"] <= policy["max_length"]
            assert metadata["targets_after_shift"] > 0
            if counts[part] in selected_indices[part]:
                dataset_reference[part, counts[part]] = encode_sft_conversation(convo, tokenizer)
            counts[part] += 1
            totals[part] += metadata["targets_after_shift"]
            strata[part][metadata["stratum"]] += 1
            if part == "validation":
                feature = encode_sft_conversation(convo, tokenizer)
                assert digest(canonical(feature).encode()) == metadata["encoding_sha256"]
                assert len(feature["input_ids"]) == metadata["input_tokens"]
                assert sum(v != -100 for v in feature["labels"][1:]) == metadata["targets_after_shift"]
                validation_reencoded += 1
        for stream in outputs.values():
            assert stream.read() == b""
    for part in ("train", "validation"):
        summary = contract["split_statistics"][part]
        assert counts[part] == summary["records"]
        assert totals[part] == summary["total_targets_after_shift"]
        assert dict(strata[part]) == summary["strata"]
    dataset_checks = {}
    if load_datasets:
        for part in ("train", "validation"):
            dataset = SFTDataset(artifacts[part]["path"], tokenizer)
            assert len(dataset) == counts[part]
            indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
            for index in indices:
                assert dataset[index] == dataset_reference[part, index]
            batch = SFTCollator(tokenizer, policy["max_length"])([dataset[i] for i in indices])
            assert batch["input_ids"].shape == (len(indices), policy["max_length"])
            dataset_checks[part] = {"records_loaded": len(dataset), "selected_indices_encoded": indices,
                                    "collator_shape": list(batch["input_ids"].shape)}
    result = {"status": "passed", "contract_sha256": digest(contract_raw),
              "all_artifact_hashes_counts_and_sizes_verified": True,
              "all_input_actions_and_written_bytes_verified": True,
              "exact_conversation_duplicates_after_rewrite": 0,
              "normalized_prompt_cross_split_overlap": 0,
              "normalized_conversation_cross_split_overlap": 0,
              "development_records_in_validation": development_records_in_validation,
              "validation_records_independently_reencoded": validation_reencoded,
              "splits": dict(counts), "targets_after_shift": dict(totals), "dataset_checks": dataset_checks,
              "semantic_factual_quality_verified": False, "training_started": False,
              "verifier_sha256": digest(Path(__file__).read_bytes()),
              "elapsed_seconds": time.perf_counter() - start}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-datasets", action="store_true")
    args = parser.parse_args()
    run(args.contract, args.output, args.load_datasets)
