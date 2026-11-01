"""Build a reversible, verified SFT data candidate; never start training or approve it."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import time
import unicodedata

from transformers import AutoTokenizer

from inspect_sft_encoding import distribution, sha256_file
from inspect_sft_full_encoding import classify_record
from minimind_lab.data.sft import prepare_sft_messages, encode_sft_conversation
from minimind_lab.data.sft_protocol import check_sft_protocol
from sft_candidate_lineage import iter_candidate_rows


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = POLICY = KNOWN_REPLIES = None
UPSTREAM = re.compile(r"mini[ _-]?mind|jingyaogong", re.I)
OTHER_SELF = re.compile(
    r"(?:我是|我叫|我由|作为|i am|i'm|i was developed)[^。！？\n]{0,36}"
    r"(?:chatgpt|openai|deepseek|qwen|通义千问|claude|anthropic)", re.I)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_question(text):
    text = normalize(text)
    text = re.sub(r"[\s，,。.!！?？:：;；、]", "", text)
    text = re.sub(r"^(?:你好|您好|嗨|hello|hi)(?:qwen|通义千问)?", "", text)
    return text


def normalize_tree(value):
    if isinstance(value, str):
        return normalize(value)
    if isinstance(value, list):
        return [normalize_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_tree(item) for key, item in value.items()}
    return value


def identity_action(record, policy, known_replies):
    messages = record["conversations"]
    known = any(m["role"] == "assistant" and digest(m["content"].encode()) in known_replies for m in messages)
    direct = (len(messages) == 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant"
              and normalized_question(messages[0]["content"]) in policy["identity"]["allowed_normalized_questions"])
    if known and direct:
        # Preserve user text; only replace a bounded, known self-introduction turn.
        result = json.loads(json.dumps(record, ensure_ascii=False))
        result["conversations"][1]["content"] = policy["identity"]["reply"]
        result["conversations"][1]["reasoning_content"] = ""
        return result, "rewrite_identity", []
    texts = [m.get(field, "") for m in messages for field in ("content", "reasoning_content")]
    if any(isinstance(text, str) and UPSTREAM.search(text) for text in texts):
        return record, "hold", ["identity_upstream_context_requires_review"]
    if any(OTHER_SELF.search(m.get(field) or "") for m in messages if m["role"] == "assistant"
           for field in ("content", "reasoning_content")):
        return record, "hold", ["possible_third_party_self_claim_requires_review"]
    return record, "keep", []


def init_worker(tokenizer_dir, policy, known_replies):
    global TOKENIZER, POLICY, KNOWN_REPLIES
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    POLICY, KNOWN_REPLIES = policy, set(known_replies)


def process_record(item):
    number, raw, trace = item
    record = json.loads(raw)
    result = {"candidate_line": number, "source_line": trace["source_line"],
              "source_line_sha256": trace["source_line_sha256"], "action": "keep",
              "issues": [], "notices": []}
    try:
        prepared, tools = prepare_sft_messages(record["conversations"])
        issues, notices = check_sft_protocol(prepared, tools)
        result["notices"] = notices
        if issues:
            result.update(action="hold", issues=issues)
        elif str(trace["source_line"]) in POLICY.get("exact_review_holds", {}):
            result.update(action="hold", issues=[{"code": "exact_content_review_hold",
                                                  "reason": POLICY["exact_review_holds"][str(trace["source_line"]) ]}])
        else:
            record, action, codes = identity_action(record, POLICY, KNOWN_REPLIES)
            result.update(action=action, issues=[{"code": code} for code in codes])
        if result["action"] != "hold":
            feature = encode_sft_conversation(record["conversations"], TOKENIZER)
            n = len(feature["input_ids"])
            if n > POLICY["max_length"]:
                result.update(action="hold", issues=[{"code": "overlength_after_processing"}])
            else:
                result.update(input_tokens=n, targets_after_shift=sum(v != -100 for v in feature["labels"][1:]),
                              encoding_sha256=digest(canonical(feature).encode()),
                              stratum=classify_record(record))
    except ValueError as exc:
        result.update(action="hold", issues=[{"code": "runtime_encoding_or_structure_rejected",
                                               "message": str(exc)[:500]}])
    payload = (canonical(record) + "\n").encode() if result["action"] == "rewrite_identity" else raw
    return result, payload, record


def process_chunk(items):
    return [process_record(item) for item in items]


class Sink:
    def __init__(self, path):
        self.path = path
        self.partial = path.with_name(path.name + ".partial")
        self.stream = self.partial.open("xb")
        self.hash = hashlib.sha256()
        self.records = self.bytes = 0

    def write(self, value):
        raw = value if isinstance(value, bytes) else (canonical(value) + "\n").encode()
        self.stream.write(raw)
        self.hash.update(raw)
        self.records += 1
        self.bytes += len(raw)

    def close(self):
        self.stream.close()
        self.partial.rename(self.path)
        return {"path": str(self.path.resolve()), "records": self.records,
                "bytes": self.bytes, "sha256": self.hash.hexdigest()}


class Groups:
    def __init__(self):
        self.parent = []
        self.size = []
        self.first_source = []
        self.development = []
        self.keys = {}

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.first_source[a] = min(self.first_source[a], self.first_source[b])
        self.development[a] |= self.development[b]

    def add(self, record, source_line, development):
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        self.first_source.append(source_line)
        self.development.append(development)
        keys = {digest(("user:" + normalize(m["content"])).encode()) for m in record["conversations"]
                if m["role"] == "user" and normalize(m["content"])}
        keys.add(digest(("conversation:" + canonical(normalize_tree(record["conversations"]))).encode()))
        for key in keys:
            if key in self.keys:
                self.union(index, self.keys[key])
            else:
                self.keys[key] = index

    def validation_roots(self, target, seed):
        roots = [i for i in range(len(self.parent)) if self.find(i) == i and not self.development[i]]
        roots.sort(key=lambda i: digest(f"{seed}:{self.first_source[i]}".encode()))
        selected = set()
        remaining = target
        for root in roots:
            if self.size[root] <= remaining:
                selected.add(root)
                remaining -= self.size[root]
            if not remaining:
                return selected
        raise ValueError("not enough eligible whole groups for requested validation budget")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def code_identity(tokenizer_dir):
    code = [Path(__file__), ROOT / "scripts/sft_candidate_lineage.py", ROOT / "scripts/inspect_sft_encoding.py",
            ROOT / "scripts/inspect_sft_full_encoding.py", ROOT / "src/minimind_lab/data/sft.py",
            ROOT / "src/minimind_lab/data/sft_protocol.py", ROOT / "src/minimind_lab/data/validator.py",
            ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {"runtime_files_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in code},
            "tokenizer_files_sha256": {p.relative_to(tokenizer_dir).as_posix(): sha256_file(p)
                                       for p in sorted(tokenizer_dir.rglob("*")) if p.is_file()}}


def bounded_results(pool, source, chunk_size, workers):
    pending = deque()
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < workers * 2:
            chunk = []
            for _ in range(chunk_size):
                try:
                    chunk.append(next(source))
                except StopIteration:
                    exhausted = True
                    break
            if chunk:
                pending.append(pool.submit(process_chunk, chunk))
        if pending:
            yield from pending.popleft().result()


def run(args):
    start = time.perf_counter()
    if args.output_dir.exists() or args.report_dir.exists() or args.contract.exists():
        raise FileExistsError("use fresh output, report and contract paths")
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("positive worker and chunk counts required")
    fixed_paths = [args.candidate_report, args.policy, args.groups_report, *args.development_samples]
    fixed_hashes = {str(p.resolve()): sha256_file(p) for p in fixed_paths}
    reference = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    groups_report = json.loads(args.groups_report.read_text(encoding="utf-8"))
    source = reference["outputs"]["candidate"]
    if policy.get("input_candidate_sha256", source["sha256"]) != source["sha256"]:
        raise ValueError("policy refers to a different input candidate")
    if groups_report["input"]["sha256"] != source["sha256"]:
        raise ValueError("identity groups refer to a different candidate")
    known = [r["group_sha256"] for r in groups_report["top_groups"]["upstream_reply"]]
    development = set()
    for path in args.development_samples:
        for raw in path.read_text(encoding="utf-8").splitlines():
            development.add(json.loads(raw)["source_line"])
    before = code_identity(args.tokenizer_dir)
    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_sha256": source["sha256"], "training_started": False}
    write_json(args.report_dir / "run_manifest.json", manifest)
    try:
        sinks = {key: Sink(args.output_dir / filename) for key, filename in {
            "accepted": "accepted.jsonl", "review_hold": "review_hold.jsonl",
            "duplicates": "postrewrite_duplicates.jsonl", "lineage": "lineage.jsonl",
            "encoding_index": "encoding_index.jsonl"}.items()}
        counts, issues, notices = Counter(), Counter(), Counter()
        exact = {}
        groups = Groups()
        iterator = iter_candidate_rows(Path(source["path"]), Path(reference["outputs"]["lineage"]["path"]),
                                       source, reference["outputs"]["lineage"])
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker,
                                 initargs=(str(args.tokenizer_dir.resolve()), policy, known)) as pool:
            for row, raw, record in bounded_results(pool, iter(iterator), args.chunk_size, args.workers):
                notices.update(row["notices"])
                issues.update({issue["code"] for issue in row["issues"]})
                if row["action"] != "hold":
                    conversation_hash = digest(canonical(record["conversations"]).encode())
                    if conversation_hash in exact:
                        row.update(action="duplicate", duplicate_of_accepted_line=exact[conversation_hash])
                    else:
                        row["accepted_line"] = sinks["accepted"].records + 1
                        exact[conversation_hash] = row["accepted_line"]
                        groups.add(record, row["source_line"], row["source_line"] in development)
                        sinks["accepted"].write(raw)
                        sinks["encoding_index"].write({k: row[k] for k in (
                            "accepted_line", "source_line", "stratum", "input_tokens",
                            "targets_after_shift", "encoding_sha256")})
                row["output_record_sha256"] = digest(raw)
                counts[row["action"]] += 1
                if row["action"] in ("hold", "duplicate"):
                    sinks["review_hold" if row["action"] == "hold" else "duplicates"].write(raw)
                sinks["lineage"].write(row)
                if row["candidate_line"] % 50000 == 0:
                    print(json.dumps({"processed": row["candidate_line"], "actions": counts}), flush=True)
        outputs = {key: sink.close() for key, sink in sinks.items()}
        if outputs["lineage"]["records"] != source["records"]:
            raise ValueError("incomplete input accounting")
        selected = groups.validation_roots(policy["collection"]["validation_records_target"],
                                           policy["collection"]["split_seed"])
        partitions = {name: Sink(args.output_dir / f"{name}.jsonl") for name in ("train", "validation")}
        split_index = Sink(args.output_dir / "split_index.jsonl")
        samples = Sink(args.output_dir / "quality_review_sample.jsonl")
        stats = {name: {"records": 0, "strata": Counter(), "lengths": [], "targets": []}
                 for name in partitions}
        rngs = {name: random.Random(f"20261031:{name}") for name in ("plain", "reasoning_only", "tool_only")}
        reservoirs = defaultdict(list)
        seen_by_stratum = Counter()
        with Path(outputs["accepted"]["path"]).open("rb") as records, \
                Path(outputs["encoding_index"]["path"]).open("rb") as metadata:
            for index, (raw, meta) in enumerate(zip_longest(records, metadata)):
                if raw is None or meta is None:
                    raise ValueError("encoding index length mismatch")
                row = json.loads(meta)
                assert row["accepted_line"] == index + 1
                root = groups.find(index)
                partition = "validation" if root in selected else "train"
                partitions[partition].write(raw)
                group_id = groups.first_source[root]
                split_index.write({"accepted_line": index + 1, "source_line": row["source_line"],
                                   "group_id": group_id, "split": partition,
                                   "record_sha256": digest(raw)})
                entry = stats[partition]
                entry["records"] += 1
                entry["strata"][row["stratum"]] += 1
                entry["lengths"].append(row["input_tokens"])
                entry["targets"].append(row["targets_after_shift"])
                if partition == "train" and row["stratum"] in rngs:
                    kind = row["stratum"]
                    seen_by_stratum[kind] += 1
                    packet = {"source_line": row["source_line"], "group_id": group_id,
                              "stratum": kind, "record": json.loads(raw)}
                    pool_samples = reservoirs[kind]
                    if len(pool_samples) < 8:
                        pool_samples.append(packet)
                    else:
                        slot = rngs[kind].randrange(seen_by_stratum[kind])
                        if slot < 8:
                            pool_samples[slot] = packet
        for kind in sorted(reservoirs):
            for sample in reservoirs[kind]:
                samples.write(sample)
        outputs.update({name: sink.close() for name, sink in partitions.items()})
        outputs["split_index"] = split_index.close()
        outputs["quality_review_sample"] = samples.close()
        summary = {}
        for split, entry in stats.items():
            summary[split] = {"records": entry["records"], "strata": dict(entry["strata"]),
                              "input_length": distribution(entry["lengths"]),
                              "targets_after_shift": distribution(entry["targets"]),
                              "total_input_tokens": sum(entry["lengths"]),
                              "total_targets_after_shift": sum(entry["targets"])}
        after = code_identity(args.tokenizer_dir)
        if after != before or any(sha256_file(Path(p)) != h for p, h in fixed_hashes.items()):
            raise ValueError("processing inputs or code changed during preparation")
        for output in outputs.values():
            if sha256_file(Path(output["path"])) != output["sha256"]:
                raise ValueError("output hash mismatch")
        report = {"status": "candidate_prepared_pending_semantic_and_user_acceptance", "actions": dict(counts),
                  "input": source, "original_source": reference["original_source"],
                  "issues_by_record": dict(issues), "notices_by_record": dict(notices),
                  "outputs": outputs, "split_summary": summary,
                  "grouping": {"records": len(groups.parent), "groups": sum(groups.find(i) == i for i in range(len(groups.parent))),
                               "largest_group": max(groups.size[i] for i in range(len(groups.parent)) if groups.find(i) == i),
                               "validation_groups": len(selected), "development_source_lines": len(development),
                               "no_component_cross_split": True, "semantic_paraphrase_isolation_claimed": False},
                  "fixed_inputs_sha256": fixed_hashes, "pipeline_identity": before,
                  "elapsed_seconds": time.perf_counter() - start}
        write_json(args.report_dir / "report.json", report)
        contract = {"schema_version": 1, "contract_id": "minimind_mini_sft_data_768_v1_candidate",
                    "status": "candidate_pending_review_not_frozen", "ready_for_training": False,
                    "training_started": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": reference["original_source"], "input_candidate": source,
                    "policy": {"path": str(args.policy.resolve()), "sha256": sha256_file(args.policy)},
                    "preparation_report": {"path": str((args.report_dir / 'report.json').resolve()),
                                           "sha256": sha256_file(args.report_dir / 'report.json')},
                    "artifacts": outputs, "split_statistics": summary, "grouping": report["grouping"],
                    "encoding": {"max_length": policy["max_length"], "add_special_tokens": False,
                                 "chat_template": "tokenizer_config.json exact hashed template",
                                 "assistant_only": True, "role_headers_supervised": False,
                                 "thinking_and_tool_calls_supervised": True,
                                 "tool_results_supervised": False, "end_sequence": "<|im_end|>\\n",
                                 "padding": "right_to_768", "padding_labels": -100,
                                 "loss_alignment": "logits[..., :-1, :] vs labels[..., 1:]",
                                 "truncation": False, "zero_target_records": "reject",
                                 "reasoning": policy["reasoning"]},
                    "pipeline_identity": before,
                    "runtime": {name: importlib.metadata.version(name) for name in (
                        "transformers", "tokenizers", "jsonschema", "torch", "numpy")},
                    "limits": policy["acceptance_limits"],
                    "approval": {"identity_policy_previously_confirmed": True,
                                 "new_processing_and_split_choices_reviewed": False,
                                 "independent_output_verification_completed": False,
                                 "factual_quality_of_all_records_verified": False,
                                 "public_redistribution_permissions_resolved": False}}
        args.contract.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.contract, contract)
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        actions=dict(counts), ready_for_training=False)
        write_json(args.report_dir / "run_manifest.json", manifest)
        print(json.dumps({"actions": counts, "splits": summary, "grouping": report["grouping"]}, indent=2), flush=True)
    except BaseException as exc:
        for sink in locals().get("sinks", {}).values():
            sink.stream.close()
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        write_json(args.report_dir / "run_manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--groups-report", type=Path, required=True)
    parser.add_argument("--development-samples", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
