"""Stream a fixed full SFT source through CPU encoding diagnostics, without filtering."""

import argparse
from array import array
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from heapq import heappush, heapreplace
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

from jinja2 import TemplateError
from transformers import AutoTokenizer

from inspect_sft_encoding import boundary_paths, distribution, sha256_file
from minimind_lab.data.sft import encode_sft_conversation, prepare_sft_messages
from minimind_lab.data.validator import validate_sft_record


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = None


def classify_record(record):
    reasoning = tool = False
    for message in record["conversations"]:
        content = message["content"]
        value = message.get("reasoning_content")
        reasoning |= (isinstance(value, str) and bool(value.strip())) or "<think>" in content or "</think>" in content
        tool |= message["role"] == "tool"
        for key in ("tools", "tool_calls"):
            value = message.get(key)
            tool |= bool(value) and not (isinstance(value, str) and not value.strip())
    return "reasoning_and_tool" if reasoning and tool else "reasoning_only" if reasoning else "tool_only" if tool else "plain"


def inspect_raw_record(source_line, raw, tokenizer):
    row = {"source_line": source_line, "source_line_sha256": hashlib.sha256(raw).hexdigest(),
           "stratum": "unknown", "issues": [], "input_tokens": None, "targets_after_shift": None}
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        row["issues"].append("invalid_utf8_or_json")
        return row
    issues = validate_sft_record(record)
    if issues:
        row["issues"].append("invalid_structure")
        row["structure_issues"] = issues
        return row
    row["stratum"] = classify_record(record)
    messages = record["conversations"]
    try:
        prepared, tools = prepare_sft_messages(messages)
    except ValueError as exc:
        row["issues"].append("field_conversion_error")
        row["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
        return row
    markers = (tokenizer.bos_token, tokenizer.eos_token)
    paths = boundary_paths(prepared, markers, "messages") + boundary_paths(tools, markers, "tools")
    if paths:
        row["issues"].append("potential_message_boundary_literal")
        row["boundary_paths"] = paths[:10]
        row["boundary_path_count"] = len(paths)
    try:
        feature = encode_sft_conversation(messages, tokenizer)
    except (ValueError, TypeError, TemplateError) as exc:
        row["issues"].append("encoding_error")
        row["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
        return row
    ids, labels = feature["input_ids"], feature["labels"]
    if len(ids) != len(labels):
        raise RuntimeError("encoder returned misaligned input_ids and labels")
    row["input_tokens"] = len(ids)
    row["targets_after_shift"] = sum(label != -100 for label in labels[1:])
    if not row["targets_after_shift"]:
        row["issues"].append("zero_targets_after_shift")
    return row


class Statistics:
    def __init__(self):
        self.records = 0
        self.flagged = 0
        self.issues = Counter()
        self.lengths = array("Q")
        self.targets = array("Q")

    def add(self, row):
        self.records += 1
        self.flagged += bool(row["issues"])
        self.issues.update(row["issues"])
        if row["input_tokens"] is not None:
            self.lengths.append(row["input_tokens"])
            self.targets.append(row["targets_after_shift"])

    def summary(self):
        shifted = sum(max(n - 1, 0) for n in self.lengths)
        targets = sum(self.targets)
        return {
            "records": self.records, "encoded_records": len(self.lengths),
            "records_without_encoding_flags": self.records - self.flagged,
            "flagged_records": self.flagged, "issues": dict(sorted(self.issues.items())),
            "input_tokens": distribution(self.lengths), "targets_after_shift": distribution(self.targets),
            "total_input_positions": sum(self.lengths), "total_shifted_positions": shifted,
            "total_targets_after_shift": targets,
            "supervised_fraction_of_shifted_positions": targets / shifted if shifted else None,
            "records_exceeding_length": {str(limit): sum(n > limit for n in self.lengths)
                                         for limit in (512, 768, 1024, 1536, 2048, 4096)},
            "statistics_scope": "encoded_records_including_flagged_records_not_accepted_training_data",
        }


def init_worker(tokenizer_dir):
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if not TOKENIZER.bos_token or not TOKENIZER.eos_token:
        raise ValueError("tokenizer must define BOS and EOS text")


def inspect_chunk(chunk):
    return [inspect_raw_record(index, raw, TOKENIZER) for index, raw in chunk]


def chunks(source, chunk_size, digest):
    chunk = []
    with source.open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            digest.update(raw)
            chunk.append((index, raw))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def pipeline_identity(tokenizer_dir):
    return {
        "code_sha256": {name: sha256_file(path) for name, path in {
            "full_inspection": Path(__file__),
            "sample_inspection_helpers": ROOT / "scripts/inspect_sft_encoding.py",
            "encoder": ROOT / "src/minimind_lab/data/sft.py",
            "validator": ROOT / "src/minimind_lab/data/validator.py",
        }.items()},
        "tokenizer_files_sha256": {
            path.relative_to(tokenizer_dir).as_posix(): sha256_file(path)
            for path in sorted(tokenizer_dir.rglob("*.json"))
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def run(args):
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk size must be positive")
    for path in (args.report_dir, args.details_dir):
        if path.exists():
            raise FileExistsError(f"use a fresh output directory: {path}")
    report_path, details_path = args.report_dir.resolve(), args.details_dir.resolve()
    if report_path == details_path or report_path in details_path.parents or details_path in report_path.parents:
        raise ValueError("keep summary and large details in separate non-nested directories")
    start = time.perf_counter()
    reference_sha = sha256_file(args.source_report)
    reference = json.loads(args.source_report.read_text(encoding="utf-8"))
    expected = reference["source"]
    expected_records = reference["counts"]["total_lines"]
    if (args.source.stat().st_size != expected["size"]
            or sha256_file(args.source) != expected["sha256"]):
        raise ValueError("full source identity mismatch")
    before = pipeline_identity(args.tokenizer_dir)
    if not before["tokenizer_files_sha256"]:
        raise ValueError("tokenizer JSON assets are missing")

    args.report_dir.mkdir(parents=True, exist_ok=False)
    args.details_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": expected["sha256"], "expected_records": expected_records,
                "processed_records": 0, "workers": args.workers, "chunk_size": args.chunk_size,
                "details_dir": str(args.details_dir), "filtering_executed": False, "training_started": False}
    manifest_path = args.report_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    overall = Statistics()
    by_stratum = {}
    longest = []
    flagged_examples = []
    digest = hashlib.sha256()
    records_path = args.details_dir / "records.jsonl"
    issues_path = args.details_dir / "issues.jsonl"
    try:
        with records_path.open("x", encoding="utf-8", newline="\n") as records_stream, \
                issues_path.open("x", encoding="utf-8", newline="\n") as issues_stream:
            def consume(rows):
                for row in rows:
                    records_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    overall.add(row)
                    if row["stratum"] not in by_stratum:
                        by_stratum[row["stratum"]] = Statistics()
                    by_stratum[row["stratum"]].add(row)
                    if row["issues"]:
                        issues_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if len(flagged_examples) < 20:
                            flagged_examples.append(row)
                    if row["input_tokens"] is not None:
                        item = (row["input_tokens"], -row["source_line"], row)
                        if len(longest) < 10:
                            heappush(longest, item)
                        elif item[:2] > longest[0][:2]:
                            heapreplace(longest, item)
                if overall.records % 10000 == 0 or overall.records == expected_records:
                    elapsed = time.perf_counter() - start
                    manifest.update({"processed_records": overall.records, "flagged_records": overall.flagged,
                                     "elapsed_seconds": round(elapsed, 2)})
                    write_json(manifest_path, manifest)
                    print(f"Encoded {overall.records:,}/{expected_records:,}; flags={overall.flagged}; "
                          f"elapsed={elapsed:.1f}s; rate={overall.records / elapsed:.1f} records/s", flush=True)

            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=init_worker, initargs=(str(args.tokenizer_dir.resolve()),)) as pool:
                pending = deque()
                for chunk in chunks(args.source, args.chunk_size, digest):
                    pending.append(pool.submit(inspect_chunk, chunk))
                    # Bound submitted work and emit results in original source order.
                    if len(pending) >= 2 * args.workers:
                        consume(pending.popleft().result())
                while pending:
                    consume(pending.popleft().result())

        if overall.records != expected_records or digest.hexdigest() != expected["sha256"]:
            raise ValueError("source count or bytes changed during full inspection")
        if (pipeline_identity(args.tokenizer_dir) != before
                or sha256_file(args.source_report) != reference_sha):
            raise ValueError("pipeline or reference identity changed during inspection")
        result = {
            "schema_version": 1,
            "status": "encoding_flags_found" if overall.flagged else "full_candidate_encoding_diagnostics_clear",
            "source": {"path": str(args.source), "size": expected["size"], "sha256": digest.hexdigest(),
                       "reference_report_sha256": reference_sha, "identity_checked_before_and_during_run": True},
            "overall": overall.summary(),
            "by_stratum": {name: statistics.summary() for name, statistics in sorted(by_stratum.items())},
            "longest_examples": [item[2] for item in sorted(longest, key=lambda item: (-item[0], -item[1]))],
            "flagged_examples": flagged_examples,
            "files": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                      for name, path in (("records", records_path), ("issues", issues_path))},
            "encoding": {"entry": "encode_sft_conversation", "truncation": False, "padding": False,
                         "retained_thinking_supervised": True, "ending": "EOS plus trailing newline",
                         "percentile_method": "numpy linear"},
            "pipeline": before,
            "execution": {"workers": args.workers, "chunk_size": args.chunk_size,
                          "elapsed_seconds": time.perf_counter() - start,
                          "device": "CPU; no model or CUDA execution"},
            "runtime": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "numpy")},
            "scope": {"full_candidate_source_checked": True, "final_training_data_accepted": False,
                      "quality_acceptance_claimed": False, "complete_tool_protocol_verified": False,
                      "boundary_detection_is_conservative_not_complete": True,
                      "training_runtime_gates_changed": False, "filtering_executed": False,
                      "identity_rewriting_applied": False, "training_recipe_frozen": False,
                      "training_started": False},
        }
        write_json(args.report_dir / "report.json", result)
        manifest.update({"status": "completed", "processed_records": overall.records,
                         "flagged_records": overall.flagged, "diagnostic_status": result["status"],
                         "elapsed_seconds": time.perf_counter() - start,
                         "report_sha256": sha256_file(args.report_dir / "report.json")})
        write_json(manifest_path, manifest)
        print(json.dumps({"report": str(args.report_dir / "report.json"), "overall": result["overall"]}, indent=2), flush=True)
        return 2 if overall.flagged else 0
    except BaseException as exc:
        manifest.update({"status": "failed", "processed_records": overall.records,
                         "elapsed_seconds": time.perf_counter() - start,
                         "error": {"type": type(exc).__name__, "message": str(exc)[:400]}})
        write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--details-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "assets/minimind_tokenizer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
