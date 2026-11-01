"""Inspect SFT probe inputs without loading model weights or generating answers."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from minimind_lab.data.sft import prepare_sft_messages
from minimind_lab.data.sft_protocol import check_sft_protocol
from prepare_sft_recipe import normalize


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def render_probe(case, config, tokenizer):
    messages = [{"role": "system", "content": config["system_prompt"]}, *deepcopy(case["messages"])]
    if case["tools"]:
        messages[0]["tools"] = deepcopy(case["tools"])
    # The completed copy checks prefix protocol only; the placeholder is never rendered.
    completed = [*deepcopy(messages), {"role": "assistant", "content": "protocol audit placeholder"}]
    prepared, tools = prepare_sft_messages(completed)
    issues, notices = check_sft_protocol(prepared, tools)
    if issues:
        raise ValueError(f"invalid probe protocol: {case['id']}: {issues}")
    prefix = prepared[:-1]
    text = tokenizer.apply_chat_template(prefix, tools=tools, tokenize=False,
                                         add_generation_prompt=True,
                                         open_thinking=config["generation"]["open_thinking"])
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return text, ids, notices


def inspect(cases_path, config_path, output):
    if output.exists():
        raise FileExistsError("do not overwrite a draft inspection report")
    inputs_before = {str(p.resolve()): sha(p) for p in (cases_path, config_path, Path(__file__))}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = [json.loads(raw) for raw in cases_path.read_text(encoding="utf-8").splitlines()]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    expected = {"everyday": 8, "instruction": 8, "grounded": 8, "context": 8, "tool": 4}
    if dict(Counter(case["category"] for case in cases)) != expected:
        raise ValueError("unexpected probe composition")
    contract_path = (config_path.parent / config["data_contract"]).resolve()
    if sha(contract_path) != config["data_contract_sha256"]:
        raise ValueError("data contract changed")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract["status"] != "frozen":
        raise ValueError("expected frozen data")
    tokenizer_dir = ROOT / "assets/minimind_tokenizer"
    tokenizer_hashes = {p.relative_to(tokenizer_dir).as_posix(): sha(p)
                        for p in tokenizer_dir.rglob("*") if p.is_file()}
    if tokenizer_hashes != contract["pipeline_identity"]["tokenizer_files_sha256"]:
        raise ValueError("tokenizer differs from data recipe")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    query_cases = defaultdict(set)
    details = []
    for case in cases:
        if (not case["checks"] or not all(isinstance(c, str) and c.strip() for c in case["checks"])
                or case["tier"] != ("supplementary" if case["category"] == "tool" else "core")):
            raise ValueError(f"invalid rubric or tier: {case['id']}")
        text, ids, notices = render_probe(case, config, tokenizer)
        total = len(ids) + config["generation"]["max_new_tokens"]
        if total > config["generation"]["context_budget"]:
            raise ValueError(f"prompt plus generation exceeds context budget: {case['id']}")
        details.append({"id": case["id"], "input_tokens": len(ids), "reserved_total_tokens": total,
                        "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        "rendered_prompt": text, "protocol_notices": notices})
        for message in case["messages"]:
            if message["role"] == "user":
                query_cases[normalize(message["content"])].add(case["id"])
    overlaps, scanned = {}, {}
    for split in ("train", "validation"):
        desc = contract["artifacts"][split]
        hits = []
        count = size = 0
        digest = hashlib.sha256()
        with Path(desc["path"]).open("rb") as stream:
            for count, raw in enumerate(stream, 1):
                digest.update(raw)
                size += len(raw)
                for index, message in enumerate(json.loads(raw)["conversations"]):
                    if message["role"] == "user":
                        match = query_cases.get(normalize(message["content"]))
                        if match:
                            hits.append({"split_line": count, "message_index": index, "case_ids": sorted(match)})
        if (digest.hexdigest() != desc["sha256"] or count != desc["records"] or size != desc["bytes"]):
            raise ValueError(f"frozen split changed: {split}")
        overlaps[split] = {"matched_user_messages": len(hits), "examples": hits[:20]}
        scanned[split] = count
    if any(sha(Path(path)) != value for path, value in inputs_before.items()):
        raise ValueError("draft or inspector changed during inspection")
    report = {"status": "draft_inputs_inspected_not_approved_or_run", "draft_inputs_sha256": inputs_before,
              "data_contract_sha256": config["data_contract_sha256"], "cases": len(cases),
              "categories": expected, "core_cases": 32, "supplementary_cases": 4,
              "max_prompt_tokens": max(r["input_tokens"] for r in details),
              "max_reserved_total_tokens": max(r["reserved_total_tokens"] for r in details),
              "split_records_scanned": scanned, "normalized_user_text_overlap": overlaps,
              "semantic_near_duplicate_or_pretraining_contamination_clearance": False,
              "model_weights_loaded": False, "generation_performed": False,
              "rubric_semantic_quality_approved": False, "evaluation_recipe_frozen": False,
              "prompt_details": details}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "prompt_details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.cases, args.config, args.output)
