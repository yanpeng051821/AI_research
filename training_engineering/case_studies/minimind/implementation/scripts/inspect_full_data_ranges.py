"""Estimate full-source record density from verified bounded HTTP byte ranges."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from huggingface_hub import HfApi, hf_hub_url
import requests
from transformers import AutoTokenizer

from minimind_lab.data.pretrain import encode_pretrain_text
from minimind_lab.data.sft import encode_sft_conversation


REPO = "jingyaogong/minimind_dataset"
REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/minimind_tokenizer"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite range inspection")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    files = {entry.path: entry for entry in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION)}
    result = {"purpose": "bounded_sampling_not_full_audit_or_freeze", "source_repo": REPO,
              "source_revision": REVISION, "files": {},
              "limitations": ["Byte-stratified range samples, not uniform record samples or a complete line count.",
                             "Estimates can miss regional heterogeneity; no statistical confidence interval is claimed.",
                             "SFT structural encoding and length only; content/name holds, deduplication and split not applied.",
                             "No new training authorized by this report."]}
    for stage in ("pretrain", "sft"):
        name = f"{stage}_t2t.jsonl"
        size = files[name].size
        ranges = []
        lengths = []
        valid = 0
        reasons = {}
        representative_errors = {}
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            begin = int(size * fraction)
            end = min(size - 1, begin + 1024 * 1024 - 1)
            url = hf_hub_url(REPO, name, repo_type="dataset", revision=REVISION)
            with requests.get(url, headers={"Range": f"bytes={begin}-{end}"}, timeout=(15, 45), stream=True) as response:
                response.raise_for_status()
                expected = f"bytes {begin}-{end}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"server did not return the requested range for {name}")
                body = response.raw.read(end - begin + 2)
            if len(body) != end - begin + 1:
                raise ValueError("range length mismatch")
            first, last = body.find(b"\n"), body.rfind(b"\n")
            if first < 0 or last <= first:
                raise ValueError("range does not contain complete records")
            complete = body[first + 1:last + 1]
            rows = complete.splitlines()
            range_valid = 0
            range_lengths = []
            range_rejections = {}
            for row in rows:
                try:
                    record = json.loads(row)
                    if stage == "pretrain":
                        if not isinstance(record.get("text"), str) or not record["text"].strip():
                            raise ValueError("invalid_or_empty_text")
                        ids = encode_pretrain_text(record["text"], tokenizer, 768)
                    else:
                        ids = encode_sft_conversation(record["conversations"], tokenizer)["input_ids"]
                        if len(ids) > 768:
                            raise ValueError("overlength_768")
                    valid += 1
                    range_valid += 1
                    lengths.append(len(ids))
                    range_lengths.append(len(ids))
                except Exception as error:
                    reason = ("overlength_768" if str(error) == "overlength_768"
                              else type(error).__name__ + ":" + str(error).split(":", 1)[0][:100])
                    reasons[reason] = reasons.get(reason, 0) + 1
                    range_rejections[reason] = range_rejections.get(reason, 0) + 1
                    representative_errors.setdefault(reason, str(error)[:400])
            ranges.append({"start": begin, "end": end, "sha256": hashlib.sha256(body).hexdigest(),
                           "content_range": expected, "complete_bytes": len(complete), "records": len(rows),
                           "structural_length_accepted": range_valid,
                           "rejections": range_rejections,
                           "mean_accepted_length": statistics.mean(range_lengths) if range_lengths else None})
            print(stage, fraction, len(rows), range_valid, flush=True)
        sampled_records = sum(item["records"] for item in ranges)
        density = sampled_records / sum(item["complete_bytes"] for item in ranges)
        # Extrapolate each byte stratum independently to expose heterogeneity.
        regional_counts = [size * item["records"] / item["complete_bytes"] for item in ranges]
        accepted_regional = [size * item["structural_length_accepted"] / item["complete_bytes"] for item in ranges]
        result["files"][stage] = {
            "filename": name, "source_bytes": size, "ranges": ranges, "sampled_records": sampled_records,
            "sample_accepted_records": valid, "sample_rejections": reasons,
            "representative_error_messages": representative_errors,
            "estimated_raw_records": size * density,
            "regional_raw_estimate_min_max": [min(regional_counts), max(regional_counts)],
            "estimated_structural_length_accepted_records": size * density * valid / sampled_records,
            "regional_accepted_estimate_min_max": [min(accepted_regional), max(accepted_regional)],
            "mean_accepted_encoded_length": statistics.mean(lengths) if lengths else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({stage: {key: value for key, value in data.items() if key != "ranges"}
                      for stage, data in result["files"].items()}), flush=True)


if __name__ == "__main__":
    main()
