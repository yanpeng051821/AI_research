"""Check a native export and its public-source declarations without uploading anything."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "f659b55761b754d306bd140573493a6543cafd7f"
DATA_REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
SECRET_PATTERNS = [r"hf_[A-Za-z0-9]{25,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                   r"github_pat_[A-Za-z0-9_]{20,}", r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}",
                   r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "minimind-publication-review"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def github_file(name):
    url = (f"https://api.github.com/repos/jingyaogong/minimind/contents/{name}"
           f"?ref={UPSTREAM_REVISION}")
    metadata = json.loads(fetch(url))
    return base64.b64decode(metadata["content"]), url


def scan_texts(paths):
    findings = []
    for label, path in paths:
        text = path.read_text(encoding="utf-8")
        if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
            findings.append({"file": label, "kind": "credential_pattern"})
        if re.search(r"[A-Za-z]:[\\/](?:Users|pythonlearning)[\\/]|/Users/[^\s/]+/", text):
            findings.append({"file": label, "kind": "local_absolute_path"})
    return {"text_files_checked": len(paths), "findings": findings,
            "scope": "Candidate bundle text plus model-card draft only; not full Git history, "
                     "raw training data, PII analysis, or memorization testing. "
                     "No matching secret values are included in this report."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new review report version")
    bundle = args.bundle_dir.resolve()
    manifest_path = bundle / "export_manifest.json"
    manifest = read_json(manifest_path)
    verification = read_json(args.experiment_dir / "results/export_fp32_v2/verification.json")
    dataset_identity = read_json(args.experiment_dir / "data_recipe/manifest.json")
    tokenizer_identity = read_json(bundle / "tokenizer/manifest.json")
    if (dataset_identity["source_repo"] != "jingyaogong/minimind_dataset"
            or dataset_identity["source_revision"] != DATA_REVISION
            or tokenizer_identity["source_repository"] != "jingyaogong/minimind"
            or tokenizer_identity["source_revision"] != UPSTREAM_REVISION):
        raise ValueError("release sources differ from the revisions reviewed by this script")
    if manifest["format"] != "minimind-lab-native-fp32-v1":
        raise ValueError("unexpected bundle format")
    if verification["status"] != "passed" or verification["export_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("bundle does not match the verified v2 export")
    text_paths = [("MODEL_CARD_DRAFT.md", args.card)]
    seen = set()
    for item in manifest["files"]:
        path = (bundle / item["path"]).resolve()
        if not path.is_relative_to(bundle) or item["path"] in seen:
            raise ValueError("invalid bundle path")
        seen.add(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"bundle file differs: {item['path']}")
        if path.suffix in {".py", ".json", ".md", ".txt"}:
            text_paths.append((item["path"], path))
    text_paths.append(("export_manifest.json", manifest_path))
    scan = scan_texts(text_paths)
    license_bytes, license_url = github_file("LICENSE")
    if "Apache License" not in license_bytes.decode("utf-8"):
        raise ValueError("upstream code license declaration changed")
    assets = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        upstream, url = github_file("model/" + name)
        current = (bundle / "tokenizer" / name).read_bytes()
        if json.loads(current) != json.loads(upstream):
            raise ValueError(f"tokenizer JSON content differs from declared upstream: {name}")
        assets.append({"file": name, "upstream_api_url": url,
                       "upstream_sha256": sha256_bytes(upstream),
                       "export_sha256": sha256_bytes(current),
                       "same_bytes": current == upstream,
                       "json_content_equal": True,
                       "newline_normalized_equal": current.replace(b"\r\n", b"\n") == upstream.replace(b"\r\n", b"\n")})
    data_url = ("https://huggingface.co/datasets/jingyaogong/minimind_dataset/raw/"
                f"{DATA_REVISION}/README.md")
    data_readme = fetch(data_url)
    data_text = data_readme.decode("utf-8")
    if "apache-2.0" not in data_text or "cc-by-nc-2.0" not in data_text:
        raise ValueError("data license declarations require a new manual review")
    repo_url = "https://api.github.com/repos/yanpeng051821/AI_research"
    repo = json.loads(fetch(repo_url))
    license_files = [p.name for p in ROOT.parents[3].glob("LICENSE*")]
    # Gates are explicit; source metadata is not a blanket grant for our model weights.
    unresolved = [
        "Clarify the applicable terms for pretrain_t2t_mini.jsonl and resulting weight redistribution; "
        "dataset-level mixed labels are not a per-file license mapping.",
        "User must choose explicit terms for original code and model weights, after source review; "
        "no new project-wide license was selected automatically.",
        "Add applicable upstream license/attribution materials to a separately versioned release payload.",
        "Commit/push the new export tools and evidence, then pin actual accessible GitHub links.",
        "Assemble and recheck the public payload; do not upload raw local-path logs by default.",
        "Upload and verify a specific HF revision only after release decisions are resolved.",
    ]
    report = {
        "schema_version": 1, "status": "draft_not_approved_for_upload",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_script_sha256": sha256_file(Path(__file__)),
        "model_card_draft_sha256": sha256_file(args.card),
        "export_manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
        "bundle_files_verified": len(seen), "local_export_validation": "passed",
        "credential_and_path_scan": scan,
        "upstream_code": {"repo": "jingyaogong/minimind", "revision": UPSTREAM_REVISION,
                          "license_declaration": "Apache-2.0", "license_api_url": license_url,
                          "license_file_sha256": sha256_bytes(license_bytes),
                          "scope": "Code repository declaration; not an automatic model-weight license."},
        "tokenizer": {"repo": "jingyaogong/minimind", "revision": UPSTREAM_REVISION,
                      "files": assets, "separate_asset_license_found": False,
                      "scope": "Repository tree has no separately named tokenizer license; "
                               "not a legal conclusion about grant scope."},
        "dataset": {"repo": "jingyaogong/minimind_dataset", "revision": DATA_REVISION,
                    "file": "pretrain_t2t_mini.jsonl", "readme_url": data_url,
                    "readme_sha256": sha256_bytes(data_readme),
                    "declared_licenses": ["apache-2.0", "cc-by-nc-2.0"],
                    "per_file_weight_redistribution_rights": "not_established_by_this_review"},
        "own_repository": {"url": repo["html_url"], "public": not repo["private"],
                           "github_detected_license": repo.get("license"),
                           "local_root_license_files": license_files},
        "unresolved_before_upload": unresolved,
        "hf_upload": "not_performed", "source_bundle_modified": False,
        "limitations": ["This records source declarations and release risks; it is not legal advice.",
                        "A private or noncommercial label does not by itself supply missing permissions.",
                        "No HF model identifier, license choice, or public-release permission was invented."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "bundle_files_verified": len(seen),
                      "scan": scan, "unresolved_items": len(unresolved),
                      "report": str(args.output)}))


if __name__ == "__main__":
    main()
