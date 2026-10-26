"""Package a disclosed research release; optionally upload and verify a pinned HF snapshot."""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from safetensors.torch import load_model

from audit_pretrain_release import scan_texts
from export_pretrain_bundle import verify_in_fresh_process
from load_exported_pretrain import sha256_file, state_digest, write_json
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from transformers import AutoTokenizer
import torch


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def credential(path):
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        value = (value if separator else key).strip().strip('"').strip("'")
        if re.fullmatch(r"hf_[A-Za-z0-9]+", value):
            values.append(value)
    if len(set(values)) != 1:
        raise ValueError("credential file must contain one unambiguous HF credential")
    return values[0]


def describe(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def check_files(bundle, manifest):
    names = set()
    for item in manifest["files"]:
        path = (bundle / item["path"]).resolve()
        if not path.is_relative_to(bundle.resolve()) or item["path"] in names:
            raise ValueError("invalid manifest path")
        names.add(item["path"])
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"file identity mismatch: {item['path']}")


def prepare(source, experiment, output, reports, repo_id, code_revision):
    if output.exists() or reports.exists():
        raise FileExistsError("choose a new public payload/report version")
    original = read_json(source / "export_manifest.json")
    verification = read_json(experiment / "results/export_fp32_v2/verification.json")
    if verification["export_manifest_sha256"] != sha256_file(source / "export_manifest.json"):
        raise ValueError("source is not the verified v2 export")
    check_files(source, original)
    output.mkdir(parents=True)
    for item in original["files"]:
        if item["path"] == "README.md":
            continue
        target = output / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / item["path"], target)
    card = (experiment / "release/MODEL_CARD_DRAFT.md").read_text(encoding="utf-8")
    card = card.replace("研究 checkpoint 发布候选。", "公开研究 checkpoint。")
    card = card.replace("500d31868bf04f1572281c730d5ddd2a413708a7", code_revision)
    card = card.replace("新导出工具和发布资料尚待提交后重新固定链接。",
                        "本次导出与发布工具也包含在所链接提交中。")
    card = card.replace("实际仓库地址与已验证 revision 由发布记录给出，",
                        f"仓库为 https://huggingface.co/{repo_id}；发布 revision 见 GitHub 发布记录，")
    card += ("\n## 9. 发布文件\n\n"
             "recipes/ 保存实际训练配置、核心合同及无本机路径的数据配方摘要；"
             "evidence/ 保存逐点验证 NLL、成本、曲线、原始人工抽查和本地导出核对。"
             "完整源码、冻结配方与数据准备入口见上述固定 GitHub 链接。"
             "上游许可证及来源说明位于 licenses/ 与 THIRD_PARTY_NOTICES.md，"
             "它们不替代尚未明确的整体或权重许可。\n")
    write_text(output / "README.md", card)
    write_text(output / "THIRD_PARTY_NOTICES.md",
               "# Sources and license status\n\n"
               "Architecture reference and tokenizer: jingyaogong/minimind at "
               "f659b55761b754d306bd140573493a6543cafd7f, whose root license is Apache-2.0.\n"
               "The runtime was implemented in a guided learning project; reference attribution "
               "does not mean upstream weights were loaded. Source file hashes are in the manifest.\n"
               "Tokenizer JSON content matches the reference; local copies use CRLF instead of LF.\n"
               "Data: jingyaogong/minimind_dataset at 312afb4f76391145c6902f765bb51691c09a12f5, "
               "pretrain_t2t_mini.jsonl. The dataset card lists Apache-2.0 and CC-BY-NC-2.0. "
               "Per-file provenance and the resulting weight redistribution terms are not fully established.\n"
               "No blanket code/weight license or cleared commercial rights are claimed. "
               "Public download is not a grant of otherwise unspecified rights. "
               "The included upstream license is not declared to be this model's weight license.\n")
    license_url = ("https://api.github.com/repos/jingyaogong/minimind/contents/LICENSE"
                   "?ref=f659b55761b754d306bd140573493a6543cafd7f")
    request = urllib.request.Request(license_url, headers={"User-Agent": "minimind-research-release"})
    with urllib.request.urlopen(request, timeout=30) as response:
        license_bytes = base64.b64decode(json.load(response)["content"])
    review = read_json(experiment / "release/release_review_v2.json")
    import hashlib
    if hashlib.sha256(license_bytes).hexdigest() != review["upstream_code"]["license_file_sha256"]:
        raise ValueError("license file differs from reviewed version")
    (output / "licenses").mkdir()
    (output / "licenses/MiniMind-Apache-2.0.txt").write_bytes(license_bytes)
    recipes = output / "recipes"
    recipes.mkdir()
    for name in ("training_config.json", "contract.json"):
        shutil.copyfile(experiment / "training_recipe" / name, recipes / name)
    data = read_json(experiment / "data_recipe/manifest.json")
    encoding = read_json(experiment / "data_recipe/encoding_audit_768.json")
    write_json(recipes / "data_recipe.json", {
        "source_repo": data["source_repo"], "source_revision": data["source_revision"],
        "source_file": data["source_file"], "source_sha256": data["inputs"]["source"]["sha256"],
        "split_policy": data["split_policy"], "counts": data["counts"], "files": data["files"],
        "encoding_policy": encoding["encoding_policy"], "max_length": encoding["max_length"],
        "special_token_ids": encoding["special_token_ids"],
        "tokenizer_files_sha256": encoding["tokenizer_files_sha256"],
        "statistics": encoding["statistics"],
        "source_split_manifest_sha256": sha256_file(experiment / "data_recipe/manifest.json"),
        "source_encoding_report_sha256": sha256_file(experiment / "data_recipe/encoding_audit_768.json"),
        "scope": "Public summary; original local-path reports remain unchanged. No raw data included.",
    })
    evidence = output / "evidence"
    evidence.mkdir()
    for local_name, public_name in (("runtime_cost_v1.json", "runtime_cost.json"),
                                    ("manual_generation_v1.json", "manual_generation.json"),
                                    ("learning_curves_v1/learning_curves.png", "learning_curves.png")):
        shutil.copyfile(experiment / "results" / local_name, evidence / public_name)
    metrics = [json.loads(line) for line in
               (experiment / "results/final_run_v1/run/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    write_json(evidence / "validation_history.json", [row for row in metrics
               if row["event"] in {"initial_validation", "periodic_validation", "final_validation"}])
    write_json(evidence / "export_verification.json", {key: value for key, value in verification.items()
                                                       if key != "bundle_location"})
    manifest = {**original, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "public_release": {"repo_id": repo_id, "code_revision": code_revision,
                    "packaging_script_sha256": sha256_file(Path(__file__)),
                    "source_export_manifest_sha256": sha256_file(source / "export_manifest.json"),
                    "human_authorization": "User requested publication in this conversation on 2026-10-27.",
                    "license_status": "Declared uncertainty retained; no blanket license selected."}}
    manifest["files"] = [describe(p, output) for p in sorted(output.rglob("*")) if p.is_file()]
    text_paths = [(p.relative_to(output).as_posix(), p) for p in output.rglob("*")
                  if p.is_file() and p.suffix in {".py", ".json", ".txt", ".md"}]
    scan = scan_texts(text_paths)
    if scan["findings"]:
        raise ValueError(f"public payload requires review: {scan['findings']}")
    write_json(output / "export_manifest.json", manifest)
    check_files(output, manifest)
    reports.mkdir(parents=True)
    shutil.copyfile(output / "export_manifest.json", reports / "export_manifest.json")
    write_json(reports / "preparation.json", {
        "status": "prepared_for_user_authorized_research_publication",
        "repo_id": repo_id, "code_revision": code_revision,
        "source_export_manifest_sha256": manifest["public_release"]["source_export_manifest_sha256"],
        "public_export_manifest_sha256": sha256_file(output / "export_manifest.json"),
        "scan": scan, "model_weights_unchanged":
            sha256_file(output / "model.safetensors") == sha256_file(source / "model.safetensors"),
        "license_scope_fully_established": False, "hf_upload": "not_yet_performed",
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    code_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source, output = args.source_bundle.resolve(), args.output_dir.resolve()
    manifest = prepare(source, args.experiment_dir, output, args.report_dir, args.repo_id, code_revision)
    if not args.publish:
        print(json.dumps({"status": "prepared_not_uploaded", "repo_id": args.repo_id}))
        return
    if args.credential_file is None:
        parser.error("publication requires the explicitly authorized credential file")
    api = HfApi(token=credential(args.credential_file))
    account = api.whoami()["name"]
    if args.repo_id.split("/")[0] != account:
        raise ValueError("target namespace is not the authenticated user's namespace")
    try:
        api.model_info(args.repo_id)
    except RepositoryNotFoundError:
        pass
    else:
        raise FileExistsError("target repo exists; do not overwrite another release")
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=False)
    print("Public repository created; uploading whitelisted payload files.", flush=True)
    allowed = [item["path"] for item in manifest["files"]] + ["export_manifest.json"]
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=output, repo_type="model",
                               allow_patterns=allowed, commit_message="Publish MiniMind mini pretraining research checkpoint")
    write_json(args.report_dir / "upload.json", {"repo_id": args.repo_id, "revision": commit.oid,
               "url": f"https://huggingface.co/{args.repo_id}", "commit_url": commit.commit_url,
               "export_manifest_sha256": sha256_file(output / "export_manifest.json"),
               "status": "uploaded_download_verification_pending"})
    print("Upload complete; downloading the exact revision with a fresh local cache.", flush=True)
    with tempfile.TemporaryDirectory(prefix="minimind-hf-verify-") as temporary:
        temporary = Path(temporary)
        download = Path(snapshot_download(args.repo_id, repo_type="model", revision=commit.oid,
                       token=api.token, cache_dir=temporary / "cache", local_dir=temporary / "snapshot",
                       allow_patterns=allowed, max_workers=2, force_download=True))
        if sha256_file(download / "export_manifest.json") != sha256_file(output / "export_manifest.json"):
            raise ValueError("downloaded manifest differs from the published payload")
        check_files(download, manifest)
        config = read_json(source / "config.json")
        model = MinimindForCausalLM(ModelConfig(**config["model"]))
        load_model(model, source / "model.safetensors", strict=True, device="cpu")
        tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
        if state_digest(model) != manifest["model_state_sha256"]:
            raise ValueError("reference export state differs")
        checked = verify_in_fresh_process(download, model, tokenizer)
    write_json(args.report_dir / "download_verification.json", {
        **checked, "repo_id": args.repo_id, "revision": commit.oid,
        "downloaded_files_verified": len(allowed), "fresh_download_cache": True,
        "export_manifest_sha256": sha256_file(output / "export_manifest.json"),
        "code_revision": code_revision, "license_scope_fully_established": False,
        "independent_resumable_checkpoint_backup": "not_verified",
    })
    print(json.dumps({"status": "published_and_revision_download_verified", "repo_id": args.repo_id,
                      "revision": commit.oid, "report": str(args.report_dir)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Network exceptions can include credential-bearing URLs; report only safe metadata.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "http_status": getattr(getattr(exc, "response", None), "status_code", None)}),
              file=sys.stderr)
        raise SystemExit(1)
