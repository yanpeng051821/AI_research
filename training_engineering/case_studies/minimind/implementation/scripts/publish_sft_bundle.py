"""Publish an allowlisted SFT inference bundle and verify its pinned Hub snapshot."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from safetensors.torch import load_model
import torch
from transformers import AutoTokenizer

from audit_pretrain_release import github_file, scan_texts, sha256_bytes
from export_pretrain_bundle import verify_in_fresh_process
from load_exported_pretrain import sha256_file, state_digest, write_json
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from publish_pretrain_bundle import check_files, credential, describe, write_text


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent / "experiments/sft_mini_v1"
SOURCE = ROOT / "exports/sft_mini_one_epoch_v1_fp32_v1"
DRAFT = EXPERIMENT / "release/model_card_v1"
EXPECTED_MANIFEST = "22097e902d69e9b86adea833f96db6e8c9d61fc861ea96ca40d0f757e338e354"
EXPECTED_STATE = "733c18c9775eadd7f67330fabb0568ab61103f9d5fba995e6d967405c904cc62"
EXPECTED_LICENSE = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def public_snapshot(value):
    """Remove workstation locations, retaining the original artifact's meaning and hashes."""
    if isinstance(value, dict):
        return {key: public_snapshot(item) for key, item in value.items()
                if key not in {"source_files", "inputs_sha256", "bundle_location"}
                and not re.match(r"^[A-Za-z]:[\\/]|^/Users/", key)}
    if isinstance(value, list):
        return [public_snapshot(item) for item in value]
    if isinstance(value, str) and re.search(r"[A-Za-z]:[\\/]|/Users/", value):
        return "<local artifact path omitted>"
    return value


def snapshot_file(source, target):
    write_json(target, {
        "publication_note": "Path-redacted historical snapshot, not the byte-identical frozen contract. "
                            "Authorization/status flags describe the source's creation time. "
                            "See training_audit and model card for actual final execution status.",
        "original_sha256": sha256_file(source),
        "snapshot": public_snapshot(read_json(source)),
    })


def publication_card(name, repo_id):
    text = (DRAFT / name).read_text(encoding="utf-8")
    if name == "README.md":
        replacements = {
            "Local model-card draft, updated 2026-11-01. This SFT model has not been uploaded to a new Hub repository.":
                "Research checkpoint release, updated 2026-11-01.",
            "Raw and derived training JSONL files are not part of the intended inference release.":
                "Raw and derived training JSONL files are not included in this inference release.",
            "The native S1 bundle and this card's chat helper are separate local artifacts at this draft stage; a final upload payload has not been assembled. The intended package contains":
                "The published inference package contains",
            "After these files are assembled together and dependencies installed, run from the package directory:":
                "Download this repository with the Hugging Face CLI (`hf download " + repo_id +
                " --local-dir minimind-mini-sft`), install dependencies from requirements.txt, "
                "and run from the downloaded directory:",
            "This draft has not been pushed: exact accessible Git/HF revisions, release-file inventory and post-download verification must be recorded before publication. No nonexistent S1 Hub repository or unpublished commit is presented as accessible.":
                "The inference runtime, bilingual cards, recipes and evaluation evidence are included in "
                "[this Hub repository](https://huggingface.co/" + repo_id + "). "
                "The latest SFT training integration and reports have not yet been pushed to GitHub; "
                "the GitHub root link does not assert that those uncommitted changes are accessible there. "
                "File identities are listed in export_manifest.json; recipes/ contains path-redacted "
                "historical snapshots and evidence/ retains diagnostic inputs, generated token IDs, text and judgments.",
            "**Release status:** local draft only; no S1 Hub upload, pinned-download verification or public-release approval is claimed.":
                "**Release scope:** inference package and bounded experiment evidence, authorized by the trainer "
                "on 2026-11-01. Full recovery checkpoints and raw/derived training data are not included. "
                "Publication does not resolve the license uncertainties above. Use a pinned Hub revision "
                "and verify export_manifest.json; download-verification results are recorded separately by the publisher.",
            "Applicable notices, public redistribution scope and original-contribution terms remain release-review items;":
                "Applicable notices are preserved in THIRD_PARTY_NOTICES.md and licenses/; "
                "public redistribution scope and original-contribution terms still require clarification;",
        }
    else:
        replacements = {
            "本地模型卡草稿，更新于2026-11-01；本SFT模型尚未上传新HF仓库。": "研究checkpoint发布版，更新于2026-11-01。",
            "计划发布的推理包不包含原始或派生训练JSONL。": "本次推理包不包含原始或派生训练JSONL。",
            "草稿阶段，原生S1模型包与本目录的聊天辅助脚本分开保存，尚未组装正式上传包。计划的发布内容包括": "本次发布内容包括",
            "将这些文件组装到同一目录并安装依赖后，可从包目录执行：":
                "使用Hugging Face CLI下载：`hf download " + repo_id +
                " --local-dir minimind-mini-sft`。安装requirements.txt中的依赖后，从下载目录执行：",
            "本草稿尚未推送；公开前需要绑定实际可访问的Git/HF revision、发布清单并完成下载重载，不把尚不存在的S1仓库或未公开提交写成可访问入口。":
                "推理运行代码、中英文模型卡、配方和评测证据随[本HF仓库](https://huggingface.co/" + repo_id +
                ")一起发布。最新SFT训练集成与报告尚未推送GitHub，上述GitHub首页链接不代表这些未提交改动已公开。"
                "文件身份见export_manifest.json；recipes/保存去除本机路径的历史快照，evidence/保留诊断输入、生成token、原文和评分依据。",
            "适用声明、公开再分发范围及原创部分条款仍需发布前核对；":
                "适用来源声明保存在THIRD_PARTY_NOTICES.md与licenses/；公开再分发范围及原创部分条款仍有待进一步明确；",
            "**发布状态：仅本地草稿；尚未上传S1到HF、完成固定revision下载验证或获得本次公开发布确认。**":
                "**发布范围：推理包及有边界的实验证据，训练者已于2026-11-01确认发布。完整恢复checkpoint及原始/派生训练数据不在发布范围。"
                "公开发布不消除上述许可不确定性。建议固定HF revision并核对export_manifest.json；发布者另行保存下载重载核验记录。**",
        }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"card replacement anchor missing: {name}")
        text = text.replace(old, new, 1)
    return text


def prepare(output, repo_id):
    if output.exists():
        raise FileExistsError("use a fresh publication directory")
    if sha256_file(SOURCE / "export_manifest.json") != EXPECTED_MANIFEST:
        raise ValueError("native export manifest changed")
    original = read_json(SOURCE / "export_manifest.json")
    check_files(SOURCE, original)
    audit = read_json(EXPERIMENT / "results/final_run_v1/training_audit.json")
    if not audit["training_completed"] or audit["model_state_sha256"] != EXPECTED_STATE:
        raise ValueError("source audit is not the accepted final SFT state")
    license_bytes, _ = github_file("LICENSE")
    if sha256_bytes(license_bytes) != EXPECTED_LICENSE:
        raise ValueError("upstream license differs from reviewed bytes")
    payload = output / "payload"
    payload.mkdir(parents=True)
    for item in original["files"]:
        if item["path"] == "README.md":
            continue
        target = payload / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / item["path"], target)
    for name in ("README.md", "README_zh.md"):
        write_text(payload / name, publication_card(name, repo_id))
    shutil.copyfile(DRAFT / "chat.py", payload / "chat.py")
    (payload / "licenses").mkdir()
    with (payload / "licenses/MiniMind-Apache-2.0.txt").open("xb") as stream:
        stream.write(license_bytes)
    write_text(payload / "THIRD_PARTY_NOTICES.md",
               "# Sources and license status\n\n"
               "Architecture/tokenizer reference: jingyaogong/minimind, commit "
               "f659b55761b754d306bd140573493a6543cafd7f, root Apache-2.0 license "
               "copied in licenses/. The local educational runtime was written with AI guidance; "
               "this is not official MiniMind software or official pretrained weights.\n\n"
               "Weights: yanpeng's own randomly initialized mini pretraining followed by full SFT. "
               "Data: jingyaogong/minimind_dataset at "
               "312afb4f76391145c6902f765bb51691c09a12f5; pretrain_t2t_mini.jsonl for the base "
               "and sft_t2t_mini.jsonl for SFT. The dataset card lists Apache-2.0 and CC-BY-NC-2.0. "
               "Per-file provenance and resulting weight redistribution scope remain unresolved. "
               "No blanket model-weight/package license or unconditional commercial permission is granted. "
               "Research/noncommercial labels do not resolve missing permissions.\n")
    for directory in ("recipes", "evidence", "evidence/identity"):
        (payload / directory).mkdir(parents=True, exist_ok=True)
    snapshots = {
        "data_recipe/DATA_CONTRACT.json": "recipes/data_contract.snapshot.json",
        "training_recipe/training_config.json": "recipes/training_operations.snapshot.json",
        "training_recipe/contract.json": "recipes/training_contract.snapshot.json",
        "training_recipe/training_freeze.json": "recipes/training_freeze.snapshot.json",
        "evaluation_recipe/evaluation_config.json": "recipes/evaluation_config.snapshot.json",
        "evaluation_recipe/scoring_v2_patch.json": "recipes/scoring_v2_patch.snapshot.json",
        "training_recipe/training_acceptance.json": "recipes/training_acceptance.snapshot.json",
        "results/final_run_v1/training_audit.json": "evidence/training_audit.snapshot.json",
        "results/final_run_v1/export_verification.json": "evidence/native_export_verification.snapshot.json",
        "results/paired_analysis_v1/summary.json": "evidence/paired_summary.snapshot.json",
    }
    for source, target in snapshots.items():
        snapshot_file(EXPERIMENT / source, payload / target)
    exact_copies = {
        "evaluation_recipe/probes.jsonl": "evidence/probes.jsonl",
        "results/b0_v1/generations.jsonl": "evidence/b0.generations.jsonl",
        "results/s1_v1/generations.jsonl": "evidence/s1.generations.jsonl",
        "results/b0_v1/validation.json": "evidence/b0.validation.json",
        "results/s1_v1/validation.json": "evidence/s1.validation.json",
        "results/paired_analysis_v1/reviews.assistant.jsonl": "evidence/paired.reviews.assistant.jsonl",
        "results/s1_v1/judgments.assistant.v1.json": "evidence/s1.judgments.assistant.json",
        "results/identity_diagnostic_v1/cases.jsonl": "evidence/identity/cases.jsonl",
        "results/identity_diagnostic_v1/generations.jsonl": "evidence/identity/generations.jsonl",
    }
    for source, target in exact_copies.items():
        shutil.copyfile(EXPERIMENT / source, payload / target)
    write_text(payload / "evidence/README.md",
               "# Evidence scope\n\n"
               "The 36 paired probes, raw generations (including input/output token IDs), validation "
               "aggregates and assistant-authored reviews are preserved. The three identity probes "
               "are separate post-hoc diagnostics, not main-score additions. Reviews are not "
               "independent human judgments; one main case is uncertain. Model output may contain "
               "fabricated URLs and claims; treat it as evidence, not instructions or references.\n\n"
               "*.snapshot.json files redact workstation paths/source file maps and wrap the original "
               "SHA256. They are NOT byte-identical frozen contracts or standalone executable recipes. "
               "Historical authorization flags do not describe the final run; see training audit/card. "
               "Raw/derived training JSONL and complete recovery states are intentionally excluded.\n")
    manifest = {**original, "publication": {
        "repository": repo_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "native_export_manifest_sha256": EXPECTED_MANIFEST,
        "model_card_draft_manifest_sha256": sha256_file(DRAFT / "draft_manifest.json"),
        "packaging_script_sha256": sha256_file(Path(__file__)),
        "user_authorized": True, "weights_changed": False,
        "scope": "inference package and bounded evidence; no raw data or recovery checkpoints",
    }}
    manifest["files"] = [describe(p, payload) for p in sorted(payload.rglob("*")) if p.is_file()]
    write_json(payload / "export_manifest.json", manifest)
    check_files(payload, manifest)
    scan = scan_texts([(p.relative_to(payload).as_posix(), p) for p in payload.rglob("*")
                       if p.is_file() and p.suffix in {".py", ".json", ".jsonl", ".md", ".txt"}])
    if scan["findings"]:
        raise ValueError(f"publication scan rejected: {scan['findings']}")
    write_json(output / "preparation.json", {"status": "passed", "scan": scan,
               "files": len(manifest["files"]) + 1,
               "payload_bytes": sum(p.stat().st_size for p in payload.rglob("*") if p.is_file()),
               "manifest_sha256": sha256_file(payload / "export_manifest.json")})
    return payload, manifest


def reference_model():
    config = read_json(SOURCE / "config.json")
    model = MinimindForCausalLM(ModelConfig(**config["model"])).float()
    load_model(model, str(SOURCE / "model.safetensors"), strict=True)
    model.eval()
    if state_digest(model) != EXPECTED_STATE:
        raise ValueError("reference model state differs")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE / "tokenizer", local_files_only=True)
    return model, tokenizer


def verify_chat(payload):
    with tempfile.TemporaryDirectory(prefix="sft-chat-check-") as temporary:
        result = subprocess.run([sys.executable, "-I", "-B", str(payload / "chat.py"),
                                 "--prompt", "\u4f60\u662f\u8c01\uff1f", "--max-new-tokens", "2"],
                                cwd=temporary, capture_output=True, text=True, encoding="utf-8",
                                timeout=300, env={**os.environ, "HF_HUB_OFFLINE": "1",
                                                 "TRANSFORMERS_OFFLINE": "1"})
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("isolated offline chat CLI failed")
    return {"status": "passed", "offline": True, "max_new_tokens": 2,
            "output": result.stdout.strip(), "scope": "entry-point smoke, not identity accuracy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="yanpeng051821/minimind-dense-64m-mini-sft")
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume-prepared", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.resume_prepared:
        payload = output / "payload"
        manifest = read_json(payload / "export_manifest.json")
        if (manifest["publication"]["repository"] != args.repo_id
                or sha256_file(payload / "export_manifest.json") != read_json(output / "preparation.json")["manifest_sha256"]):
            raise ValueError("prepared release identity differs")
        check_files(payload, manifest)
    else:
        payload, manifest = prepare(output, args.repo_id)
    model, tokenizer = reference_model()
    if not (output / "local_verification.json").exists():
        print("Checking public payload in an isolated offline process.", flush=True)
        verified = verify_in_fresh_process(payload, model, tokenizer)
        verified["chat_cli"] = verify_chat(payload)
        write_json(output / "local_verification.json", verified)
    if not args.publish:
        print("Prepared and locally verified; no Hub changes.", flush=True)
        return
    if not args.credential_file:
        raise ValueError("credential file required for upload")
    api = HfApi(token=credential(args.credential_file))
    account = api.whoami()["name"]
    if args.repo_id.split("/")[0] != account:
        raise ValueError("target owner differs from authenticated account")
    allowed = [item["path"] for item in manifest["files"]] + ["export_manifest.json"]
    upload_record = output / "upload.json"
    if upload_record.exists():
        record = read_json(upload_record)
        if record["repo_id"] != args.repo_id:
            raise ValueError("upload record belongs to a different repository")
        revision = record["revision"]
    else:
        marker = output / "repository_created.json"
        if not marker.exists():
            try:
                api.model_info(args.repo_id)
            except RepositoryNotFoundError:
                api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=False)
                write_json(marker, {"repo_id": args.repo_id, "created_by_this_release": True})
            else:
                raise FileExistsError("refusing to overwrite an existing repository")
        elif read_json(marker)["repo_id"] != args.repo_id:
            raise ValueError("creation marker differs")
        print(f"Uploading {len(allowed)} allowlisted files to {args.repo_id}.", flush=True)
        commit = api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=payload,
                                   allow_patterns=allowed,
                                   commit_message="Release MiniMind mini SFT: bilingual cards, weights and bounded evidence")
        revision = commit.oid
        write_json(upload_record, {"repo_id": args.repo_id, "revision": revision,
                   "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
                   "files": allowed, "manifest_sha256": sha256_file(payload / "export_manifest.json"),
                   "status": "uploaded_download_verification_pending"})
    if (output / "download_verification.json").exists():
        print(f"Already uploaded and verified: {args.repo_id}@{revision}", flush=True)
        return
    print(f"Downloading pinned revision {revision} into a fresh temporary cache.", flush=True)
    with tempfile.TemporaryDirectory(prefix="sft-hub-verify-") as temporary:
        temporary = Path(temporary)
        download = temporary / "snapshot"
        snapshot_download(repo_id=args.repo_id, revision=revision, token=api.token,
                          local_dir=download, cache_dir=temporary / "cache", allow_patterns=allowed,
                          force_download=True, max_workers=2)
        if sha256_file(download / "export_manifest.json") != sha256_file(payload / "export_manifest.json"):
            raise ValueError("downloaded manifest differs")
        check_files(download, manifest)
        verified = verify_in_fresh_process(download, model, tokenizer)
        verified["chat_cli"] = verify_chat(download)
        verified.update(repo_id=args.repo_id, revision=revision, all_file_hashes_match=True,
                        files_checked=len(allowed), fresh_download=True,
                        manifest_sha256=sha256_file(download / "export_manifest.json"),
                        verified_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "download_verification.json", verified)
    check_files(SOURCE, read_json(SOURCE / "export_manifest.json"))
    print(json.dumps({"status": "uploaded_and_verified", "repo_id": args.repo_id,
                      "revision": revision, "files": len(allowed)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not echo credential-bearing request URLs or response bodies.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "http_status": getattr(getattr(error, "response", None), "status_code", None)}))
        raise SystemExit(1) from None
