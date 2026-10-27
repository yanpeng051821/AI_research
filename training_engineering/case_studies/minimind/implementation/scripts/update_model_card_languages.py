"""Update only the English/Chinese cards and their native-package manifest on HF."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, ModelCard, hf_hub_download

from publish_pretrain_bundle import credential, read_json, write_json, write_text
from load_exported_pretrain import sha256_file


REPO = "yanpeng051821/minimind-dense-64m-mini-pretrain"


def add_chinese_navigation(card):
    if not card.startswith("---\n"):
        raise ValueError("expected existing model-card front matter")
    end = card.index("\n---\n", 4) + len("\n---\n")
    nav = f"\n[English](https://huggingface.co/{REPO}) | **简体中文**\n"
    return card[:end] + nav + card[end:]


def check_translation(english, chinese, original):
    if ModelCard(english).data.to_dict() != ModelCard(original).data.to_dict():
        raise ValueError("model metadata must remain unchanged")
    if ModelCard(chinese).data.to_dict() != ModelCard(original).data.to_dict():
        raise ValueError("Chinese metadata must remain unchanged")
    blocks = lambda text: re.findall(r"```(?:text|shell)\n(.*?)```", text, re.S)
    if blocks(english) != blocks(original) or blocks(chinese) != blocks(original):
        raise ValueError("preserve original commands and generated Chinese text verbatim")
    for value in ("63,912,192", "1,265,235", "5,000", "301,810,690", "1,206,758",
                  "8.912246", "1.998545", "4,943", "32", "768"):
        if value not in english or value not in chinese:
            raise ValueError(f"missing experimental value in translation: {value}")
    for digest in re.findall(r"\b[0-9a-f]{64}\b", original):
        if digest not in english or digest not in chinese:
            raise ValueError("preserve every original SHA256 identity")
    if f"https://huggingface.co/{REPO}/blob/main/README_zh.md" not in english:
        raise ValueError("missing Chinese navigation")
    if f"https://huggingface.co/{REPO}" not in chinese:
        raise ValueError("missing English navigation")


def new_manifest(parent, cards, parent_revision):
    result = {**parent, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "documentation_update": {
                  "parent_hf_revision": parent_revision,
                  "kind": "English default card with linked Chinese version",
                  "changed_files": ["README.md", "README_zh.md", "export_manifest.json"],
                  "update_script_sha256": sha256_file(Path(__file__)),
                  "model_runtime_or_weights_changed": False,
              }}
    items = [item for item in parent["files"] if item["path"] not in cards]
    for name, text in cards.items():
        raw = text.encode("utf-8")
        items.append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    result["files"] = sorted(items, key=lambda item: item["path"])
    return result


def remote_identities(info):
    return {s.rfilename: {"blob_id": s.blob_id, "bytes": s.size,
                          "lfs_sha256": s.lfs.sha256 if s.lfs else None} for s in info.siblings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-card", type=Path, required=True)
    parser.add_argument("--parent-revision", required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    args = parser.parse_args()
    if args.report_dir.exists():
        raise FileExistsError("choose a new documentation update version")
    api = HfApi(token=credential(args.credential_file))
    if api.whoami()["name"] != REPO.split("/")[0]:
        raise ValueError("unexpected authenticated account")
    before = api.model_info(REPO, files_metadata=True)
    if before.sha != args.parent_revision:
        raise ValueError("HF main changed; review it before updating")
    with tempfile.TemporaryDirectory(prefix="minimind-card-parent-") as temporary:
        download = lambda name: Path(hf_hub_download(REPO, name, revision=before.sha,
                                    token=False, cache_dir=temporary, force_download=True))
        parent_raw = download("export_manifest.json").read_bytes()
        if parent_raw != args.parent_manifest.read_bytes():
            raise ValueError("remote manifest differs from the recorded parent release")
        original = download("README.md").read_text(encoding="utf-8")
    english = args.english_card.read_text(encoding="utf-8")
    chinese = add_chinese_navigation(original)
    check_translation(english, chinese, original)
    cards = {"README.md": english, "README_zh.md": chinese}
    manifest = new_manifest(json.loads(parent_raw), cards, before.sha)
    args.report_dir.mkdir(parents=True)
    payload = args.report_dir / "payload"
    for name, text in cards.items():
        write_text(payload / name, text)
    write_json(payload / "export_manifest.json", manifest)
    changed = ["README.md", "README_zh.md", "export_manifest.json"]
    commit = api.create_commit(repo_id=REPO, repo_type="model", parent_commit=before.sha,
                operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload / name)
                            for name in changed],
                commit_message="Add English model card and linked Chinese version")
    write_json(args.report_dir / "upload.json", {"repo_id": REPO, "parent_revision": before.sha,
               "revision": commit.oid, "changed_files": changed,
               "export_manifest_sha256": sha256_file(payload / "export_manifest.json")})
    after = HfApi(token=False).model_info(REPO, revision=commit.oid, files_metadata=True)
    old_ids, new_ids = remote_identities(before), remote_identities(after)
    unchanged = set(old_ids) - set(changed)
    if any(new_ids.get(name) != old_ids[name] for name in unchanged):
        raise ValueError("an unexpected non-documentation file changed")
    if set(new_ids) != set(old_ids) | {"README_zh.md"}:
        raise ValueError("unexpected additions or removals")
    with tempfile.TemporaryDirectory(prefix="minimind-card-verify-") as temporary:
        for name in changed:
            path = Path(hf_hub_download(REPO, name, revision=commit.oid, token=False,
                                       cache_dir=temporary, force_download=True))
            if path.read_bytes() != (payload / name).read_bytes():
                raise ValueError(f"downloaded documentation differs: {name}")
    write_json(args.report_dir / "verification.json", {
        "status": "documentation_updated_and_verified", "repo_id": REPO,
        "parent_revision": before.sha, "revision": commit.oid,
        "changed_files": changed, "unchanged_remote_files_verified": len(unchanged),
        "weight_lfs_sha256": new_ids["model.safetensors"]["lfs_sha256"],
        "english_source_sha256": sha256_file(args.english_card),
        "update_script_sha256": sha256_file(Path(__file__)),
        "export_manifest_sha256": sha256_file(payload / "export_manifest.json"),
        "default_card_language": "English", "chinese_card": "README_zh.md",
        "navigation": "Markdown links between documents, not a native HF language setting",
        "scope": "Fresh anonymous download of all three updated files; unchanged weight/runtime "
                 "identities compared with the previously download-and-load-verified parent. "
                 "No new training, weight conversion, full weight download, or inference run.",
    })
    print(json.dumps({"status": "documentation_updated_and_verified", "repo_id": REPO,
                      "revision": commit.oid, "unchanged_files": len(unchanged)}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "http_status": getattr(getattr(exc, "response", None), "status_code", None)}))
        raise SystemExit(1)
