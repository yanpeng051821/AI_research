"""Download and verify the two official full-data files at a pinned revision."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
import requests


REPO = "jingyaogong/minimind_dataset"
REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
FILES = ("pretrain_t2t.jsonl", "sft_t2t.jsonl")
ENV_FILE = Path(r"D:\pythonlearning\AI_research\.env") if os.name == "nt" else Path("/mnt/d/pythonlearning/AI_research/.env")
CHUNK_SIZE = 16 * 1024 * 1024
WORKERS = 6


def read_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if not ENV_FILE.is_file():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().lower() == "hf":
            value = value.strip().strip("\"'")
            return value or None
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(path, status, files, error=None):
    payload = {
        "repo": REPO,
        "revision": REVISION,
        "status": status,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def download_chunk(url, start, end, size, part_path, token, refresh_url):
    expected_len = end - start + 1
    if part_path.is_file() and part_path.stat().st_size == expected_len:
        return hashlib.sha256(part_path.read_bytes()).hexdigest()
    part_path.unlink(missing_ok=True)
    temp = part_path.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    current_url = url
    for attempt in range(6):
        try:
            with requests.get(
                current_url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(15, 60),
            ) as response:
                if response.status_code in (403, 410):
                    current_url = refresh_url()
                    continue
                response.raise_for_status()
                expected_range = f"bytes {start}-{end}/{size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"unexpected range response for bytes {start}-{end}")
                digest = hashlib.sha256()
                written = 0
                with temp.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
                            digest.update(block)
                            written += len(block)
                if written != expected_len:
                    raise RuntimeError(f"short range: expected {expected_len}, received {written}")
                temp.replace(part_path)
                return digest.hexdigest()
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))
            current_url = refresh_url()
    raise RuntimeError(f"could not download bytes {start}-{end}")


def prepare_url(filename, token):
    url = hf_hub_url(REPO, filename, repo_type="dataset", revision=REVISION)
    metadata = get_hf_file_metadata(url, token=token, timeout=20)
    guard = threading.Lock()
    state = {"url": metadata.location}

    def current_url():
        with guard:
            return state["url"]

    def refresh_url():
        fresh = get_hf_file_metadata(url, token=token, timeout=20)
        with guard:
            state["url"] = fresh.location
            return state["url"]

    return metadata, current_url(), refresh_url


def download_file(name, identity, output_dir, state, status_path, status_lock, token):
    size = identity["size_bytes"]
    parts_dir = output_dir / ".parts" / name
    parts_dir.mkdir(parents=True, exist_ok=True)
    metadata, url, refresh_url = prepare_url(name, token)
    if metadata.size != size or metadata.etag != identity["sha256"]:
        raise RuntimeError(f"pinned metadata changed for {name}")
    count = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
    state[name].update({"status": "downloading", "chunks_total": count, "chunks_verified": 0})
    with status_lock:
        write_status(status_path, "downloading", state)
    part_hashes = [None] * count
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for index in range(count):
            start = index * CHUNK_SIZE
            end = min(size - 1, start + CHUNK_SIZE - 1)
            part_path = parts_dir / f"{index:06d}.part"
            future = pool.submit(download_chunk, url, start, end, size, part_path, token, refresh_url)
            futures[future] = (index, part_path)
        try:
            for future in as_completed(futures):
                index, _ = futures[future]
                part_hashes[index] = future.result()
                completed += 1
                if completed % 8 == 0 or completed == count:
                    state[name]["chunks_verified"] = completed
                    with status_lock:
                        write_status(status_path, "downloading", state)
        except Exception:
            for future in futures:
                future.cancel()
            raise

    target = output_dir / name
    temp_target = target.with_suffix(target.suffix + ".assembling")
    with temp_target.open("wb") as output:
        for index in range(count):
            part_path = parts_dir / f"{index:06d}.part"
            with part_path.open("rb") as part:
                for block in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    output.write(block)
    actual_size = temp_target.stat().st_size
    actual_sha = sha256_file(temp_target)
    if actual_size != size or actual_sha != identity["sha256"]:
        raise RuntimeError(f"whole-file size/SHA256 mismatch for {name}")
    temp_target.replace(target)
    state[name].update({
        "path": str(target),
        "actual_size_bytes": actual_size,
        "actual_sha256": actual_sha,
        "chunks_verified": count,
        "status": "verified",
    })
    with status_lock:
        write_status(status_path, "downloading", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    token = read_token()
    api = HfApi(token=token)
    entries = {item.path: item for item in api.list_repo_tree(
        REPO, repo_type="dataset", revision=REVISION
    )}
    expected = {}
    for name in FILES:
        item = entries[name]
        lfs = getattr(item, "lfs", None)
        if not lfs or not getattr(lfs, "sha256", None):
            raise RuntimeError(f"pinned LFS SHA256 unavailable for {name}")
        expected[name] = {"size_bytes": item.size, "sha256": lfs.sha256}

    state = {name: {**identity, "status": "pending"} for name, identity in expected.items()}
    write_status(args.status_file, "downloading", state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_lock = threading.Lock()
    try:
        for name in FILES:
            download_file(name, expected[name], args.output_dir, state, args.status_file, status_lock, token)
        write_status(args.status_file, "completed", state)
    except Exception as exc:
        write_status(args.status_file, "failed", state, f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
