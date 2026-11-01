"""Persistent gate-and-run supervisor for the pinned full-data 2+2 experiment."""

import argparse
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # pragma: no cover - supervisor is expected to run in WSL/Linux.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AlreadyRunningError(RuntimeError):
    pass


def acquire_singleton_lock(root):
    if fcntl is None:
        raise RuntimeError("the full-data supervisor requires POSIX fcntl locking")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".pipeline.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise AlreadyRunningError(f"another supervisor holds {lock_path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def build_smoke_resume_command(command, checkpoint):
    resumed = list(command)
    try:
        stop_index = resumed.index("--stop-after-updates")
    except ValueError as exc:
        raise ValueError("smoke command is missing its stop boundary") from exc
    if stop_index + 1 >= len(resumed) or resumed[stop_index + 1] != "2":
        raise ValueError("unexpected smoke stop boundary")
    del resumed[stop_index:stop_index + 2]
    if "--smoke-records" not in resumed:
        raise ValueError("smoke resume must preserve the smoke record limit")
    return [*resumed, "--resume", str(checkpoint)]


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.lock_file = acquire_singleton_lock(self.root)
        self.status_path = self.root / "pipeline_status.json"
        self.log_path = self.root / "logs" / "pipeline.log"
        self.recipe_path = args.recipe.resolve()
        self.recipe = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        self.code_root = Path(__file__).resolve().parents[1]
        self.python = Path(sys.executable).resolve()
        self.status = {
            "status": "starting", "pid": os.getpid(), "started_at_utc": utc_now(),
            "recipe_sha256": sha256_file(self.recipe_path), "current_stage": None,
            "child_pid": None, "attempt": None,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, status, **fields):
        self.status.update(status=status, updated_at_utc=utc_now(), **fields)
        atomic_json(self.status_path, self.status)

    def log(self, message):
        line = f"{utc_now()} {message}"
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)

    def verify_download(self):
        status_path = self.args.download_status.resolve()
        while True:
            if not status_path.is_file():
                self.update("waiting_for_download", download_status_file=str(status_path))
                time.sleep(self.args.poll_seconds)
                continue
            state = json.loads(status_path.read_text(encoding="utf-8"))
            self.update("waiting_for_download", download_status_file=str(status_path),
                        download_updated_at_utc=state.get("updated_at_utc"),
                        download_files=state.get("files", {}))
            if state.get("status") == "failed":
                raise RuntimeError(f"official source downloader failed: {state.get('error', 'see downloader log')}")
            if state.get("status") == "completed":
                break
            time.sleep(self.args.poll_seconds)

        source_dir = self.args.source_dir.resolve()
        verified = {}
        for stage in ("pretrain", "sft"):
            expected = self.recipe["data"][stage]
            entry = state.get("files", {}).get(expected["filename"], {})
            source = source_dir / expected["filename"]
            if (entry.get("status") != "completed"
                    or entry.get("sha256") != expected["lfs_sha256"]
                    or entry.get("size_bytes") != expected["size_bytes"]
                    or not source.is_file()
                    or source.stat().st_size != expected["size_bytes"]):
                raise ValueError(f"download status/size gate failed for {source}")
            actual = sha256_file(source)
            if actual != expected["lfs_sha256"]:
                raise ValueError(f"whole-file SHA256 failed for {source}")
            verified[stage] = {"path": str(source), "size_bytes": source.stat().st_size,
                               "sha256": actual}
        self.update("sources_verified", source_files=verified)
        self.log("both full source files passed pinned size and whole-file SHA256")

    def run_command(self, name, command, *, retries=0, stage=None, attempt_base=1):
        command = [str(item) for item in command]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.code_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_DATASETS_CACHE"] = str(self.root / "data" / "cache" / "datasets")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONUNBUFFERED"] = "1"
        log_path = self.root / "logs" / f"{name}.log"
        for attempt in range(attempt_base, attempt_base + retries + 1):
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(f"\n[{utc_now()}] attempt={attempt} command={json.dumps(command)}\n")
                log.flush()
                self.update("running", current_stage=stage or name, child_name=name,
                            child_command=command, child_log=str(log_path), attempt=attempt)
                child = subprocess.Popen(command, cwd=self.code_root, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                self.status["child_pid"] = child.pid
                atomic_json(self.status_path, self.status)
                code = child.wait()
            self.log(f"{name} attempt {attempt} exited {code}")
            if code == 0:
                return
            if attempt >= attempt_base + retries:
                raise subprocess.CalledProcessError(code, command)
            self.log(f"{name} failed; one bounded checkpoint-resume retry remains")

    def train_stage(self, stage, *, smoke=False, initial_model=None):
        stage_dir = self.root / "stages" / (f"smoke_{stage}" if smoke else stage)
        command = [self.python, self.code_root / "scripts" / "train_full_data_stage.py",
                   "--stage", stage, "--recipe", self.recipe_path,
                   "--source-dir", self.args.source_dir.resolve(),
                   "--prepared-dir", self.args.prepared_dir.resolve(),
                   "--output-dir", stage_dir]
        if initial_model is not None:
            command.extend(["--initial-model", initial_model])
        if smoke:
            command.extend(["--smoke-records", "1024" if stage == "pretrain" else "64",
                            "--stop-after-updates", "2"])
        command_name = f"{'smoke_' if smoke else ''}{stage}"
        try:
            self.run_command(command_name, command, stage=stage)
        except subprocess.CalledProcessError:
            if smoke:
                raise
            manifest_path = stage_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = manifest.get("latest_checkpoint")
            if not checkpoint or not Path(checkpoint).is_file():
                raise RuntimeError(f"{stage} failed without a resumable checkpoint")
            self.log(f"resuming {stage} once from {checkpoint}")
            self.run_command(f"{stage}_resume", [*command, "--resume", checkpoint],
                             stage=f"{stage}_resume")
        if smoke:
            manifest_path = stage_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = manifest.get("latest_checkpoint")
            if manifest.get("status") != "paused" or not checkpoint:
                raise RuntimeError(f"{stage} smoke did not stop at a resumable checkpoint")
            resume = build_smoke_resume_command(command, checkpoint)
            self.run_command(f"smoke_{stage}_resume", resume, stage=f"smoke_{stage}_resume")
            result = json.loads((stage_dir / "result.json").read_text(encoding="utf-8"))
            if result.get("status") != "completed" or result.get("epochs") != 2:
                raise RuntimeError(f"{stage} two-epoch smoke/resume gate failed")
            return stage_dir / "final_model.pt"

        result_path = stage_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"{stage} exited successfully without a completed result")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        model_path = Path(result["final_model"])
        if (result.get("status") != "completed" or result.get("epochs") != 2
                or not model_path.is_file()
                or sha256_file(model_path) != result.get("final_model_sha256")):
            raise RuntimeError(f"{stage} final two-epoch artifact verification failed")
        self.log(f"{stage} completed: epochs=2 updates={result['global_step']} final_sha256={result['final_model_sha256']}")
        return model_path

    def estimate_formal_stage(self, stage):
        report = json.loads((self.args.prepared_dir.resolve() / stage / "audit_report.json").read_text(encoding="utf-8"))
        records = report["counts"]["train_records"]
        training = self.recipe[stage]
        updates = math.ceil(records / training["effective_batch_records"]) * training["epochs"]
        metrics = self.root / "stages" / f"smoke_{stage}" / "metrics.jsonl"
        seconds = [json.loads(line)["seconds_since_previous_update"]
                   for line in metrics.read_text(encoding="utf-8").splitlines()
                   if line and json.loads(line).get("event") == "update"]
        if not seconds:
            raise RuntimeError(f"{stage} smoke did not provide update timing")
        median = statistics.median(seconds)
        return {"train_records_per_epoch": records, "planned_updates": updates,
                "smoke_update_seconds": seconds, "median_update_seconds": median,
                "estimated_update_compute_seconds": median * updates,
                "estimate_scope": "optimizer-update time extrapolation; excludes audit, epoch validation, checkpoint I/O and final export"}

    def run(self):
        try:
            self.update("waiting_for_download", download_status_file=str(self.args.download_status.resolve()))
            self.verify_download()
            self.update("auditing_full_sources")
            audit = [self.python, self.code_root / "scripts" / "prepare_full_data.py",
                     "--source-dir", self.args.source_dir.resolve(),
                     "--output-dir", self.args.prepared_dir.resolve()]
            self.run_command("full_data_audit", audit, stage="auditing_full_sources")
            reports = {}
            for stage in ("pretrain", "sft"):
                report = self.args.prepared_dir.resolve() / stage / "audit_report.json"
                data = json.loads(report.read_text(encoding="utf-8"))
                if data.get("status") != "completed" or data["source"]["sha256"] != self.recipe["data"][stage]["lfs_sha256"]:
                    raise RuntimeError(f"{stage} full audit identity gate failed")
                reports[stage] = {"counts": data["counts"], "valid_targets": data["valid_targets"],
                                  "offset_index_sha256": data["offset_index_sha256"]}
            self.update("full_audit_completed", audit_summaries=reports)
            smoke_pretrain = self.train_stage("pretrain", smoke=True)
            self.train_stage("sft", smoke=True, initial_model=smoke_pretrain)
            estimates = {stage: self.estimate_formal_stage(stage) for stage in ("pretrain", "sft")}
            self.update("smoke_resume_gates_completed", audit_summaries=reports,
                        runtime_estimates=estimates)
            self.log("short-smoke extrapolation (optimizer updates only): " +
                     json.dumps({stage: round(value["estimated_update_compute_seconds"] / 3600, 2)
                                 for stage, value in estimates.items()}))

            pretrain_final = self.train_stage("pretrain")
            # Verify strict architecture/state loading before allowing SFT to start.
            verify = [self.python, "-c",
                      "import json,sys,torch; from pathlib import Path; "
                      "sys.path.insert(0,sys.argv[1]); from minimind_lab.models.config import ModelConfig; "
                      "from minimind_lab.models.model import MinimindForCausalLM; "
                      "p=Path(sys.argv[2]); c=json.loads(Path(sys.argv[3]).read_text()); "
                      "m=MinimindForCausalLM(ModelConfig(**{k:v for k,v in c['model'].items() if k not in {'source','tokenizer'}})); "
                      "m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True); "
                      "print('strict_load_ok',sum(x.numel() for x in m.parameters()))",
                      self.code_root / "src", pretrain_final, self.recipe_path]
            self.run_command("pretrain_strict_gate", verify, stage="pretrain_strict_gate")
            self.update("pretrain_accepted_sft_starting", pretrain_final_model=str(pretrain_final),
                        pretrain_final_model_sha256=sha256_file(pretrain_final))
            self.train_stage("sft", initial_model=pretrain_final)
            self.update("completed", completed_at_utc=utc_now(),
                        audit_summaries=reports, pretrain_final_model=str(pretrain_final))
            self.log("full-data pretrain 2 epochs -> SFT 2 epochs pipeline completed")
        except BaseException as exc:
            self.status["error"] = {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()}
            self.update("failed")
            self.log(f"PIPELINE FAILED: {type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--download-status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        Supervisor(args).run()
    except AlreadyRunningError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
