"""Bounded CUDA allocation diagnostic; does not load or train a model."""

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path

import torch


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
        )
    ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def snapshot(label):
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    fn = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    fn.argtypes = [ctypes.POINTER(PerformanceInfo), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    free, total = torch.cuda.mem_get_info()
    return {"label": label, "system_commit_bytes": info.CommitTotal * info.PageSize,
            "system_commit_limit_bytes": info.CommitLimit * info.PageSize,
            "system_commit_available_bytes": (info.CommitLimit - info.CommitTotal) * info.PageSize,
            "physical_available_bytes": info.PhysicalAvailable * info.PageSize,
            "cuda_free_bytes": free, "cuda_total_bytes": total,
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved()}


def run(output, max_mib):
    if output.exists():
        raise FileExistsError(output)
    if not 64 <= max_mib <= 2048 or max_mib % 64:
        raise ValueError("require a multiple of 64 MiB, capped at 2048 MiB")
    rows, held = [], []
    status, error = "allocation_cap_reached", None
    try:
        rows.append(snapshot("cuda_initialized"))
        for index in range(max_mib // 64):
            if rows[-1]["system_commit_available_bytes"] < 1536 * 1024**2:
                status = "stopped_at_system_commit_safety_margin"
                break
            if rows[-1]["cuda_free_bytes"] < 1024**3:
                status = "stopped_at_cuda_free_safety_margin"
                break
            held.append(torch.empty(64 * 1024**2, dtype=torch.uint8, device="cuda"))
            held[-1].fill_(1)
            torch.cuda.synchronize()
            rows.append(snapshot(f"allocated_{(index + 1) * 64}_MiB"))
    except torch.OutOfMemoryError as exc:
        status, error = "allocation_failed", str(exc)
        rows.append(snapshot("allocation_failure"))
    finally:
        held.clear()
        torch.cuda.empty_cache()
    result = {"status": status, "error": error, "rows": rows,
              "training_started": False, "max_requested_mib": max_mib,
              "limitations": "Isolated allocation diagnostic, not model forward or backward."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=1024)
    args = parser.parse_args()
    run(args.output, args.max_mib)
