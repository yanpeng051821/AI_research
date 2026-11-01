# Full-Data Pretrain + SFT v1

Status: **official full-data download active; persistent 2+2 supervisor being started** (2026-11-01). Formal training remains gated on full SHA verification, full audit, and smoke/resume checks.

This directory is reserved for a new experiment. It does not alter the completed mini pretraining/SFT runs, their frozen contracts, local checkpoints, or published model packages.

## Official reference checked

- Verified official GitHub main HEAD on 2026-11-01: `233d00fa068ec2b8c0d4f07b703934d6391a5a73`. The user's local reference checkout is older and has unrelated working-tree edits, so it was not changed. The pinned official README identifies `pretrain_t2t.jsonl` + `sft_t2t.jsonl` as the MiniMind-3 mainline and recommends pretraining sequence length around 380; mini files are for quick reproduction. Mainline SFT includes tool-call examples.
- Official trainer defaults at the pinned HEAD: pretrain 2 epochs, batch 32, accumulation 8, LR 5e-4, max length 340; full SFT 2 epochs, batch 16, accumulation 1, LR 1e-5, max length 768. The official schedule is cosine to 0.1x peak LR. These are entry defaults; the README's pretraining length recommendation is about 380.
- Official sources pinned to the checked HEAD: [MiniMind README](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/README.md), [pretraining entry](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_pretrain.py), [full SFT entry](https://github.com/jingyaogong/minimind/blob/233d00fa068ec2b8c0d4f07b703934d6391a5a73/trainer/train_full_sft.py), and [official dataset repository at its immutable revision](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5).

## Material differences from the completed mini runs

| Dimension | Completed mini run | Official full-data direction | Consequence |
| --- | --- | --- | --- |
| Pretrain source | 1,265,235 records, own split/manifest | `pretrain_t2t.jsonl` (~10 GB) | Different corpus, source revision and record count must be frozen. |
| Pretrain length | 768 tokens | README recommends about 380; trainer default 340 | Changes truncation, compute and supervised-token distribution. |
| SFT source | 785,377 records, audited mini source and derived split | `sft_t2t.jsonl` (~14 GB) | Different source and format; the mini audit cannot stand in for full-file validation. |
| Epochs | One complete epoch per stage | Official trainer defaults to two epochs | Approximately doubles stage update count relative to one epoch. |
| Execution | Own Dense ~64M model, tokenizer, optimizer/training components | Keep own Dense ~64M model and tokenizer; add the already-tested Linux Flash adapter in a new bounded entry | Not an official upstream model implementation; backend changes numerical execution and needs its own identity and gates. |
| Batch on 8 GB | Mini probes used pretrain micro 4 x accum 64 and SFT micro 4 x accum 4 | Official pretrain micro 32 x accum 8 = effective 256; SFT micro 16 x accum 1 = effective 16 | Full entry uses microbatch 4 with accumulation 64 / 4, preserving those effective batches; the forced Flash backend must pass a real-data smoke gate. |

## Blocking evidence and decision gate

Correction to the initial note: range reports `full_data_capacity_probe_v1/full_source_range_inspection.json` and `_v2.json` already sampled both full files at dataset revision `312afb4f76391145c6902f765bb51691c09a12f5`. Five byte ranges per file returned HTTP 206 and were checked against `Content-Range`; these are bounded samples, not full-file hashes or structure audits. The source size recorded there is 8,275,074,893 bytes for pretraining and 14,096,018,369 bytes for SFT.

On 2026-11-01, a Windows `huggingface_hub` metadata request succeeded using the existing local `.env` credential (the credential itself is never logged). Official pinned LFS identities:

| File | Immutable revision | Size | LFS SHA256 |
| --- | --- | ---: | --- |
| `pretrain_t2t.jsonl` | `312afb4f76391145c6902f765bb51691c09a12f5` | 8,275,074,893 | `31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d` |
| `sft_t2t.jsonl` | `312afb4f76391145c6902f765bb51691c09a12f5` | 14,096,018,369 | `b5dab3d590b3b4ae3bfeec891ba0c76b0de5acbec4f4b078dcd3912eb1735c79` |

The local HTTPS tests were mixed: Windows `curl.exe` and WSL Python/curl had TLS/connection failures; authenticated Windows `huggingface_hub` successfully resolved both file entries, sizes, Git blob IDs and LFS SHA256. A resumable 16 MiB HTTP Range downloader then made verified progress through the Hub CDN using normal TLS verification. The earlier TLS failure was not evidence that the official files or revision were absent.

The prior bounded SFT scan at this same revision found length and protocol-adapter observations, including `orphan_tool_result`; these are diagnostic categories, not automatic exclusions. Full-file structural/encoding audit and full training-target accounting remain required after download.

The local WSL `MiniMind-Flash-Bench` reports an RTX 4060 Ti with 8 GiB VRAM and 908 GiB available on E:. Its existing isolated Linux environment is `/opt/minimind-flash-venv` (Torch 2.14.0+cu130). A previous short probe measured about 45.59 pretraining records/s at length 768 and microbatch 4; it is not a valid estimate for length 340 or these full files. After full audit, 1,024-record pretrain and 64-record SFT real-data smoke gates record update timings and extrapolate update-only runtime before formal training. The user confirmed two complete epochs for each stage and accepts runs over 100 hours.

## Execution and storage

- Raw pinned JSONL, SQLite dedup ledger, compact 64-bit offset indices, reports, logs, checkpoints, and model exports are under `E:/AI_research_checkpoints/minimind/full_pretrain_sft_v1`; code and recipe remain in this repository.
- Audit reads each JSONL line-by-line. Training lazily reads only indexed rows; it does not load full JSONL or tokenized corpora into RAM. A deterministic 1,024-record block shuffle limits mechanical-drive seeks while retaining full-epoch coverage; order and committed position are checkpointed.
- The new Linux entry force-selects `SDPBackend.FLASH_ATTENTION`, verifies strict right padding, and fails rather than falling back to hand-written attention. Runtime logs and identity record the selected backend.
- `implementation/scripts/full_data_supervisor.py` waits for downloader whole-file SHA success, rechecks both files, runs full streaming audit/split and two-epoch pretrain/SFT smoke-resume gates, then starts formal pretraining. Only a completed pretrain result passing file SHA and strict model load may seed SFT. Each formal stage permits at most one identity-bound checkpoint-resume attempt. Full-state checkpoints are written every 200 optimizer updates, with latest periodic and epoch-boundary checkpoints retained.
- The WSL supervisor holds a nonblocking OS `fcntl.flock` on E:'s `.pipeline.lock` throughout execution. The lock file is intentionally persistent and never deleted; a second supervisor exits before touching status or launching a child.
- The audit reports structural exclusions, exact duplicates, encoding/template exclusions, truncation, valid targets, protocol diagnostics, train/validation index hashes, and bounded review samples. Unsupported tool protocol is diagnostic, not an automatic exclusion when encoding succeeds.
- User authorized persistent local 2+2 GPU execution. No paid/cloud compute, HF upload, old checkpoint replacement, or Git push is part of this run.

Formal training gates are enforced by the persistent supervisor. The confirmed schedule is two epochs per stage; pretraining length 340 follows the official entry default while the README recommends approximately 380.

Download entry: `implementation/scripts/download_full_sources.py`. It reads the existing `.env` credential without printing it, downloads the pinned files into E:, writes atomic status, and checks each full SHA256. The first Hub-managed download (PID `42960`) was stopped after it remained at 0 bytes. The resumable range downloader is running as Windows PID `52088` (Python child `53424`); the most recent live status recorded 256 of 494 pretraining chunks verified, with SFT pending. This is active download progress, not whole-file verification. Launch record is `E:/AI_research_checkpoints/minimind/full_pretrain_sft_v1/download_process.json`, live status is `download_status.json`, and logs are in `logs/`. The lock-protected WSL supervisor is PID `375`, waiting on the full-SHA gate. To inspect:

```powershell
Get-Content E:\AI_research_checkpoints\minimind\full_pretrain_sft_v1\download_status.json
Get-Process -Id 52088
Get-Content E:\AI_research_checkpoints\minimind\full_pretrain_sft_v1\logs\download.stderr.log -Tail 30
Get-Content E:\AI_research_checkpoints\minimind\full_pretrain_sft_v1\pipeline_status.json
Get-Content E:\AI_research_checkpoints\minimind\full_pretrain_sft_v1\logs\pipeline.log -Tail 30
```

Raw/derived data, checkpoints and logs belong under `E:/AI_research_checkpoints/minimind/full_pretrain_sft_v1` (mounted at `/mnt/e/...`). SFT starts only after this experiment's pretraining exits 0 and its final model passes SHA, strict load, and validation gates. No old mini checkpoint is a fallback. This file is an execution-status record, not a frozen training contract.
