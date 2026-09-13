# Evidence index

This directory is a compact, public evidence bundle for the S1 checkpoint. Paths embedded in JSON files
refer to the original training server and are provenance fields, not expected local paths.

| File | Meaning |
| --- | --- |
| `sft_s1_16k.yaml` | Resolved formal training configuration |
| `run_manifest.json` | Training identity, environment, terminal status and metrics |
| `cuda_memory_summary.json` | CUDA allocator summary |
| `validation_nll_comparison.json` | Paired per-record NLL comparison |
| `gsm8k_comparison.json` | Paired GSM8K summary |
| `regression_comparison.json` | Paired 59-task regression summary |
| `math500_smoke_comparison.json` | Truncated 20-question smoke result |
| `math500_abort.json` | Budget-guard record for the incomplete full run |
| `generation_health_2048.json` | Four-prompt bounded generation-health result |
| `evaluation_v1_executed.yaml` | Historical contract used by completed evaluations |
| `evaluation_v2_deferred.yaml` | Revised, guarded contract for future evaluation |
| `SHA256SUMS` | Integrity hashes for the evidence files |

Large per-sample outputs, training data and optimizer checkpoints are intentionally omitted. The compact
summaries retain hashes that bind them to the original detailed artifacts.
