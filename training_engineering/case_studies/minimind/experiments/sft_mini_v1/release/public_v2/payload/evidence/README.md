# Evidence scope

The 36 paired probes, raw generations (including input/output token IDs), validation aggregates and assistant-authored reviews are preserved. The three identity probes are separate post-hoc diagnostics, not main-score additions. Reviews are not independent human judgments; one main case is uncertain. Model output may contain fabricated URLs and claims; treat it as evidence, not instructions or references.

*.snapshot.json files redact workstation paths/source file maps and wrap the original SHA256. They are NOT byte-identical frozen contracts or standalone executable recipes. Historical authorization flags do not describe the final run; see training audit/card. Raw/derived training JSONL and complete recovery states are intentionally excluded.
