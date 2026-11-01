"""Stream and verify a deduplicated candidate against its full lineage ledger."""

import hashlib
from itertools import zip_longest
import json


def iter_candidate_rows(source, lineage, expected_source, expected_lineage):
    """Consumers must exhaust the iterator before accepting any output."""
    lineage_hash = hashlib.sha256()
    lineage_count = lineage_bytes = 0

    def kept_rows():
        nonlocal lineage_count, lineage_bytes
        kept = 0
        with lineage.open("rb") as stream:
            for lineage_count, raw in enumerate(stream, 1):
                lineage_hash.update(raw)
                lineage_bytes += len(raw)
                row = json.loads(raw)
                if row["input_line"] != lineage_count:
                    raise ValueError("unordered lineage input lines")
                if row["action"] == "keep":
                    kept += 1
                    if (row["retained_output_line"] != kept
                            or row["retained_source_line"] != row["source_line"]):
                        raise ValueError("invalid retained lineage position")
                    yield row
                elif row["action"] not in ("exclude_exact_duplicate", "exclude_review"):
                    raise ValueError("unknown lineage action")
        if (lineage_count != expected_lineage["records"] or lineage_bytes != expected_lineage["bytes"]
                or lineage_hash.hexdigest() != expected_lineage["sha256"]):
            raise ValueError("lineage identity/count mismatch")

    source_hash = hashlib.sha256()
    source_bytes = total = 0
    with source.open("rb") as stream:
        for total, (raw, row) in enumerate(zip_longest(stream, kept_rows()), 1):
            if raw is None or row is None:
                raise ValueError("candidate and kept lineage have different counts")
            source_hash.update(raw)
            source_bytes += len(raw)
            if hashlib.sha256(raw).hexdigest() != row["source_line_sha256"]:
                raise ValueError(f"candidate/lineage mismatch at line {total}")
            yield total, raw, row
    if (total != expected_source["records"] or source_bytes != expected_source["bytes"]
            or source_hash.hexdigest() != expected_source["sha256"]):
        raise ValueError("candidate identity/count mismatch")
