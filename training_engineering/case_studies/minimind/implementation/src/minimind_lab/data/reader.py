import json

from pathlib import Path
from collections.abc import Iterator


def parse_json_line(line: str, line_number: int) -> dict:
    record = None
    error_code = None

    if line.strip() != "":
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            error_code = "invalid_json"
    else:
        error_code = "blank_line"

    return {
        "line_number": line_number,
        "record": record,
        "error_code": error_code,
    }



def iter_jsonl(path: str | Path)-> Iterator[dict]:
    jsonl_path = Path(path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            result = parse_json_line(line, line_number)  

            yield {
                "source_path": str(jsonl_path),
                "raw_line": line,
                "line_number": line_number,
                "record": result["record"],
                "error_code": result["error_code"],
            }  
