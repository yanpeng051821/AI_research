import json


from minimind_lab.data.validator import validate_pretrain_record, validate_sft_record
from minimind_lab.data.reader import iter_jsonl

from pathlib import Path
from collections.abc import Iterator


VALIDATORS = {
    "pretrain" : validate_pretrain_record,
    "sft" : validate_sft_record,
}

def audit_parsed_line(parsed_line: dict, dataset_type: str) -> dict:
    if dataset_type not in VALIDATORS:
        raise ValueError(f"unsupported dataset type: {dataset_type}")    

    if parsed_line["error_code"] is not None:
        return {
            "source_path": parsed_line["source_path"],
            "line_number": parsed_line["line_number"],
            "raw_line": parsed_line["raw_line"],
            "record": parsed_line["record"],
            "issues": [parsed_line["error_code"]],
            "accepted": False,
        }

    validator = VALIDATORS[dataset_type]
    issues = validator(parsed_line["record"])

    accepted = len(issues) == 0

    return {
        "source_path": parsed_line["source_path"],
        "line_number": parsed_line["line_number"],
        "raw_line": parsed_line["raw_line"],
        "record": parsed_line["record"],
        "issues": issues,
        "accepted": accepted,
    }


def iter_audited_jsonl(path:str | Path, dataset_type: str)-> Iterator[dict]:
    
    for parsed_line in iter_jsonl(path):
        yield audit_parsed_line(parsed_line, dataset_type)


def audit_jsonl(input_path: str | Path, dataset_type: str, output_dir: str | Path)->dict:
    # 检查数据类型
    if dataset_type not in VALIDATORS:
        raise ValueError(f"unsupported dataset type: {dataset_type}")

    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8"):
        pass
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "report.json"

    accepted_counts = 0
    rejected_counts = 0

    errors_by_code = {}

    with (
        accepted_path.open("w", encoding="utf-8") as accepted_file,
        rejected_path.open("w", encoding="utf-8") as rejected_file,
    ):
        for audited_line in iter_audited_jsonl(input_path, dataset_type):
            output_record = {
                key: value
                for key , value in audited_line.items()
                if key != "accepted"
            }

            serialized_record = json.dumps(output_record, ensure_ascii=False)

            if audited_line["accepted"]:
                accepted_file.write(json.dumps(audited_line["record"], ensure_ascii=False) + "\n")
                accepted_counts += 1
            else:
                rejected_file.write(serialized_record + "\n")
                rejected_counts += 1
                
                for error_code in audited_line["issues"]:
                    errors_by_code[error_code] = errors_by_code.get(error_code, 0) + 1

    report = {
                "source_path": str(input_path),
                "dataset_type": dataset_type,
                "total_lines": accepted_counts + rejected_counts,
                "accepted_lines": accepted_counts,
                "rejected_lines": rejected_counts,
                "errors_by_code": dict(sorted(errors_by_code.items())),
            }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    return report

