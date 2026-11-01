import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_sft_tool_declarations", ROOT / "scripts/audit_sft_tool_declarations.py")
audit = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(audit)


def messages(name="clock", tools=None):
    return [{"role": "system", "content": "tools", "tools": tools if tools is not None else
             '[{"type":"function","function":{"name":"clock","parameters":{}}}]'},
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": "", "tool_calls":
             [{"type": "function", "function": {"name": name, "arguments": "{}"}}]}]


def test_declared_call_is_not_flagged_and_empty_assistant_is_allowed():
    result = audit.inspect_declarations(messages())
    assert result == {"tool_record": True, "calls": 1, "issues": []}


def test_undeclared_call_is_located_by_message_and_call():
    result = audit.inspect_declarations(messages("weather"))
    assert result["issues"] == [{"code": "undeclared_function_call", "message_index": 2,
                                  "call_index": 0, "called_name": "weather", "declared_names": ["clock"]}]


def test_calls_without_declarations_are_flagged():
    result = audit.inspect_declarations(messages(tools=[]))
    assert result["issues"][0]["declared_names"] == []


def test_plain_text_tool_mentions_and_clarification_are_not_calls():
    result = audit.inspect_declarations([{"role": "user", "content": "use clock"},
                                         {"role": "assistant", "content": "No clock tool is available."}])
    assert result == {"tool_record": False, "calls": 0, "issues": []}


def test_duplicate_declarations_are_separate_review_flags():
    tools = [{"function": {"name": "clock"}}, {"function": {"name": "clock"}}]
    result = audit.inspect_declarations(messages(tools=tools))
    assert result["issues"] == [{"code": "duplicate_declared_name", "name": "clock"}]


def test_bad_arguments_are_adapter_errors_not_silently_ignored():
    data = messages()
    data[2]["tool_calls"][0]["function"]["arguments"] = "not JSON"
    result = audit.inspect_declarations(data)
    assert result["issues"][0]["code"] == "adapter_error"
