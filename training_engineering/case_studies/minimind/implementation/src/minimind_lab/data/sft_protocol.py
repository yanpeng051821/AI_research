"""Offline checks for the accepted SFT message and function-call protocol."""

from functools import lru_cache
import json

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import NoSuchResource


def _deny_remote(uri):
    raise NoSuchResource(ref=uri)


def _check_refs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("$ref", "$dynamicRef") and (not isinstance(item, str) or not item.startswith("#")):
                raise ValueError("external_schema_reference_not_allowed")
            _check_refs(item)
    elif isinstance(value, list):
        for item in value:
            _check_refs(item)


@lru_cache(maxsize=2048)
def _schema_validator(text):
    schema = json.loads(text)
    _check_refs(schema)
    cls = validator_for(schema, default=None) if "$schema" in schema else Draft202012Validator
    if cls is None:
        raise ValueError("unsupported_schema_dialect")
    cls.check_schema(schema)
    return cls(schema, registry=Registry(retrieve=_deny_remote))


def check_sft_protocol(messages, tools):
    """Check prepared messages without executing tools or inventing call IDs.

    Anonymous responses may bundle multiple results; their position can be
    checked, but the semantic mapping of individual results is not verified.
    """
    issues = []
    notices = []
    declarations = {}
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        if name in declarations:
            issues.append({"code": "duplicate_tool_declaration", "name": name})
        declarations[name] = function
    expected = "user"
    pending = []
    result_ids = set()
    all_call_ids = set()
    result_messages = 0

    def close_results():
        if not pending:
            return
        ids = [call.get("id") for call in pending]
        if all(isinstance(value, str) and value for value in ids) and result_ids:
            if set(ids) != result_ids:
                issues.append({"code": "missing_tool_result_ids"})
        elif result_messages:
            notices.append("anonymous_tool_result_block_not_semantically_matched")

    for index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            if index != 0:
                issues.append({"code": "noninitial_system", "message_index": index})
            continue
        if role == "tool":
            if not pending:
                issues.append({"code": "orphan_tool_result", "message_index": index})
            result_messages += 1
            result_id = message.get("tool_call_id")
            if result_id is not None:
                valid_ids = {call.get("id") for call in pending if isinstance(call.get("id"), str)}
                if not isinstance(result_id, str) or result_id not in valid_ids or result_id in result_ids:
                    issues.append({"code": "invalid_tool_result_id", "message_index": index})
                else:
                    result_ids.add(result_id)
            expected = "assistant_after_tool"
            continue
        if pending:
            if role != "assistant" or not result_messages:
                issues.append({"code": "unfinished_tool_exchange", "message_index": index})
            close_results()
            pending = []
            result_ids = set()
            result_messages = 0
        if role == "user":
            if expected != "user":
                issues.append({"code": "unexpected_user", "message_index": index})
            expected = "assistant"
        elif role == "assistant":
            if expected not in ("assistant", "assistant_after_tool"):
                issues.append({"code": "unexpected_assistant", "message_index": index})
            calls = message.get("tool_calls") or []
            for call in calls:
                call_id = call.get("id")
                if call_id is not None:
                    if not isinstance(call_id, str) or not call_id or call_id in all_call_ids:
                        issues.append({"code": "invalid_or_duplicate_call_id", "message_index": index})
                    else:
                        all_call_ids.add(call_id)
                function = call["function"]
                declaration = declarations.get(function["name"])
                if declaration is None:
                    issues.append({"code": "undeclared_function_call", "message_index": index})
                    continue
                arguments = function["arguments"]
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                schema = declaration.get("parameters")
                if schema is None:
                    notices.append("tool_parameter_schema_missing")
                    continue
                try:
                    validator = _schema_validator(json.dumps(schema, ensure_ascii=False, sort_keys=True))
                    for error in validator.iter_errors(arguments):
                        issues.append({"code": "invalid_tool_arguments", "message_index": index,
                                       "keyword": str(error.validator), "path": list(error.absolute_path)})
                except Exception as exc:
                    issues.append({"code": "unverifiable_parameter_schema", "message_index": index,
                                   "error_type": type(exc).__name__})
            pending = calls
            expected = "tool" if calls else "user"
        else:
            issues.append({"code": "unsupported_message_role", "message_index": index})
    if pending:
        close_results()
        issues.append({"code": "unfinished_tool_exchange", "message_index": len(messages) - 1})
    if expected != "user":
        issues.append({"code": "conversation_not_finished_by_assistant"})
    return issues, sorted(set(notices))
