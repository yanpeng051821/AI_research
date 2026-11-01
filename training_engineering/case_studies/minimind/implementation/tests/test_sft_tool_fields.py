from copy import deepcopy
import json

import pytest

from minimind_lab.data.sft import prepare_sft_messages


@pytest.fixture
def tool_messages():
    return [
        {"role": "system", "content": "instructions", "tools": [{
            "function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }},
        }]},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "reasoning_content": "plan",
         "tool_calls": [{"id": "call-1", "function": {
             "name": "lookup", "arguments": '{ "query": "question" }',
         }}]},
        {"role": "tool", "content": '{"result":"one"}'},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.parametrize("serialized", [False, True])
def test_prepare_normalizes_lists_and_preserves_original_record(tool_messages, serialized):
    if serialized:
        tool_messages[0]["tools"] = json.dumps(tool_messages[0]["tools"])
        tool_messages[2]["tool_calls"] = json.dumps(tool_messages[2]["tool_calls"])
    before = deepcopy(tool_messages)
    messages, tools = prepare_sft_messages(tool_messages)
    assert [m["role"] for m in messages] == [m["role"] for m in before]
    assert "tools" not in messages[0]
    assert tools[0]["function"]["name"] == "lookup"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{ "query": "question" }'
    assert messages[2]["reasoning_content"] == "plan"
    assert messages[2]["tool_calls"][0]["id"] == "call-1"
    assert messages[3]["content"] == before[3]["content"]
    tools[0]["function"]["parameters"]["required"].append("extra")
    messages[2]["tool_calls"][0]["function"]["name"] = "changed"
    assert tool_messages == before


def test_prepare_accepts_argument_objects_without_aliasing(tool_messages):
    tool_messages[2]["tool_calls"][0]["function"]["arguments"] = {"query": "question"}
    messages, _ = prepare_sft_messages(tool_messages)
    messages[2]["tool_calls"][0]["function"]["arguments"]["query"] = "changed"
    assert tool_messages[2]["tool_calls"][0]["function"]["arguments"] == {"query": "question"}


@pytest.mark.parametrize("empty", [None, "", "  ", [], "[]"])
def test_optional_empty_tool_fields_are_normalized(empty):
    messages = [{"role": "user", "content": "question", "tools": empty},
                {"role": "assistant", "content": "answer", "tool_calls": empty,
                 "reasoning_content": None}]
    prepared, tools = prepare_sft_messages(messages)
    assert tools == []
    assert "tools" not in prepared[0]
    assert prepared[1]["tool_calls"] == []
    assert prepared[1]["reasoning_content"] is None


@pytest.mark.parametrize("field,message_index", [("tools", 0), ("tool_calls", 2)])
@pytest.mark.parametrize("bad", ["broken JSON", "{}", 1])
def test_invalid_list_fields_fail_with_location(tool_messages, field, message_index, bad):
    tool_messages[message_index][field] = bad
    before = deepcopy(tool_messages)
    with pytest.raises(ValueError, match=rf"messages\[{message_index}\].{field}"):
        prepare_sft_messages(tool_messages)
    assert tool_messages == before


@pytest.mark.parametrize("field,message_index", [("tools", 0), ("tool_calls", 2)])
@pytest.mark.parametrize("bad_function", [None, {"name": " "}])
def test_invalid_function_structure_is_rejected(tool_messages, field, message_index, bad_function):
    tool_messages[message_index][field] = [{"function": bad_function}]
    with pytest.raises(ValueError, match="function"):
        prepare_sft_messages(tool_messages)


@pytest.mark.parametrize("bad", [None, [], "[]", "broken JSON"])
def test_invalid_arguments_are_rejected(tool_messages, bad):
    tool_messages[2]["tool_calls"][0]["function"]["arguments"] = bad
    with pytest.raises(ValueError, match="arguments"):
        prepare_sft_messages(tool_messages)


def test_missing_arguments_are_not_fabricated(tool_messages):
    del tool_messages[2]["tool_calls"][0]["function"]["arguments"]
    with pytest.raises(ValueError, match="arguments is required"):
        prepare_sft_messages(tool_messages)


def test_noninitial_tool_declarations_are_not_silently_dropped(tool_messages):
    tools = tool_messages[0].pop("tools")
    tool_messages.insert(2, {"role": "system", "content": "later", "tools": tools})
    with pytest.raises(ValueError, match="first system message"):
        prepare_sft_messages(tool_messages)


def test_tool_calls_require_assistant_role(tool_messages):
    tool_messages[1]["tool_calls"] = tool_messages[2]["tool_calls"]
    with pytest.raises(ValueError, match="requires the assistant role"):
        prepare_sft_messages(tool_messages)


def test_reasoning_and_parameters_have_explicit_type_checks(tool_messages):
    tool_messages[2]["reasoning_content"] = ["not text"]
    with pytest.raises(ValueError, match="reasoning_content"):
        prepare_sft_messages(tool_messages)
    tool_messages[2]["reasoning_content"] = "plan"
    tool_messages[0]["tools"][0]["function"]["parameters"] = []
    with pytest.raises(ValueError, match="parameters"):
        prepare_sft_messages(tool_messages)


def test_adapter_does_not_claim_tool_protocol_or_quality_acceptance(tool_messages):
    tool_messages[2]["tool_calls"][0]["function"] = {
        "name": "undeclared_tool", "arguments": "{}",
    }
    messages, tools = prepare_sft_messages(tool_messages)
    assert tools[0]["function"]["name"] == "lookup"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "undeclared_tool"


def test_adapter_reuses_basic_conversation_validation():
    with pytest.raises(ValueError, match="empty_conversations"):
        prepare_sft_messages([])
