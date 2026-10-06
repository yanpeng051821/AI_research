import pytest 

from minimind_lab.data.validator import (
    validate_pretrain_record,
    validate_sft_record,
)

from copy import deepcopy



@pytest.mark.parametrize(
    ("record", "expected_issues"),
    [
    ([], ["record_not_object"]),
    (123, ["record_not_object"]),
    ({}, ["missing_text"]),
    ({"text": None}, ["text_not_string"]),
    ({"text": ""}, ["empty_text"]),
    ({"text": " \n\t"}, ["empty_text"]),
    ({"text": "hello"}, []),
    ({"text": "hello", "source": "wiki"},[]),
    ({"text": 123}, ["text_not_string"]),
    ]
)
def test_validate_pretrain_record(record, expected_issues):
    assert validate_pretrain_record(record) == expected_issues


def test_validate_pretrain_record_does_not_modify_input():
    record = {
        "text": "hello",
        "metadata": {"source": "wiki"},
    }
    original_record = deepcopy(record)
    
    validate_pretrain_record(record)


    assert record == original_record



@pytest.mark.parametrize(
    ("record", "expected_issues"),
    [
        ([], ["record_not_object"]),
        ({}, ["missing_conversations"]),
        ({"conversations": "hello"}, ["conversations_not_list"]),
        ({"conversations": []}, ["empty_conversations"]),
        ({"conversations": [{"role": "user", "content": "hello"},{"role": "assistant", "content": "hello"}]}, []),
        ({"conversations": [["xxx"], {"role": "assistant", "content": "hello"}]}, ["message_not_object"]),
        ({"conversations": [{"content": "hello"}]}, ["missing_role", "missing_assistant"]),
        ({"conversations": [{"role": 123, "content": "hello"}, {"role": "assistant", "content": "hello"}]}, ["role_not_string"]),
        ({"conversations": [{"role": "unknown", "content": "hello"}, {"role": "assistant", "content": "hello"}]}, ["invalid_role"]),
        ({"conversations": [{"role": "assistant"}]}, ["missing_content"]),
        ({"conversations": [{"role": "assistant", "content": None}]}, ["content_not_string"]),
        ({"conversations": [{"role": "user"}, {"role": "assistant"}]}, ["missing_content"])
    ]
)
def test_validate_sft_record(record, expected_issues):
    assert validate_sft_record(record) == expected_issues


def test_validate_sft_record_does_not_modify_input():
    record = {
        "conversations": [
            {"role": "system", "content": "hello"},
            {"role": "assistant", "content": "hello"}
        ]
    }

    original_record = deepcopy(record)

    validate_sft_record(record)
    assert record == original_record