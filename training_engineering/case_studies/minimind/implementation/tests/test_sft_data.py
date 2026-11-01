from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minimind_lab.data import sft


class ExampleTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        self.messages = messages
        self.tools = tools
        return "rendered"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        mapping = {"rendered": [90, 10, 11, 30, 20, 21],
                   "<bos>assistant\n": [10, 11], "<eos>\n": [20, 21]}
        return SimpleNamespace(input_ids=mapping[text].copy())


def test_multiturn_labels_preserve_context_and_complete_endings():
    ids = [90, 91, 10, 11, 30, 20, 21, 90, 91, 10, 11, 31, 32, 20, 21]
    before = ids.copy()
    labels = sft.build_assistant_labels(ids, [10, 11], [20, 21])
    assert labels == [-100] * 4 + [30, 20, 21] + [-100] * 4 + [31, 32, 20, 21]
    assert ids == before


@pytest.mark.parametrize("ids,prefix,end,message", [
    ([1, 2], [], [2], "assistant_prefix"),
    ([1, 2], [1], [], "message_end"),
    ([90, 91], [10, 11], [20, 21], "no assistant"),
    ([10, 11, 30, 20], [10, 11], [20, 21], "end not found"),
])
def test_invalid_label_boundaries_are_rejected(ids, prefix, end, message):
    with pytest.raises(ValueError, match=message):
        sft.build_assistant_labels(ids, prefix, end)


def test_encoding_reuses_template_and_preserves_label_alignment():
    tokenizer = ExampleTokenizer()
    messages = [{"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"}]
    feature = sft.encode_sft_conversation(messages, tokenizer)
    assert tokenizer.messages == messages
    assert tokenizer.tools == []
    assert feature == {"input_ids": [90, 10, 11, 30, 20, 21],
                       "labels": [-100, -100, -100, 30, 20, 21]}
    with pytest.raises(ValueError, match="empty_conversations"):
        sft.encode_sft_conversation([], tokenizer)


def test_encoding_passes_structured_tool_fields_to_template():
    tokenizer = ExampleTokenizer()
    messages = [
        {"role": "system", "content": "", "tools":
         '[{"function":{"name":"lookup","parameters":{"type":"object"}}}]'},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls":
         '[{"function":{"name":"lookup","arguments":"{}"}}]'},
        {"role": "tool", "content": '{"result":"one"}'},
        {"role": "assistant", "content": "answer"},
    ]
    before = deepcopy(messages)
    feature = sft.encode_sft_conversation(messages, tokenizer)
    assert tokenizer.tools[0]["function"]["name"] == "lookup"
    assert "tools" not in tokenizer.messages[0]
    assert tokenizer.messages[2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert tokenizer.messages[3]["content"] == '{"result":"one"}'
    assert feature["labels"] == [-100, -100, -100, 30, 20, 21]
    assert messages == before


def test_collator_preserves_masks_padding_dtype_and_input_records():
    features = [{"input_ids": [90, 10, 30, 20, 21],
                 "labels": [-100, -100, 30, 20, 21]},
                {"input_ids": [90, 30, 20], "labels": [-100, 30, 20]}]
    before = deepcopy(features)
    batch = sft.SFTCollator(ExampleTokenizer(), 5)(features)
    assert all(tensor.shape == (2, 5) and tensor.dtype == torch.long
               for tensor in batch.values())
    assert batch["input_ids"].tolist() == [[90, 10, 30, 20, 21], [90, 30, 20, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["labels"].tolist() == [before[0]["labels"], [-100, 30, 20, -100, -100]]
    assert features == before


@pytest.mark.parametrize("features,message", [
    ([], "features must not be empty"),
    ([{"input_ids": [1, 2], "labels": [2]}], "same length"),
    ([{"input_ids": [1] * 6, "labels": [1] * 6}], "exceed max_length"),
])
def test_collator_rejects_invalid_batch_shapes(features, message):
    with pytest.raises(ValueError, match=message):
        sft.SFTCollator(ExampleTokenizer(), 5)(features)


def test_dataset_passes_indexed_conversation_to_encoder(tmp_path):
    import json
    import pickle
    records = [{"conversations": [{"role": "user", "content": "first"},
                                   {"role": "assistant", "content": "one"}]},
               {"conversations": [{"role": "user", "content": "second"},
                                   {"role": "assistant", "content": "two"}]}]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    tokenizer = ExampleTokenizer()
    dataset = sft.SFTDataset(path, tokenizer)
    assert len(dataset) == 2
    feature = dataset[1]
    assert tokenizer.messages == records[1]["conversations"]
    assert feature == sft.encode_sft_conversation(records[1]["conversations"], tokenizer)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.data._stream is None
    assert restored[0] == dataset[0]


def test_jsonl_reader_preserves_late_optional_fields_and_private_streams(tmp_path):
    import json
    import os
    path = tmp_path / "mixed.jsonl"
    rows = [{"conversations": [{"role": "assistant", "content": "plain"}]},
            {"conversations": [{"role": "assistant", "content": "", "tool_calls": "[]"}]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    data = sft._JsonlRecords(path)
    assert data[0] == rows[0] and data[1] == rows[1]
    previous = data._stream
    data._pid = os.getpid() + 1
    assert data[0] == rows[0]
    assert data._stream is not previous and previous.closed
