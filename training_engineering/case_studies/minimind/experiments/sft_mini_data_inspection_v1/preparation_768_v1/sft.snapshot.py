from copy import deepcopy
import json

import torch

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path

from minimind_lab.data.validator import validate_sft_record
from minimind_lab.data.sft_protocol import check_sft_protocol


def _parse_tool_list(value: object, location: str) -> list:
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location} must contain valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list or JSON list string")
    return value


def prepare_sft_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Copy validated messages and adapt tool fields to the local chat template.

    This checks representational structure, not tool availability, argument-schema
    compliance, content quality, or permission to include a record in training.
    """
    issues = validate_sft_record({"conversations": messages})
    if issues:
        raise ValueError(f"invalid SFT conversation: {issues}")

    prepared = deepcopy(messages)
    tools = []
    for index, message in enumerate(prepared):
        location = f"messages[{index}]"
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f"{location}.reasoning_content must be a string or null")

        declarations = _parse_tool_list(message.pop("tools", None), f"{location}.tools")
        if declarations:
            if index != 0 or message["role"] != "system":
                raise ValueError("nonempty tools must belong to the first system message")
            for tool_index, tool in enumerate(declarations):
                tool_location = f"{location}.tools[{tool_index}]"
                function = tool.get("function") if isinstance(tool, dict) else None
                if not isinstance(function, dict):
                    raise ValueError(f"{tool_location}.function must be an object")
                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"{tool_location}.function.name must be a nonempty string")
                if "parameters" in function and not isinstance(function["parameters"], dict):
                    raise ValueError(f"{tool_location}.function.parameters must be an object")
            tools = declarations

        if "tool_calls" not in message:
            continue
        calls = _parse_tool_list(message["tool_calls"], f"{location}.tool_calls")
        if calls and message["role"] != "assistant":
            raise ValueError(f"{location}.tool_calls requires the assistant role")
        for call_index, call in enumerate(calls):
            call_location = f"{location}.tool_calls[{call_index}]"
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                raise ValueError(f"{call_location}.function must be an object")
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{call_location}.function.name must be a nonempty string")
            if "arguments" not in function:
                raise ValueError(f"{call_location}.function.arguments is required")
            arguments = function["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{call_location}.function.arguments must contain valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"{call_location}.function.arguments must be an object or JSON object string")
            # The template supports JSON strings; preserve their exact formatting.
        message["tool_calls"] = calls

    return prepared, tools


def build_assistant_labels(
    input_ids: list[int],
    assistant_prefix: list[int],
    message_end: list[int],
) -> list[int]:
    if not assistant_prefix:
        raise ValueError("assistant_prefix must not be empty")

    if not message_end:
        raise ValueError("message_end must not be empty")
    
    labels = [-100] * len(input_ids)

    assistant_spans = []

    for i in range(len(input_ids)-len(assistant_prefix)+1):
        if input_ids[i:i+len(assistant_prefix)] == assistant_prefix:
            body_start = i + len(assistant_prefix)
            body_end = None

            for j in range(body_start, len(input_ids)-len(message_end)+1):
                if input_ids[j:j+len(message_end)] == message_end:
                    body_end = j + len(message_end)
                    break
            
            if body_end is None:
                raise ValueError("assistant message end not found")

            assistant_spans.append([body_start, body_end])
            for j in range(body_start, body_end):
                labels[j] = input_ids[j]

    if not assistant_spans:
        raise ValueError("no assistant message found")

    return labels

def encode_sft_conversation(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[int]]:
    prepared_messages, tools = prepare_sft_messages(messages)

    def reject_boundary_literals(value):
        if isinstance(value, str):
            if any(marker and marker in value for marker in (tokenizer.bos_token, tokenizer.eos_token)):
                raise ValueError("message_boundary_literal_in_source")
        elif isinstance(value, dict):
            for key, item in value.items():
                reject_boundary_literals(key)
                reject_boundary_literals(item)
        elif isinstance(value, list):
            for item in value:
                reject_boundary_literals(item)

    reject_boundary_literals(prepared_messages)
    reject_boundary_literals(tools)
    protocol_issues, _ = check_sft_protocol(prepared_messages, tools)
    if protocol_issues:
        raise ValueError(f"invalid_sft_protocol: {protocol_issues[:5]}")

    text = tokenizer.apply_chat_template(
        prepared_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False
    )

    input_ids = tokenizer(text, add_special_tokens=False).input_ids

    assistant_prefix = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    message_end       = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    labels = build_assistant_labels(input_ids, assistant_prefix, message_end)
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("zero_targets_after_shift")
    
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class SFTCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        if max_length < 2:
            raise ValueError("max length must be at least 2")

        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        if features is None or len(features) == 0:
            raise ValueError("features must not be empty")

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]

            if len(input_ids) != len(labels):
                raise ValueError("input_ids and labels must have the same length")

            if len(input_ids) > self.max_length:
                raise ValueError("input_ids exceed max_length")
            if not any(label != -100 for label in labels[1:]):
                raise ValueError("zero_targets_after_shift")
            
            padding_length = self.max_length - len(input_ids)
            real_length = len(input_ids)

            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = [1] * real_length + [0] * padding_length
            labels = labels + [-100] * padding_length

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.data = load_dataset(
            "json",
            data_files=str(data_path),
            split="train",
        )
        self.tokenizer = tokenizer
        

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.data[index]["conversations"]
        encoding = encode_sft_conversation(messages, self.tokenizer)

        return {
            "input_ids": encoding["input_ids"],
            "labels": encoding["labels"],
        }
