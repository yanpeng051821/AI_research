"""Offset-indexed datasets for the pinned full-data experiment."""

from array import array
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from minimind_lab.data.pretrain import encode_pretrain_text
from minimind_lab.data.sft import build_assistant_labels, prepare_sft_messages


class OffsetJSONLDataset(Dataset):
    def __init__(self, source_path, offset_path, stage, tokenizer, max_length):
        self.source_path = Path(source_path)
        self.offsets = array("Q")
        with Path(offset_path).open("rb") as handle:
            self.offsets.fromfile(handle, Path(offset_path).stat().st_size // self.offsets.itemsize)
        if Path(offset_path).stat().st_size % self.offsets.itemsize:
            raise ValueError("offset index has a partial entry")
        self.stage = stage
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.limit_records = None
        self.stream = None
        self.stream_pid = None

    def __len__(self):
        return min(len(self.offsets), self.limit_records) if self.limit_records is not None else len(self.offsets)

    def _record(self, index):
        import os

        if self.stream is None or self.stream_pid != os.getpid():
            if self.stream is not None:
                self.stream.close()
            self.stream = self.source_path.open("rb")
            self.stream_pid = os.getpid()
        self.stream.seek(self.offsets[index])
        return json.loads(self.stream.readline())

    def __getitem__(self, index):
        record = self._record(index)
        if self.stage == "pretrain":
            ids = encode_pretrain_text(record["text"], self.tokenizer, self.max_length)
            return {"input_ids": ids}
        messages, tools = prepare_sft_messages(record["conversations"])

        def reject_boundary_literals(value):
            if isinstance(value, str):
                if any(marker and marker in value for marker in (self.tokenizer.bos_token, self.tokenizer.eos_token)):
                    raise ValueError("message_boundary_literal_in_source")
            elif isinstance(value, dict):
                for key, item in value.items():
                    reject_boundary_literals(key)
                    reject_boundary_literals(item)
            elif isinstance(value, list):
                for item in value:
                    reject_boundary_literals(item)

        reject_boundary_literals(messages)
        reject_boundary_literals(tools)
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False
        )
        ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        assistant_prefix = self.tokenizer(
            f"{self.tokenizer.bos_token}assistant\n", add_special_tokens=False
        )["input_ids"]
        message_end = self.tokenizer(
            f"{self.tokenizer.eos_token}\n", add_special_tokens=False
        )["input_ids"]
        labels = build_assistant_labels(ids, assistant_prefix, message_end)
        ids = ids[:self.max_length]
        labels = labels[:self.max_length]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError("audited SFT row lost all causal targets after truncation")
        return {"input_ids": ids, "labels": labels}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(stream=None, stream_pid=None)
        return state

    def __del__(self):
        stream = getattr(self, "stream", None)
        if stream is not None:
            stream.close()


class BlockShuffleSampler(Sampler):
    """Shuffle contiguous source windows to reduce random seeks on E: storage."""

    def __init__(self, size, seed, block_size=1024):
        if size < 0 or block_size <= 0:
            raise ValueError("invalid sampler size or block size")
        self.size = size
        self.block_size = block_size
        self.order = torch.randperm((size + block_size - 1) // block_size,
                                    generator=torch.Generator().manual_seed(seed)).tolist()
        self.position = 0
        self.committed_position = 0

    def __len__(self):
        return self.size - self.position

    def __iter__(self):
        logical = 0
        for block in self.order:
            start = block * self.block_size
            stop = min(self.size, start + self.block_size)
            for index in range(start, stop):
                if logical >= self.position:
                    self.position += 1
                    yield index
                logical += 1

    def commit(self, count):
        if count < 0 or self.committed_position + count > self.position:
            raise ValueError("cannot commit unsampled records")
        self.committed_position += count

    def state_dict(self):
        return {"size": self.size, "block_size": self.block_size,
                "order": self.order.copy(), "committed_position": self.committed_position}

    def load_state_dict(self, state):
        if state.get("size") != self.size or state.get("block_size") != self.block_size:
            raise ValueError("sampler dimensions differ from checkpoint")
        order = state.get("order")
        if sorted(order) != list(range(len(self.order))):
            raise ValueError("checkpoint block order is not a permutation")
        position = state.get("committed_position")
        if type(position) is not int or not 0 <= position <= self.size:
            raise ValueError("invalid committed sampler position")
        self.order = order.copy()
        self.position = position
        self.committed_position = position
