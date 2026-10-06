import torch

from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path


def encode_pretrain_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> list[int]:
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2")

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length-2,
    )

    content_ids = encoding.input_ids
    full_ids = [tokenizer.bos_token_id] + content_ids + [tokenizer.eos_token_id]

    return full_ids


class PretrainDataset(Dataset):
    def __init__(
        self, 
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int) :
        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        self.data = load_dataset(
            "json",
            data_files=str(data_path),
            split="train",
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) :
        text = self.data[index]["text"]

        input_ids = encode_pretrain_text(
            text,
            self.tokenizer,
            self.max_length,
        )

        return {
            "input_ids" : input_ids
        }


class PretrainCollator:
    def __init__(self, 
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
    ) -> None:
        if max_length < 2:
            raise ValueError("max length must be at least 2")

        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")

        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, features: list[dict] ) -> dict[str, torch.Tensor]:
        if features is None or len(features) == 0:
            raise ValueError("features must not be empty")

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for feature in features:
            input_ids = feature["input_ids"]
            if len(input_ids) > self.max_length:
                raise ValueError(f"input_ids exceed max_length")

            padding_length = self.max_length - len(input_ids)
            real_length = len(input_ids)
            # 1. 右侧padding
            input_ids = input_ids + [self.pad_token_id] * padding_length

            # 2. 构造attention_mask
            attention_mask = [1] * real_length + [0] * padding_length

            # 3. 构造labels
            labels = [
                token_id if mask == 1 else -100
                for token_id, mask in zip(input_ids, attention_mask)
            ]

            # 4. 合并batch
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids" : torch.tensor(
                input_ids_batch, 
                dtype=torch.long),
            "attention_mask" : torch.tensor(
                attention_mask_batch, 
                dtype=torch.long),
            "labels" : torch.tensor(
                labels_batch, 
                dtype=torch.long)
        }



