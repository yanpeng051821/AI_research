
import pytest, torch

from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from minimind_lab.data.pretrain import encode_pretrain_text, PretrainDataset, PretrainCollator

TOKENIZER_PATH = (
    Path(__file__).parents[1]
    / "assets"
    / "minimind_tokenizer"
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True,
)

def test_encode_pretrain_text_adds_boundaries():
    text = "你好，MiniMind!"

    result = encode_pretrain_text(
        text, 
        tokenizer, 
        max_length=10
        )

    assert result == [
        1,
        1968, 294, 80, 301,
        108, 80, 916, 36,
        2,
    ]



def test_encode_pretrain_text_reserves_space_before_truncating():
    result = encode_pretrain_text(
        "你好，MiniMind!",
        tokenizer,
        max_length=5,
    )

    assert result == [
        1,
        1968, 294, 80,
        2,
    ]

def test_encode_empty_text_contains_boundaries():
    assert encode_pretrain_text(
        "",
        tokenizer,
        max_length=8,
    ) == [1, 2]


def test_encode_rejects_length_smaller_than_boundaries():
    with pytest.raises(
        ValueError,
        match="max_length must be at least 2",
    ):
        encode_pretrain_text(
            "hello",
            tokenizer,
            max_length=1,
        )



@pytest.fixture
def pretrain_jsonl(tmp_path):
    path = tmp_path / "pretrain.jsonl"
    path.write_text(
        '{"text": "你好，MiniMind!"}\n'
        '{"text": "第二条训练文本"}\n',
        encoding="utf-8",
    )
    return path

def test_pretrain_dataset(pretrain_jsonl):
    path = str(pretrain_jsonl)

    dataset = PretrainDataset(
        path,
        tokenizer,
        10,
    )

    assert len(dataset) == 2
    assert dataset[0]["input_ids"] == [
        1,
        1968, 294, 80, 301,
        108, 80, 916, 36,
        2,
    ]

    dataset_2 = PretrainDataset(
        path,
        tokenizer,
        5,
    )

    assert len(dataset_2[0]["input_ids"]) == 5

    with pytest.raises(ValueError, match="max_length must be at least 2"):
        PretrainDataset(
            path,
            tokenizer,
            1,
        )   

def test_pretrain_collator_pads_and_builds_labels():
    collator = PretrainCollator(tokenizer, max_length=6)

    batch = collator([
        {"input_ids": [1, 10, 11, 2]},
        {"input_ids": [1, 20, 2]},
    ])

    assert torch.equal(
        batch["input_ids"],
        torch.tensor([
            [1, 10, 11, 2, 0, 0],
            [1, 20, 2, 0, 0, 0],
        ]),
    )
    assert torch.equal(
        batch["attention_mask"],
        torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ]),
    )
    assert torch.equal(
        batch["labels"],
        torch.tensor([
            [1, 10, 11, 2, -100, -100],
            [1, 20, 2, -100, -100, -100],
        ]),
    )


def test_pretrain_collator_rejects_empty_features():
    collator = PretrainCollator(tokenizer, max_length=6)

    with pytest.raises(ValueError, match="features must not be empty"):
        collator([])


def test_pretrain_collator_rejects_overlong_sequence():
    collator = PretrainCollator(tokenizer, max_length=6)

    with pytest.raises(ValueError, match="input_ids exceed max_length"):
        collator([
            {"input_ids": [1, 10, 11, 12, 13, 14, 2]},
        ])


def test_pretrain_collator_does_not_mask_real_pad_token_id():
    collator = PretrainCollator(tokenizer, max_length=5)

    batch = collator([
        {"input_ids": [1, 0, 2]},
    ])

    assert torch.equal(
        batch["attention_mask"],
        torch.tensor([[1, 1, 1, 0, 0]]),
    )
    assert torch.equal(
        batch["labels"],
        torch.tensor([[1, 0, 2, -100, -100]]),
    )

def test_dataloader_builds_pretrain_batch(pretrain_jsonl):
    pretrain_jsonl = str(pretrain_jsonl)
    dataset = PretrainDataset(
        pretrain_jsonl,
        tokenizer,
        max_length=8,
    )
    collator = PretrainCollator(
        tokenizer,
        max_length=8,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(dataloader))

    assert torch.equal(
        batch["input_ids"],
        torch.tensor([
            [1, 1968, 294, 80, 301, 108, 80, 2],
            [1, 4337, 1396, 1857, 1896, 2, 0, 0],
        ]),
    )
    assert torch.equal(
        batch["attention_mask"],
        torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]),
    )
    assert torch.equal(
        batch["labels"],
        torch.tensor([
            [1, 1968, 294, 80, 301, 108, 80, 2],
            [1, 4337, 1396, 1857, 1896, 2, -100, -100],
        ]),
    )