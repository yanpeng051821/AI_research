from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_PATH = (
    Path(__file__).parents[1]
    / "assets"
    / "minimind_tokenizer"
)


def test_official_tokenizer_contract():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    assert len(tokenizer) == 6400
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0


def test_official_tokenizer_does_not_add_boundaries():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    text = "你好，MiniMind!"

    default_ids = tokenizer(text).input_ids
    plain_ids = tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    assert default_ids == plain_ids
    assert tokenizer.decode(plain_ids) == text