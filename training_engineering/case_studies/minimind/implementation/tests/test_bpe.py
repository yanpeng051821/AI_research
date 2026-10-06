
from minimind_lab.models.bpe import count_pairs ,merge_pair, select_best_pair, train_bpe, encode, decode, save_bpe, load_bpe

import json

import pytest


def test_bpe_pair_counts():
    sequences = [
        [97, 98, 97, 98],
        [97, 98],
        [],
        [99],
    ]

    pair_counts = count_pairs(sequences)

    assert pair_counts == {(97, 98): 3, (98, 97): 1}

def test_count_pairs_counts_overlapping_positions():
    assert count_pairs([[1, 1, 1]]) == {(1, 1): 2}


@pytest.mark.parametrize(
    ("sequence", "pair", "new_token_id", "expected"),
    [
        ([97, 98, 97, 98], (97, 98), 256, [256, 256]),
        ([1, 1, 1], (1, 1), 256, [256, 1]),
        ([1, 2, 3], (4, 5), 256, [1, 2, 3]),
        ([], (1, 2), 256, []),
        ([1], (1, 2), 256, [1]),
    ],
)
def test_merge_pair(sequence, pair, new_token_id, expected):
    assert merge_pair(sequence, pair, new_token_id) == expected


def test_merge_pair_does_not_modify_input():
    sequence = [1, 2, 1, 2]

    merge_pair(sequence, (1, 2), 256)

    assert sequence == [1, 2, 1, 2]

@pytest.mark.parametrize(
    ("pair_counts", "expected"),
    [
        ({(97, 98): 3,(98, 97): 1,}, (97, 98)),
        ({(97, 98): 2,(99, 100): 2,}, (99, 100)),
        ( {}, None)
    ]
)
def test_select_best_pair(pair_counts, expected):
    best_pair = select_best_pair(pair_counts)
    assert best_pair == expected


@pytest.mark.parametrize(
    ("texts", "vocab_size", "expected_merges"),
    [
        (["abab", "ab"], 258, {(97, 98): 256, (256, 256): 257,})
    ]
)
def test_train_bpe(texts, vocab_size, expected_merges):
    vocab, merges = train_bpe(texts,vocab_size)

    assert merges == expected_merges
    assert vocab[256] == b"ab"
    assert vocab[257] == b"abab"
    


def test_train_bpe_stops_when_no_pairs_remain():
    vocab, merges = train_bpe(["a", ""], vocab_size=260)

    assert len(vocab) == 256
    assert merges == {}


def test_train_bpe_rejects_vocab_smaller_than_byte_vocab():
    with pytest.raises(
        ValueError,
        match="vocab_size must be at least 256",
    ):
        train_bpe(["hello"], vocab_size=255)

def test_encode_uses_learned_merges_in_order():
    _, merges = train_bpe(
        ["abab", "ab"],
        vocab_size=258,
    )

    assert encode("abab", merges) == [257]
    assert encode("ab", merges) == [256]
    assert encode("aba", merges) == [256, 97]
    assert encode("", merges) == []

def test_decode_reconstructs_merged_token():
    vocab, merges = train_bpe(
        ["abab", "ab"],
        vocab_size=258,
    )

    token_ids = encode("abab", merges)

    assert token_ids == [257]
    assert decode(token_ids, vocab) == "abab"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "你好",
        "hello，世界",
    ],
)
def test_encode_decode_round_trip(text):
    vocab, merges = train_bpe(
        ["你好你好", "hello hello"],
        vocab_size=270,
    )

    token_ids = encode(text, merges)

    assert decode(token_ids, vocab) == text


def test_decode_rejects_unknown_token_id():
    vocab, _ = train_bpe(
        ["hello"],
        vocab_size=256,
    )

    with pytest.raises(
        ValueError,
        match="unknown token id: 999",
    ):
        decode([999], vocab)


def test_save_and_load_bpe_preserves_behavior(
    tmp_path,
):
    vocab, merges = train_bpe(
        ["abab", "ab"],
        vocab_size=258,
    )

    path = tmp_path / "tokenizer.json"
    save_bpe(path, merges)

    loaded_vocab, loaded_merges = load_bpe(path)

    assert loaded_vocab == vocab
    assert loaded_merges == merges

    original_ids = encode("abab", merges)
    loaded_ids = encode("abab", loaded_merges)

    assert loaded_ids == original_ids
    assert decode(loaded_ids, loaded_vocab) == "abab"

def test_save_bpe_writes_versioned_ordered_format(
    tmp_path,
):
    path = tmp_path / "tokenizer.json"
    merges = {
        (256, 256): 257,
        (97, 98): 256,
    }

    save_bpe(path, merges)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert payload == {
        "format_version": 1,
        "merges": [
            [97, 98, 256],
            [256, 256, 257],
        ],
    }

def test_load_bpe_rejects_unknown_version(
    tmp_path,
):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps({
            "format_version": 2,
            "merges": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unsupported BPE format version",
    ):
        load_bpe(path)