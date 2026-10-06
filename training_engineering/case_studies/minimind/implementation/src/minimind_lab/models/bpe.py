import json

from pathlib import Path

def count_pairs(
    sequences: list[list[int]],
)-> dict[tuple[int, int], int]:
    pair_counts: dict[tuple[int, int], int] = {}
    for sequence in sequences:
        for (a, b) in zip(sequence, sequence[1:]):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    return pair_counts


def merge_pair(
    sequence: list[int],
    pair: tuple[int, int],
    new_token_id:int,
)-> list[int]:
    result = []

    i = 0
    while i < (len(sequence) -1):
        if (sequence[i], sequence[i+1]) == pair:
            result.append(new_token_id)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    
    if i == len(sequence) - 1:
        result.append(sequence[i])
    
    return result

def select_best_pair(pair_counts: dict[tuple[int, int], int],) -> tuple[int, int] | None:
    if not pair_counts:
        return None

    best_pair = max(pair_counts, key= lambda pair: (pair_counts[pair], pair))

    return best_pair


def train_bpe(texts: list[str], vocab_size: int) -> tuple[
        dict[int, bytes],
        dict[tuple[int, int], int]
    ]:
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be at least 256")

    # 0. 定义合并规则
    merges: dict[tuple[int, int], int] = {}

    # 1. 初始词表包含 0～255 的所有单字节 token。
    vocab = {
        token_id : bytes([token_id])
        for token_id in range(256)
    }
    
    # 2. 先将文本编码成序列
    sequences = [
        list(text.encode("utf-8"))
        for text in texts
        ]

    while len(vocab) < vocab_size:
        # 3. 统计序列中出现的字节对
        pair_counts = count_pairs(sequences)

        # 4. 选择出现次数最多的字节对
        best_pair = select_best_pair(pair_counts)

        if best_pair is None:
            break

        # 5. 合并原序列中的token 为 new_token_id
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # 6. 记录合并规则
        merges[(best_pair)] = new_token_id

        # 7. 合并old token id to new token id
        sequences = [
            merge_pair(sequence, best_pair, new_token_id)
            for sequence in sequences
        ]

    return (vocab, merges)

def encode(text:str, merges: dict[tuple[int, int], int]) -> list[int]:
    # 1. 将文本编码成序列
    sequence = list(text.encode("utf-8"))

    # 2. 按照new token id 的从小到大的顺序来依次合并序列中的token
    for pair, new_token_id in sorted(merges.items(), key= lambda item: item[1]):
        sequence = merge_pair(sequence, pair, new_token_id)
    
    return sequence

def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    byte_part = []
    for token_id in token_ids:
        if token_id not in vocab:
            raise ValueError(f"unknown token id: {token_id}")
        
        byte_part.append(vocab[token_id])

    raw_bytes = b"".join(byte_part)
    
    return raw_bytes.decode("utf-8")


def save_bpe(path: str | Path, merges: dict[tuple[int, int], int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merge_records = [
        [pair[0], pair[1], new_token_id]
        for pair, new_token_id 
        in sorted(
            merges.items(), 
            key=lambda item: item[1]
        )
    ]

    save_bped = {
        "format_version": 1,
        "merges": merge_records
    }

    path.write_text(
        json.dumps(
            save_bped, 
            ensure_ascii=False, 
            indent=2)
        + "\n", encoding="utf-8")


def load_bpe(path: str | Path) -> tuple[ 
    dict[int, bytes],
    dict[tuple[int, int], int]
]:    
    path = Path(path)
    load_bped = json.loads(path.read_text(encoding="utf-8"))

    vocab = {
        token_id : bytes([token_id])
        for token_id in range(256)
    }

    if load_bped["format_version"] != 1:
        raise ValueError("unsupported BPE format version")
    
    merges: dict[tuple[int, int], int] = {}
    for merge_record in load_bped["merges"]:
        pair = (merge_record[0], merge_record[1])
        new_token_id = merge_record[2]
        merges[pair] = new_token_id
        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]
    
    return vocab, merges




