import json

from pathlib import Path
from transformers import AutoTokenizer

from minimind_lab.data.sft import build_assistant_labels

sample_path = Path("data/source_samples/sft_sample.jsonl")

tokenizer = AutoTokenizer.from_pretrained("./assets/minimind_tokenizer", local_files_only=True)

with sample_path.open("r", encoding='utf-8') as f:
    first_line = f.readline()
    record = json.loads(first_line)
    assistant_prefix = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    message_end       = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    
    messages = record["conversations"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    input_ids = tokenizer(text, add_special_tokens=False).input_ids

    print(text)
    print("token count:", len(input_ids))

    labels = build_assistant_labels(input_ids, assistant_prefix, message_end)

    print("supervised targets:", sum(label != -100 for label in labels))
    print("token count:", len(input_ids))

    for index, token_id in enumerate(input_ids):
        piece = tokenizer.decode([token_id])
        print(index, token_id, repr(piece), "label:", labels[index])
















