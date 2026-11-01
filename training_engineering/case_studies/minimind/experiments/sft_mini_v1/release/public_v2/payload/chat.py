"""Single-turn chat for the native FP32 bundle; no tool execution or remote code fetch."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch


DEFAULT_SYSTEM = '你是一个中文助手。请根据用户请求和已提供的信息回答，不确定或缺少必要信息时如实说明。'


def encode_prompt(tokenizer, prompt, system=DEFAULT_SYSTEM):
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, open_thinking=False)
    return text, tokenizer(text, add_special_tokens=False).input_ids


def check_budget(ids, max_new_tokens, context_length):
    if not ids or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('a nonempty input and positive generation budget are required')
    if len(ids) + max_new_tokens > context_length:
        raise ValueError('prompt plus generation budget exceeds the trained length; no silent truncation')


class CheckedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if not torch.isfinite(logits[:, -1]).all():
            raise FloatingPointError('non-finite next-token logits')
        return logits


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error('prompt must not be blank')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; no fallback')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    bundle = args.bundle_dir.resolve()
    # Load the local package loader before importing its copied minimind_lab runtime.
    spec = importlib.util.spec_from_file_location('native_bundle_loader', bundle/'load_model.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model, tokenizer, config = loader.load_bundle(bundle)
    _, ids = encode_prompt(tokenizer, args.prompt)
    check_budget(ids, args.max_new_tokens, config['trained_sequence_length'])
    from minimind_lab.models.generation import greedy_generate

    model.to(args.device).eval()
    with torch.inference_mode():
        output = greedy_generate(CheckedModel(model), torch.tensor([ids], device=args.device),
                                 args.max_new_tokens, tokenizer.eos_token_id)
    print(tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True))


if __name__ == '__main__':
    main()
