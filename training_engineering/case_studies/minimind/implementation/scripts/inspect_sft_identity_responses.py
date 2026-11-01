"""Run three post-hoc S1 identity probes without changing the frozen main evaluation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from evaluate_sft import generate_case, load_weights
from load_exported_pretrain import read_json, sha256_file, state_digest, write_json


QUESTIONS = ['你是谁？', '你是由谁训练的？', '你的预训练权重是从随机初始化开始训练的，还是接续官方预训练权重？']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    marker = args.output_dir / 'run_manifest.json'
    inputs = [args.experiment_dir/'evaluation_recipe/evaluation_config.json',
              args.experiment_dir/'data_recipe/DATA_CONTRACT.json',
              args.experiment_dir/'results/final_run_v1/export_verification.json',
              args.bundle/'export_manifest.json', Path(__file__),
              Path(__file__).with_name('evaluate_sft.py'), Path(__file__).with_name('inspect_sft_eval_draft.py')]
    identity = {str(p.resolve()): sha256_file(p) for p in inputs}
    record = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Post-hoc diagnostic; not a new blind test, not added to the 36 main cases.',
              'inputs_sha256': identity, 'training_performed': False}
    marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required, no silent fallback')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        config = read_json(inputs[0])
        config['status'] = 'posthoc_identity_diagnostic_not_main_evaluation'
        data = read_json(inputs[1])
        device = torch.device('cuda')
        model, tokenizer, exported, _ = load_weights(args.bundle.resolve(), data, device)
        verification = read_json(inputs[2])
        if (verification['status'] != 'passed' or verification['parameter_state_sha256'] != exported['model_state_sha256']
                or verification['export_manifest_sha256'] != sha256_file(args.bundle/'export_manifest.json')):
            raise ValueError('not the independently verified final S1 bundle')
        cases = [{'id': f'identity-{i:02}', 'category': 'identity', 'tier': 'posthoc_diagnostic',
                  'messages': [{'role': 'user', 'content': question}], 'tools': [], 'checks': []}
                 for i, question in enumerate(QUESTIONS, 1)]
        write_json(args.output_dir/'diagnostic_config.json', config)
        rows = []
        with (args.output_dir/'cases.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False)+'\n')
        with (args.output_dir/'generations.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                row = generate_case(model, tokenizer, case, config, device)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                print(json.dumps({'question': case['messages'][0]['content'], 'answer': row['raw_text'],
                                  'stop_reason': row['stop_reason'], 'generated_tokens': row['generated_tokens']},
                                 ensure_ascii=False), flush=True)
        if state_digest(model) != exported['model_state_sha256'] or any(sha256_file(Path(p)) != h for p,h in identity.items()):
            raise ValueError('model or diagnostic inputs changed')
        record.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      records=len(rows), model_state_sha256=exported['model_state_sha256'],
                      files={p.name: sha256_file(p) for p in args.output_dir.iterdir() if p.is_file() and p != marker})
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    except BaseException as exc:
        record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        marker.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
